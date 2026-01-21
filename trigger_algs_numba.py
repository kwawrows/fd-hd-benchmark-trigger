import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed 
from numba import njit


# NOTE: there are many hard-coded things for the HD here! This can't be used for any geometry out-of-the-box. 

@njit
def numba_dbscan(z, t, eps, min_samples):

    #Very simplified but very speedy DBScan implementation 

    N = len(z) #number of TPs in window 
    labels = -1 * np.ones(N, dtype=np.int32) #initialise labels to noise 
    visited = np.zeros(N, dtype=np.uint8)
    cluster_id = 0
    eps2 = eps * eps

    for i in range(N):
        if visited[i]:
            continue

        visited[i] = 1

        neigh = [] #neighbours
        zi, ti = z[i], t[i] #effective z and time

        for j in range(N):
            dz = z[j] - zi
            dt = t[j] - ti
            if dz*dz + dt*dt <= eps2:
                neigh.append(j)

        if len(neigh) < min_samples:
            labels[i] = -1
        else:
            labels[i] = cluster_id

            k = 0
            while k < len(neigh):
                j = neigh[k]

                if not visited[j]:
                    visited[j] = 1

                    # Expanding neighbours of j
                    zj, tj = z[j], t[j]
                    neigh2 = []
                    for m in range(N):
                        dz = z[m] - zj
                        dt = t[m] - tj
                        if dz*dz + dt*dt <= eps2:
                            neigh2.append(m)

                    if len(neigh2) >= min_samples:
                        for m in neigh2:
                            if m not in neigh:
                                neigh.append(m)

                if labels[j] == -1:
                    labels[j] = cluster_id

                k += 1

            cluster_id += 1

    return labels


def apply_dbscan(window_tps, epsilon=2, min_samples=2):

    # conversion parameters to move from channel ID & ticks to cm. 
    wire_pitch = 0.48 #cm 
    drift_velocity = 0.16 #cm/us
    sampling_rate = 0.5 #us/tick 
    cm_per_tick = drift_velocity * sampling_rate

    df = window_tps

    # Numba friendly arrays
    time_cm = (df["time_start"].to_numpy() * cm_per_tick).astype(np.float32)
    z = (df["channel"].to_numpy() * wire_pitch).astype(np.float32)
    adc = df["adc_integral"].to_numpy()
    df_index = df.index.to_numpy()

    labels = numba_dbscan(z, time_cm, eps=epsilon, min_samples=min_samples) 

    #  cluster energies
    valid = labels != -1
    if not np.any(valid):
        return (0, 0, 0, 0), df_index, labels

    cluster_ids = np.unique(labels[valid])
    cluster_sums = np.array([adc[labels == cid].sum() for cid in cluster_ids])

    summary = (
        len(cluster_ids),
        cluster_sums.mean(),
        cluster_sums.sum(),
        cluster_sums.max()
    )

    return summary, df_index, labels


# Helper function for getting TPC ID from TPCSet and readout_plane_id -->meant for collection only 
def get_tpcid(apa_id, plane_id):
    return apa_id * 2 + (plane_id - 2)

#TP refinement 
def apply_tp_filter(data, peak_adc_cut_=80, tot_cut_=8):
    return data[(data.adc_peak > peak_adc_cut_) & (data.samples_over_threshold > tot_cut_)]

#channel masking in z 
def mask_edge_col_channels(data, edge_cut=80):
    col_map = pd.DataFrame( np.loadtxt("./../YesOrNoTrigger/data/ViewChannelMatch_dunefd_1x2x6_full.txt", delimiter=','),columns=['col_ch','ind_ch','ind_plane','y','z'])

    valid_channels = col_map[(col_map.z < col_map.z.max() - edge_cut) & (col_map.z > col_map.z.min() + edge_cut)].col_ch

    return data[(data.readout_view == 2) & data.channel.isin(valid_channels)]



# Main TAMaker
def TAMaker(data, 
            window_size=1000, inspect_threshold=15e3, accept_threshold=55e3, #windowing + categorisation 
            cluster_cut=22e3, db_epsilon=2, db_min_samples=2, #dbscan config
            mask_channels=False, edge_cut=80, #channel masking in z
            peakadc_cut=80, tot_cut=8, #tp refinement params 
            n_threads=12, global_ta_offset=0, # global TA offset needed to get unique TA IDs for datasets with multiple subruns (repeated event IDs)
            ): 

    #TP refinement 
    df = apply_tp_filter(data, peak_adc_cut_=peakadc_cut, tot_cut_=tot_cut)

    #Channel masking in z
    if mask_channels:
        df = mask_edge_col_channels(df, edge_cut=edge_cut)

    df = df[df.readout_view == 2].copy()
    df["time_start"] /= 32
    df["tpc"] = get_tpcid(df.TPCSetID.to_numpy(), df.readout_plane_id.to_numpy())

    # binning and energy estimation 
    bins = np.arange(0, 6000 + 1, window_size)
    df["bin"] = np.digitize(df["time_start"], bins) - 1
    window_summary = (df.groupby(["event", "tpc", "bin"]).agg(total_window_energy=("adc_integral", "sum"),TP_count=("adc_integral", "size")).reset_index())


    #window categorisation 
    window_summary["flag"] = 0
    window_summary.loc[window_summary.total_window_energy > inspect_threshold, "flag"] = 1
    window_summary.loc[window_summary.total_window_energy > accept_threshold, "flag"] = 2

    #add TA IDs. 
    window_summary = window_summary.sort_values(["event", "tpc", "bin"]).reset_index(drop=True)
    window_summary["TA_id"] = np.arange(len(window_summary)) + global_ta_offset

    # Immediate accept windows
    results = []
    immediate_accept = window_summary[window_summary.flag == 2]

    for _, row in immediate_accept.iterrows():
        results.append({
            "event": row.event,
            "tpc": row.tpc,
            "window_start": row.bin * window_size,
            "flag": row.flag,
            "TA_id": row.TA_id,
            "total_window_energy": row.total_window_energy,
            "TP_count": row.TP_count,
            "n_clusters": -1,
            "mean_cluster_energy": -1,
            "total_cluster_energy": -1,
            "max_cluster_energy": -1
        })

    # DBSCAN for inspect windows
    inspect_windows = window_summary[window_summary.flag == 1]
    label_records = []

    def process_window(row):
        hits = df[(df.event == row.event) & (df.tpc == row.tpc) & (df.bin == row.bin)]
        summary, idx, lbls = apply_dbscan(hits, epsilon=db_epsilon, min_samples=db_min_samples)
        n_cl, mean_cl, tot_cl, max_cl = summary
        flag = 2 if max_cl > cluster_cut else 0

        return {
            "summary": {
                "event": row.event,
                "tpc": row.tpc,
                "window_start": row.bin * window_size,
                "flag": flag,
                "TA_id": row.TA_id,
                "total_window_energy": row.total_window_energy,
                "TP_count": row.TP_count,
                "n_clusters": n_cl,
                "mean_cluster_energy": mean_cl,
                "total_cluster_energy": tot_cl,
                "max_cluster_energy": max_cl
            },
            "tp_idx": idx,
            "labels": lbls
        }

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = [ex.submit(process_window, row) for _, row in inspect_windows.iterrows()]
        for f in tqdm(as_completed(futures), total=len(futures)):
            out = f.result()
            results.append(out["summary"])
            label_records.append(pd.DataFrame({"index": out["tp_idx"], "dbscan_label": out["labels"]}))
    tas = pd.DataFrame(results)

    # Merging DBSCAN labels back into TP dataframe
    df["dbscan_label"] = -1
    if label_records:
        all_labels = pd.concat(label_records, ignore_index=True).set_index("index")
        df.loc[all_labels.index, "dbscan_label"] = all_labels["dbscan_label"]

    df["window_start"] = df["bin"] * window_size

    tps = df.merge(tas[["event", "tpc", "window_start", "TA_id"]],on=["event", "tpc", "window_start"],how="inner" )

    return tas, tps
