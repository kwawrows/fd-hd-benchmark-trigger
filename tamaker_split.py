"""
SWIFT Trigger Activity (TA) builder for DUNE-HD.

Takes a pickle DataFrame of Trigger Primitives (TPs) and produces TAs
by sliding a time window over each readout plane, then running DBSCAN
on windows that fall in the 'inspect' ADC band.

TA flag meanings:
    2  accepted (window ADC above accept_threshold, or largest cluster above cluster_cut)
    1  inspect  (window ADC above inspect_threshold — intermediate, resolved by clustering)
    0  rejected (inspect window that failed the cluster cut)

Only flag-1 and flag-2 TAs are written to the output; within flag-1 cases,
the flag is updated to 0 or 2 after clustering, so a flag-0 row means
"was inspected but didn't make the cut".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm


# --------------------------------------------------------------------------
# Physical constants
# --------------------------------------------------------------------------

WIRE_PITCH    = 0.48   # cm per channel
DRIFT_VEL     = 0.16   # cm / µs
SAMPLING_RATE = 0.5    # µs / tick
CM_PER_TICK   = DRIFT_VEL * SAMPLING_RATE   # 0.08 cm / tick

TICKS_PER_READOUT = 6000
DEFAULT_WINDOW    = 1000  # ticks

_IND_ANGLES  = {0: np.radians(-35.7), 1: np.radians(35.7)}
_KNN_NEIGHBORS = 10


# --------------------------------------------------------------------------
# DBSCAN (Numba)
# --------------------------------------------------------------------------

@njit
def _dbscan(z, t, eps, min_samples):
    N = len(z)
    labels  = -np.ones(N, dtype=np.int32)
    visited = np.zeros(N, dtype=np.uint8)
    eps2 = eps * eps
    cluster_id = 0

    for i in range(N):
        if visited[i]:
            continue
        visited[i] = 1

        neighbours = []
        for j in range(N):
            dz = z[j] - z[i]
            dt = t[j] - t[i]
            if dz*dz + dt*dt <= eps2:
                neighbours.append(j)

        if len(neighbours) < min_samples:
            continue

        labels[i] = cluster_id
        k = 0
        while k < len(neighbours):
            j = neighbours[k]
            if not visited[j]:
                visited[j] = 1
                local = []
                for m in range(N):
                    dz = z[m] - z[j]
                    dt = t[m] - t[j]
                    if dz*dz + dt*dt <= eps2:
                        local.append(m)
                if len(local) >= min_samples:
                    for m in local:
                        if m not in neighbours:
                            neighbours.append(m)
            if labels[j] == -1:
                labels[j] = cluster_id
            k += 1

        cluster_id += 1

    return labels


def cluster_tps(ch, t, q, epsilon=2.0, min_samples=2):
    """Run DBSCAN on a window of TPs, working in physical units (cm).

    Returns (summary, labels, cluster_ids, cluster_sums) where summary is
    (n_clusters, mean_energy, total_energy, max_energy).
    """
    z_cm = np.asarray(ch, dtype=np.float32) * WIRE_PITCH
    t_cm = np.asarray(t,  dtype=np.float32) * CM_PER_TICK
    labels = _dbscan(z_cm, t_cm, eps=epsilon, min_samples=min_samples)

    valid = labels != -1
    if not np.any(valid):
        return (0, 0.0, 0.0, 0.0), labels, np.empty(0, np.int32), np.empty(0, np.float32)

    q = np.asarray(q)
    cluster_ids  = np.unique(labels[valid])
    cluster_sums = np.array([q[labels == c].sum() for c in cluster_ids])
    summary = (len(cluster_ids), cluster_sums.mean(), cluster_sums.sum(), cluster_sums.max())
    return summary, labels, cluster_ids, cluster_sums


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _filter_tps(df, peak_adc_cut, tot_cut):
    return df[(df["adc_peak"] > peak_adc_cut) & (df["samples_over_threshold"] > tot_cut)]


def _tp_moments(ch, t, q):
    """Charge-weighted moments and covariance sufficient statistics."""
    ch_mean, t_mean = ch.mean(), t.mean()
    ch_std,  t_std  = ch.std(),  t.std()

    W = q.sum()
    if W > 0:
        ch_wmean  = np.average(ch, weights=q)
        t_wmean   = np.average(t,  weights=q)
        dch       = ch - ch_wmean
        dt        = t  - t_wmean
        ch_wstd   = float(np.sqrt(np.average(dch**2, weights=q)))
        t_wstd    = float(np.sqrt(np.average(dt**2,  weights=q)))
        ch_t_wcov = float(np.average(dch * dt, weights=q))
        ch_p01, ch_p99 = np.percentile(ch, [0.01, 99.9])
        t_p01,  t_p99  = np.percentile(t,  [0.01, 99.9])
    else:
        ch_wmean = t_wmean = ch_wstd = t_wstd = -999.0
        ch_t_wcov = -999.0
        ch_p01 = ch_p99 = t_p01 = t_p99 = -999.0

    return dict(
        ch_mean=float(ch_mean),     t_mean=float(t_mean),
        ch_std=float(ch_std),       t_std=float(t_std),
        ch_wmean=float(ch_wmean),   t_wmean=float(t_wmean),
        ch_wstd=ch_wstd,            t_wstd=t_wstd,
        ch_t_wcov=ch_t_wcov,
        ch_p01=float(ch_p01),       ch_p99=float(ch_p99),
        t_p01=float(t_p01),         t_p99=float(t_p99),
        W=float(W),
    )


def _tp_moments_truth(bt_x, bt_y, bt_z, q):
    """Charge-weighted truth moments for validation."""
    W = q.sum()
    if W <= 0 or len(q) < 2:
        return None
    mu_x = np.average(bt_x, weights=q)
    mu_y = np.average(bt_y, weights=q)
    mu_z = np.average(bt_z, weights=q)
    dx, dy, dz = bt_x - mu_x, bt_y - mu_y, bt_z - mu_z
    return dict(
        bt_W=float(W),
        bt_mu_x=float(mu_x),        bt_mu_y=float(mu_y),        bt_mu_z=float(mu_z),
        bt_Cxx=float((q*dx*dx).sum()), bt_Cyy=float((q*dy*dy).sum()), bt_Czz=float((q*dz*dz).sum()),
        bt_Cxy=float((q*dx*dy).sum()), bt_Cxz=float((q*dx*dz).sum()), bt_Cyz=float((q*dy*dz).sum()),
        bt_x_p01=float(np.percentile(bt_x, 0.01)), bt_x_p99=float(np.percentile(bt_x, 99.9)),
        bt_y_p01=float(np.percentile(bt_y, 0.01)), bt_y_p99=float(np.percentile(bt_y, 99.9)),
        bt_z_p01=float(np.percentile(bt_z, 0.01)), bt_z_p99=float(np.percentile(bt_z, 99.9)),
    )


_NULL_MOMENTS = {k: -1.0 for k in [
    "ch_mean", "t_mean", "ch_std", "t_std",
    "ch_wmean", "t_wmean", "ch_wstd", "t_wstd",
    "ch_t_wcov", "ch_p01", "ch_p99", "t_p01", "t_p99", "W",
]}

_NULL_TRUTH = {k: -1.0 for k in [
    "bt_W", "bt_mu_x", "bt_mu_y", "bt_mu_z",
    "bt_Cxx", "bt_Cyy", "bt_Czz", "bt_Cxy", "bt_Cxz", "bt_Cyz",
    "bt_x_p01", "bt_x_p99", "bt_y_p01", "bt_y_p99", "bt_z_p01", "bt_z_p99",
]}


def _split_induction_groups(ch, t, plane):
    """Split induction TPs into two groups by local orientation alignment.

    Works in physical units (cm). Returns a boolean mask (True = group 0,
    False = group 1), or None if no confident split is found (one group
    would be < 10% of TPs).
    """
    from sklearn.neighbors import NearestNeighbors

    pts = np.column_stack([t * CM_PER_TICK, ch * WIRE_PITCH])
    n   = len(pts)
    k   = min(_KNN_NEIGHBORS, n - 1)
    if k < 2:
        return None

    _, indices = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(pts).kneighbors(pts)

    theta  = _IND_ANGLES[plane]
    u1     = np.array([np.cos(theta),  np.sin(theta)])
    u2     = np.array([np.cos(theta), -np.sin(theta)])
    v_diff = pts[indices] - pts[:, np.newaxis, :]
    mask   = np.var(np.tensordot(v_diff, u1, axes=([2], [0])), axis=1) > \
             np.var(np.tensordot(v_diff, u2, axes=([2], [0])), axis=1)

    frac = mask.mean()
    if frac < 0.10 or frac > 0.90:
        return None
    return mask


# --------------------------------------------------------------------------
# TAMaker
# --------------------------------------------------------------------------

class TAMaker:
    """Build Trigger Activities from a Trigger Primitive DataFrame.

    Basic usage::

        from tamaker import TAMaker
        from presets import COLLECTION_CENTRAL

        maker = TAMaker(**COLLECTION_CENTRAL)
        tas, tps = maker.run(df)

    Parameters
    ----------
    plane               Wire view to process: 0=U, 1=V, 2=collection.
    window_size         Sliding window width in readout ticks.
    inspect_threshold   Minimum window ADC to enter DBSCAN inspection.
    accept_threshold    Window ADC threshold for immediate acceptance (no clustering).
    cluster_cut         Largest-cluster ADC threshold for accept/reject decision.
    db_epsilon          DBSCAN neighbourhood radius in cm.
    db_min_samples      DBSCAN minimum points per cluster.
    peak_adc_cut        Pre-filter: minimum peak ADC per TP.
    tot_cut             Pre-filter: minimum samples-over-threshold per TP.
    """

    def __init__(
        self,
        plane             = 2,
        window_size       = DEFAULT_WINDOW,
        inspect_threshold = 13e3,
        accept_threshold  = 55e3,
        cluster_cut       = 22e3,
        db_epsilon        = 2.0,
        db_min_samples    = 2,
        peak_adc_cut      = 80.0,
        tot_cut           = 8,
    ):
        self.plane             = plane
        self.window_size       = window_size
        self.inspect_threshold = inspect_threshold
        self.accept_threshold  = accept_threshold
        self.cluster_cut       = cluster_cut
        self.db_epsilon        = db_epsilon
        self.db_min_samples    = db_min_samples
        self.peak_adc_cut      = peak_adc_cut
        self.tot_cut           = tot_cut

    def run(self, df_in, global_ta_offset=0):
        """Process a TP DataFrame and return (tas, tps).

        Parameters
        ----------
        df_in            Raw TP DataFrame (all planes, all events).
        global_ta_offset Starting TA_id — useful when stitching multiple runs.

        Returns
        -------
        tas   DataFrame with one row per Trigger Activity.
        tps   Filtered TP DataFrame with TA_id and dbscan_label columns added.
        """
        df = self._prepare(df_in)
        windows = self._build_windows(df)
        windows = self._assign_ids(windows, global_ta_offset)

        ch_arr    = df["channel"].to_numpy()
        t_arr     = df["time_start"].to_numpy()
        q_arr     = df["adc_integral"].to_numpy()
        index_map = self._build_index_map(df)

        ta_records, label_records = self._process_windows(windows, index_map, ch_arr, t_arr, q_arr, df)

        tas = pd.DataFrame(ta_records)
        tps = self._annotate_tps(df, label_records, windows)
        return tas, tps

    def run_by_run(self, df_in, output_ta=None, output_tp=None):
        """Same as run() but processes one run at a time to keep memory usage low.

        Handy for large datasets. Pass output_ta / output_tp file paths to
        save the merged results directly to disk.
        """
        runs = np.sort(df_in["run"].unique())
        all_tas, all_tps = [], []
        ta_offset = 0

        for run in tqdm(runs, desc="Runs"):
            df_run = df_in[df_in["run"] == run]
            if df_run.empty:
                continue
            tas_r, tps_r = self.run(df_run, global_ta_offset=ta_offset)
            if not tas_r.empty:
                ta_offset = int(tas_r["TA_id"].max()) + 1
            tas_r["run"] = run
            tps_r["run"] = run
            all_tas.append(tas_r)
            all_tps.append(tps_r)

        tas = pd.concat(all_tas, ignore_index=True)
        tps = pd.concat(all_tps, ignore_index=True)

        if output_ta:
            tas.to_pickle(output_ta)
        if output_tp:
            tps.to_pickle(output_tp)

        return tas, tps

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _prepare(self, df_in):
        df = _filter_tps(df_in, self.peak_adc_cut, self.tot_cut)
        df = df[df["readout_view"] == self.plane].copy()
        df["time_start"] = df["time_start"] / 32
        df["apa_id"] = df["TPCSetID"].to_numpy()
        df["rop"]    = df["readout_plane_id"].to_numpy()

        bins = np.arange(0, TICKS_PER_READOUT + 1, self.window_size)
        df["bin"] = np.digitize(df["time_start"], bins) - 1
        return df

    def _build_windows(self, df):
        windows = (
            df.groupby(["event", "run", "subrun", "bin", "rop", "apa_id"]).agg(
                total_window_energy=("adc_integral", "sum"),
                TP_count=("adc_integral", "size")).reset_index())
        windows["flag"] = 0
        mask_inspect = (windows["total_window_energy"] > self.inspect_threshold) & (windows["TP_count"] > 1)
        windows.loc[mask_inspect, "flag"] = 1
        windows.loc[windows["total_window_energy"] > self.accept_threshold, "flag"] = 2
        return windows

    def _assign_ids(self, windows, offset):
        windows = windows.sort_values(["event", "apa_id", "rop", "bin"]).reset_index(drop=True)
        windows["TA_id"] = np.arange(len(windows)) + offset
        return windows

    def _build_index_map(self, df):
        from collections import defaultdict
        index_map = defaultdict(list)
        for i, (ev, apa, rop, b) in enumerate(zip(
                df["event"].to_numpy(), df["apa_id"].to_numpy(),
                df["rop"].to_numpy(),   df["bin"].to_numpy())):
            index_map[(ev, apa, rop, b)].append(i)
        return {k: np.array(v) for k, v in index_map.items()}

    def _process_windows(self, windows, index_map, ch_arr, t_arr, q_arr, df):
        ta_records    = []
        label_records = []
        is_induction  = self.plane in _IND_ANGLES

        truth_arrays = None
        if all(c in df.columns for c in ["bt_x", "bt_y", "bt_z", "bt_edep"]):
            truth_arrays = (
                df["bt_x"].to_numpy(), df["bt_y"].to_numpy(),
                df["bt_z"].to_numpy(), df["bt_edep"].to_numpy(),
            )

        def get_truth(idx):
            if truth_arrays is None or len(idx) == 0:
                return _NULL_TRUTH
            bt_x, bt_y, bt_z, bt_edep = truth_arrays
            mask = bt_edep[idx] > 0
            if mask.sum() < 2:
                return _NULL_TRUTH
            result = _tp_moments_truth(bt_x[idx][mask], bt_y[idx][mask],
                                       bt_z[idx][mask], q_arr[idx][mask])
            return result if result is not None else _NULL_TRUTH

        for row in windows[windows["flag"] == 2].itertuples(index=False):
            idx = index_map.get((row.event, row.apa_id, row.rop, row.bin), np.array([], int))
            if not len(idx):
                ta_records.append(self._make_record(row, flag=2, n_cl=-1, mean_cl=-1,
                                                     tot_cl=-1, max_cl=-1,
                                                     moments=_NULL_MOMENTS, truth=_NULL_TRUTH))
                continue

            if is_induction:
                group_mask = _split_induction_groups(ch_arr[idx], t_arr[idx], self.plane)
            else:
                group_mask = None

            if group_mask is not None:
                for face, sel in enumerate([group_mask, ~group_mask]):
                    ta_records.append(self._make_record(row, flag=2, n_cl=-1, mean_cl=-1,
                                                         tot_cl=-1, max_cl=-1,
                                                         moments=_tp_moments(ch_arr[idx][sel], t_arr[idx][sel], q_arr[idx][sel]),
                                                         truth=get_truth(idx[sel])))
            else:
                ta_records.append(self._make_record(row, flag=2, n_cl=-1, mean_cl=-1,
                                                     tot_cl=-1, max_cl=-1,
                                                     moments=_tp_moments(ch_arr[idx], t_arr[idx], q_arr[idx]),
                                                     truth=get_truth(idx)))

        for row in tqdm(list(windows[windows["flag"] == 1].itertuples(index=False)), desc="DBSCAN inspect"):
            idx = index_map.get((row.event, row.apa_id, row.rop, row.bin), np.array([], int))
            if len(idx) == 0:
                continue

            (n_cl, mean_cl, tot_cl, max_cl), lbls, cids, csums = cluster_tps(
                ch_arr[idx], t_arr[idx], q_arr[idx],
                epsilon=self.db_epsilon, min_samples=self.db_min_samples,
            )
            flag = 2 if max_cl > self.cluster_cut else 0

            moments = _NULL_MOMENTS
            truth   = _NULL_TRUTH
            if n_cl > 0:
                sel_idx = idx[lbls == cids[np.argmax(csums)]]
                moments = _tp_moments(ch_arr[sel_idx], t_arr[sel_idx], q_arr[sel_idx])
                truth   = get_truth(sel_idx)

            ta_records.append(self._make_record(row, flag=flag, n_cl=n_cl, mean_cl=mean_cl,
                                                 tot_cl=tot_cl, max_cl=max_cl,
                                                 moments=moments, truth=truth))
            label_records.append(pd.DataFrame({"index": idx, "dbscan_label": lbls}))

        return ta_records, label_records

    def _annotate_tps(self, df, label_records, windows):
        df = df.copy()
        df["dbscan_label"] = -1
        df["window_start"] = df["bin"] * self.window_size

        if label_records:
            all_labels = pd.concat(label_records, ignore_index=True)
            labels_col = df["dbscan_label"].to_numpy().copy()
            labels_col[all_labels["index"].to_numpy()] = all_labels["dbscan_label"].to_numpy()
            df["dbscan_label"] = labels_col

        ta_index = windows.copy()
        ta_index["window_start"] = ta_index["bin"] * self.window_size
        ta_index = ta_index[["event", "run", "subrun", "apa_id", "window_start", "rop", "TA_id"]]

        return df.merge(ta_index, on=["event", "apa_id", "window_start", "rop"], how="inner")

    def _make_record(self, row, flag, n_cl, mean_cl, tot_cl, max_cl, moments, truth):
        return {
            "event":                row.event,
            "run":                  row.run,
            "subrun":               row.subrun,
            "apa_id":               row.apa_id,
            "rop":                  row.rop,
            "window_start":         row.bin * self.window_size,
            "flag":                 flag,
            "TA_id":                row.TA_id,
            "total_window_energy":  row.total_window_energy,
            "TP_count":             row.TP_count,
            "n_clusters":           n_cl,
            "mean_cluster_energy":  mean_cl,
            "total_cluster_energy": tot_cl,
            "max_cluster_energy":   max_cl,
            **moments,
            **truth,
        }
