import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import trigger_algs_numba as trg_numba
import utils as util
import os
import glob

from sklearn.cluster import DBSCAN
from tqdm import tqdm


cols_to_keep = ['event','run', 'channel', 'samples_over_threshold', 'time_start','adc_integral', 'adc_peak', 'readout_plane_id', 'readout_view', 'TPCSetID','bt_edep', 'bt_x', 'bt_y', 'bt_z', 'bt_generator_name', 'event_uid', 'bt_numelectrons', 'bt_primary_track_id']


def get_unique_event_ids(df):
    df["event_uid"] = ( df["run"].astype("int64").to_numpy()    << 32 | df["subrun"].astype("int64").to_numpy() << 16 | df["event"].astype("int64").to_numpy())


lat_params = { "accept_threshold": 130e3, "cluster_cut": 30e3 }


def GenerateTAs(df, **params):
    runs = np.sort(df.run.unique())

    TA_list = []
    TP_list = []

    ta_offset=0

    for i, run in enumerate(runs):
        print(f"Processing run {i + 1}/{len(runs)}")
        df_run = df[df.run == run]
        if df_run.empty:
            continue

        TA_tmp, TP_tmp = trg_numba.TAMaker(df_run, **params, global_ta_offset = ta_offset)
        ta_offset += TA_tmp['TA_id'].max() + 1 

        TA_tmp = TA_tmp.assign(run=run)
        TP_tmp = TP_tmp.assign(run=run)

        TA_list.append(TA_tmp)
        TP_list.append(TP_tmp)

    TAs = pd.concat(TA_list, ignore_index=True)
    TPs = pd.concat(TP_list, ignore_index=True) 

    return TPs, TAs


base = "./beam/chunks"
out = "./trigger_data/beam_chunks"
os.makedirs(out, exist_ok=True)

indices = sorted({
    int(os.path.basename(f).split("_")[-1].replace(".pkl",""))
    for f in glob.glob(f"{base}/mc/*.pkl")
})

for idx in indices:
    for part in ["nue", "numu"]:
        mc = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{base}/mc/genie_{part}_mc_{idx}.pkl")])
        nu = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{base}/nu/genie_{part}_nu_{idx}.pkl")])
        eventsum = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{base}/sum/genie_{part}_sum_{idx}.pkl")])
        tps = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{base}/tps/genie_{part}_tps_{idx}.pkl")])

        for df in [mc, nu, eventsum, tps]:
            get_unique_event_ids(df)
            df.sort_values(by="event_uid", inplace=True)

        eventsum['visible_energy'] = eventsum['tot_visible_energy_rop2'] + eventsum['tot_visible_energy_rop3']

        print("\n" + f"generating central APA efficiencies for {part} chunk {idx}")
        cTPs, cTAs = GenerateTAs(tps, add_backgrounds=True)

        print("\n" + f"generating lateral APA efficiencies for {part} chunk {idx}")
        lTPs, lTAs = GenerateTAs(tps, add_backgrounds=True, **lat_params)

        trig_cent = pd.merge(
            cTAs,
            eventsum[['event','run','visible_energy','event_uid']],
            on=['event','run'],
            how='right'
        ).fillna(-1)

        trig_lat = pd.merge(
            lTAs,
            eventsum[['event','run','visible_energy','event_uid']],
            on=['event','run'],
            how='right'
        ).fillna(-1)

        trig_cent.to_pickle(f"{out}/{part}_TAs_cbgd_{idx}.pkl")
        trig_lat.to_pickle(f"{out}/{part}_TAs_lbgd_{idx}.pkl")
        cTPs.to_pickle(f"{out}/{part}_TPs_cbgd_{idx}.pkl")
        lTPs.to_pickle(f"{out}/{part}_TPs_lbgd_{idx}.pkl")
