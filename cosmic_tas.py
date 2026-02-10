import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import trigger_algs_numba as trg_numba
import utils as util
import os
import glob

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


base = "./cosmics/pkl/"
out = "./trigger_data/cosmic_chunks"
os.makedirs(out, exist_ok=True)

indices = sorted({ int(os.path.basename(f).split("_")[-1].replace(".pkl",""))for f in glob.glob(f"{base}/*.pkl")})

print(indices)

for idx in indices:
    for part in ["cosmic"]:
        mc = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{base}/{part}_mc_00{idx}.pkl")])
        eventsum = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{base}/{part}_eventsum_00{idx}.pkl")])
        tps = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{base}/{part}_tps_00{idx}.pkl")])

        for df in [mc, eventsum, tps]:
            get_unique_event_ids(df)
            df.sort_values(by="event_uid", inplace=True)

        eventsum['visible_energy'] = eventsum['tot_visible_energy_rop2'] + eventsum['tot_visible_energy_rop3']

        print("\n" + f"generating central APA efficiencies for {part} chunk {idx}")
        cTPs, cTAs = GenerateTAs(tps)

        print("\n" + f"generating lateral APA efficiencies for {part} chunk {idx}")
        lTPs, lTAs = GenerateTAs(tps, **lat_params)

        trig_cent = pd.merge(cTAs,eventsum[['event','run','visible_energy','event_uid']],on=['event','run'],how='right').fillna(-1)

        trig_lat = pd.merge( lTAs, eventsum[['event','run','visible_energy','event_uid']], on=['event','run'],how='right').fillna(-1)

        trig_cent.to_pickle(f"{out}/{part}_TAs_cbgd_{idx}.pkl")
        trig_lat.to_pickle(f"{out}/{part}_TAs_lbgd_{idx}.pkl")
        cTPs.to_pickle(f"{out}/{part}_TPs_cbgd_{idx}.pkl")
        lTPs.to_pickle(f"{out}/{part}_TPs_lbgd_{idx}.pkl")


from pathlib import Path

folder = Path( "./trigger_data/cosmic_chunks")

print("Combining trigger dfs") 
for part in ['cosmic']:
    for bgd in ['lbgd','cbgd']:
        files = folder.glob(f"{part}_TAs_{bgd}*.pkl")

        dfs = []

        for f in files:
            df = pd.read_pickle(f)
            dfs.append(df)

        trig_df = pd.concat(dfs, ignore_index = True)
        trig_df.to_pickle(f"./trigger_data/cosmic_TAs_{bgd}.pkl")

print("Combining summary dfs")

for data_type in ['eventsum',  'mc']:
    folder = Path("./cosmics/pkl/")

    for part in ['cosmic']:
        files = folder.glob(f"{part}_{data_type}_*")
        dfs = []

        for f in files:
            df = pd.read_pickle(f)
            util.get_unique_event_ids(df)
            dfs.append(df)
        trig_df = pd.concat(dfs, ignore_index = True)
        trig_df.to_pickle(f"./cosmics/{part}_{data_type}.pkl")


print("Done.")
