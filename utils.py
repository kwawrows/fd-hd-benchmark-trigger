import numpy as np 
import pandas as pd 
import trigger_algs_numba as trg_numba


#function to generate unique event IDs by combining run, subrun, and event number. maybe there's a better way of doing this who knows. 
def get_unique_event_ids(df):
    df["event_uid"] = ( df["run"].astype("int64").to_numpy()    << 32 | df["subrun"].astype("int64").to_numpy() << 16 | df["event"].astype("int64").to_numpy())

#Apply the TAMaker alg. 

def GenerateTAs(df, **params):
    runs = np.sort(df.run.unique())

    TA_list = [] #TAs from each subrun
    TP_list = [] #TPs from each subrun

    ta_offset=0 #global offset to make sure TA IDs are unique across subruns

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



"""
Get the track angles w.r.t readout plane in range [0, 90] degrees. i.e. parallel / normal, based on particle mc momentum 
"""

def calculate_angles(px, py, pz, p_mag):

    # angle from vertical y-axis, treating any direction the same
    theta_y = np.arcsin(py/p_mag)#np.degrees( np.arcsin( np.sqrt(px**2 + pz**2) / p_mag ) )

    # xz-plane deviation from z-axis
    theta_xz = np.degrees( np.arctan( np.abs(px) / np.abs(pz) ) )

    # rotate by ±37.5° in zy plane
    theta_rot_U = np.radians(-37.5)
    theta_rot_V = np.radians(37.5)

    # U-plane
    p_y_U = py * np.cos(theta_rot_U) - pz * np.sin(theta_rot_U)
    p_z_U = py * np.sin(theta_rot_U) + pz * np.cos(theta_rot_U)
    theta_y_U = np.degrees( np.arcsin( np.sqrt(px**2 + p_z_U**2) / p_mag ) )
    theta_xz_U = np.degrees( np.arctan( np.abs(px) / np.abs(p_z_U) ) )

    # V-plane
    p_y_V = py * np.cos(theta_rot_V) - pz * np.sin(theta_rot_V)
    p_z_V = py * np.sin(theta_rot_V) + pz * np.cos(theta_rot_V)
    theta_y_V = np.degrees( np.arcsin( np.sqrt(px**2 + p_z_V**2) / p_mag ) )
    theta_xz_V = np.degrees( np.arctan( np.abs(px) / np.abs(p_z_V) ) )

    return theta_y, theta_y_U, theta_y_V, theta_xz, theta_xz_U, theta_xz_V



def calculate_col_angles(px, py, pz, p_mag):

    # angle relative to y-axis
    theta_y = np.degrees(np.arccos(py / p_mag))

    # angle in the xz-plane relative to z-axis
    theta_xz = np.degrees(np.arctan2(px, pz))


    return (theta_y), (theta_xz)


def calculate_col_angles_inplace(df):

    px = df.px; py = df.py; pz = df.pz; p_mag = df.p

    # angle relative to y-axis
    theta_y = np.degrees(np.arccos(py / p_mag))

    # angle in the xz-plane relative to z-axis
    theta_xz = np.degrees(np.arctan2(px, pz))

    df['theta_y'] = theta_y 
    df['theta_xz'] = theta_xz





