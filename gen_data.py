import os #To interpret file paths
import math #for math
import numpy as np #to manipulate images
import pickle #To save the dicts
from prettytable import PrettyTable
from open_lif import use_open_lif
from img_dict_manip import get_key_parts
from edge_detect import pkl_upkl_dct, mask

def frc_tbl(nms_dict, img_dict, delim_str):
    """Explores mask pixel amounts to get cutoffs and see data"""
    sup_kys, sub_kys = get_key_parts(nms_dict, delim_str)
    c_dir = os.getcwd()
    fld_dir = "msk_dcts"
    c_fld_dir = os.path.join(c_dir,fld_dir)
    frc_arr = np.zeros((207,len(sup_kys)-2),dtype=np.float32)
    for i in range(1,len(sup_kys)-1):
        msk_dct_nm = sup_kys[i].replace('.lif','.pkl') #get mask name
        msk_dct = pkl_upkl_dct(None,os.path.join(c_fld_dir,msk_dct_nm),False)
        ctr = 0
        for j in range(len(sub_kys)):
            key = sup_kys[i] + delim_str + sub_kys[j]
            for k in range(msk_dct[key].shape[0]):
                msk = msk_dct[key][k,:,:].copy()
                frc = np.sum(msk) / (255 * msk.shape[0] * msk.shape[1])
                if frc > 0.9:
                    # print(nms_dict[key][k])
                    # print([i,j,k])
                    msk = mask(img_dict[key][k,:,:])
                    frc = np.sum(msk) / (255 * msk.shape[0] * msk.shape[1])
                frc_arr[ctr,i-1] = frc
                ctr += 1
    nms = []
    for i in range(len(sub_kys)):
        key = sup_kys[1] + delim_str + sub_kys[i]
        nms.extend(nms_dict[key])
    tm_pts = ["18 hrs","26 hrs",
              "50 hrs","80 hrs",
              "100 hrs","123 hrs",
              "148 hrs","176 hrs"]
    table = PrettyTable()
    table.add_column("Name",nms)
    for i in range(frc_arr.shape[1]):
        table.add_column(tm_pts[i],frc_arr[:,i])
    print('\n')
    print(table)
      
def get_sum_arr(nms_dict, img_dict, delim_str, t_bool):
    """Generates Sum_Arr and extra info and computes 
    filter 4 using the mask fraction dct. Specifically
    Removes anything where the mask detected <7% total
    area and also removes data where the mask blew out
    t_bool also allows toggling of printing out prett
    tables of the results"""
    sup_kys, sub_kys = get_key_parts(nms_dict, delim_str)
    c_dir = os.getcwd()
    fld_dir = "msk_dcts"
    c_fld_dir = os.path.join(c_dir,fld_dir)
    frc_arr = np.zeros((207,len(sup_kys)-2),dtype=np.float32)
    sum_arr = np.zeros((207,len(sup_kys)-2),dtype=np.float32)
    for i in range(1,len(sup_kys)-1):
        msk_dct_nm = sup_kys[i].replace('.lif','.pkl') #get mask name
        msk_dct = pkl_upkl_dct(None,os.path.join(c_fld_dir,msk_dct_nm),False)
        ctr = 0
        for j in range(len(sub_kys)):
            key = sup_kys[i] + delim_str + sub_kys[j]
            for k in range(msk_dct[key].shape[0]):
                msk_b = msk_dct[key][k,:,:] > 0
                #converts mask to binary
                frc = np.sum(msk_b) / (msk_b.shape[0]\
                                       * msk_b.shape[1])
                if frc > 0.9:
                    msk_b = (mask(img_dict[key][k,:,:]) > 0)
                    frc = np.sum(msk_b) / (msk_b.shape[0]\
                                           * msk_b.shape[1])
                img = img_dict[key][k,:,:][msk_b]
                #masks img
                sum_arr[ctr,i-1] = np.sum(img)
                frc_arr[ctr,i-1] = frc
                ctr += 1
    for i in range(207): #remove replicates with little signal
        if frc_arr[i,3] < 0.06:
            sum_arr[i,:] = np.full(len(sum_arr[i,:]),np.nan)
    trns_idxs = [-1]
    for i in range(len(sub_kys)):
        key = sup_kys[1] + delim_str + sub_kys[i]
        trns_idxs.append(len(nms_dict[key]))
    #gets the indexes of the trans names
    trns_idxs[1:] = np.cumsum(trns_idxs[1:]) - 1 
    if t_bool: #pretty tables of the two arrays
        nms = []
        for i in range(len(sub_kys)):
            key = sup_kys[1] + delim_str + sub_kys[i]
            nms.extend(nms_dict[key])
        tm_pts = ["18 hrs","26 hrs",
                "50 hrs","80 hrs",
                "100 hrs","123 hrs",
                "148 hrs","176 hrs"]
        table_1 = PrettyTable()
        table_2 = PrettyTable()
        table_1.add_column("Name",nms)
        table_2.add_column("Name",nms)
        for i in range(frc_arr.shape[1]):
            table_1.add_column(tm_pts[i],frc_arr[:,i])
            table_2.add_column(tm_pts[i],sum_arr[:,i])
        print('\n')
        print(table_1)
        print('\n')
        print(table_2)
    return sum_arr, trns_idxs, sub_kys

def get_dta_arrs(nms_dict, img_dict, delim_str):
    """Generates all the data from sum_arr
    Note that all 4 arguments are only necessary if
    actually generating the data.pkl"""
    c_dir = os.getcwd()
    fl_nm = "data.pkl"
    fl_dir = os.path.join(c_dir,fl_nm)
    if os.path.exists(fl_dir): #read in data or regenerate it
        bad = True
        while bad: #ask if the user wants to delete mask dcts
            del_data = input("\nWould you like to delete and regenerate data? (Y/N): ")
            if del_data.capitalize() == "Y": #Delete data
                print(f"\n{fl_nm} deleted")
                print(f"\nGenerating Data")
                os.remove(fl_dir)
                mk_data = True
                bad = False
            elif del_data.capitalize() == "N": #Don't delete masks
                print(f"\nUnpickling data now")
                mk_data = False
                bad = False
                with open(fl_dir, "rb") as f: #unpickle dicts
                    trns_nms = pickle.load(f)
                    mns_arr = pickle.load(f)
                    nrm_mns_arr = pickle.load(f)
                    err_arr = pickle.load(f)
                    nrm_err_arr = pickle.load(f)
                return trns_nms, mns_arr, nrm_mns_arr, err_arr, nrm_err_arr
            else:
                print("\nInvalid input, please try again.")
    else: #generate mask dcts
            mk_data = True
            print(f"\nGenerating Data")
    if mk_data: #generate data
        sum_arr, trns_idxs, trns_nms\
        = get_sum_arr(nms_dict, img_dict, 
                      delim_str, False)
        mns_arr = np.zeros((len(trns_idxs) - 1,
                              sum_arr.shape[1]),
                              dtype=np.float32)
        nrm_mns_arr = np.zeros(mns_arr.shape,
                              dtype=np.float32)
        err_arr = np.zeros(mns_arr.shape,
                              dtype=np.float32)
        nrm_err_arr = np.zeros(mns_arr.shape,
                              dtype=np.float32)
        for i in range(1, len(trns_idxs)): #arrays
            ctr = 0
            for _, val in enumerate(sum_arr[int(trns_idxs[i-1])
                                            + 1:int(trns_idxs[i]) + 1
                                            , 1]):
                if np.isnan(val):
                    ctr += 1
            num_sums = int(trns_idxs[i]) - int(trns_idxs[i-1])
            n = num_sums - ctr
            if n == 0: #no divide by 0
                n == 1
            tmp_arr = sum_arr[int(trns_idxs[i-1]) + 1
                              :int(trns_idxs[i]) + 1, :]
            mns_arr[i-1,:] = np.nanmean(tmp_arr,axis = 0)
            err_arr[i-1,:] = np.nanstd(tmp_arr,axis = 0) / math.sqrt(n)
            norm_val = mns_arr[i-1,2]
            nrm_mns_arr[i-1,:] = mns_arr[i-1,:] / norm_val
            nrm_err_arr[i-1,:] = err_arr[i-1,:] / norm_val
        mns_arr[np.isnan(mns_arr)] = 0
        nrm_mns_arr[np.isnan(nrm_mns_arr)] = 0
        err_arr[np.isnan(err_arr)] = 0
        nrm_err_arr[np.isnan(nrm_err_arr)] = 0
        print("\ndata generated")
        with open(fl_dir, "wb") as f: #pickle data
            pickle.dump(trns_nms, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(mns_arr, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(nrm_mns_arr, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(err_arr, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(nrm_err_arr, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("\ndata pickled")
        return trns_nms, mns_arr, nrm_mns_arr, err_arr, nrm_err_arr

def arr2prtytbl(arr,clm_hds,rw_hds,rw_hds_nm):
    """Turns array into pretty table. clm_hds is
    necessary for column headers but row names and 
    headers is optional. Must put None if not using"""
    if len(clm_hds) == arr.shape[1]:
        table = PrettyTable()
        if (rw_hds is not None) &\
            (rw_hds_nm is not None):
            table.add_column(rw_hds_nm,rw_hds)
        for i in range(arr.shape[1]):
            table.add_column(clm_hds[i],arr[:,i])
        print('\n')
        print(table)
    else:
        print("Invalid column headers length!")
