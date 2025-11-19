import os #To interpret file paths
import pickle #To save the dicts
import numpy as np #to manipulate images
import matplotlib.pyplot as plt #plotting
import cv2 as cv
from prettytable import PrettyTable
from open_lif import use_open_lif
from img_dict_manip import show_imgs, get_key_parts, req_two_idx

def prompt_show_imgs(nm_dict, im_dict, delim_str):
    """Prompts user if they'd like to use show_imgs"""
    bad = True
    while bad:
            see_imgs = input("Would you like to visualize images? (Y/N): ")
            if see_imgs.capitalize() == "Y": #Open more images if yes
                show_imgs(nm_dict,im_dict,delim_str)
                bad = False
            elif see_imgs.capitalize() == "N":
                bad = False
            else:
                print("Invalid input, please try again.")

def img_select(sup_keys, sub_keys, delim_str, nm_dct, img_dct, sel_vec):
     """The first two inputs are just teh outputs of get_key_parts
        the second two are standard and are defined elsewhere
        but the last is a 2D array of indexes corresponding to the
        super_key string, sub_key string, and image array linear 
        index of the desired image for the output"""
     sel_vec_l = sel_vec.shape[0]
     img_arr = np.zeros((sel_vec_l,1080,1280))
     nm_lst = [""] * sel_vec_l
     for i, arr in enumerate(sel_vec):
          tot_key = sup_keys[arr[0]] + delim_str + sub_keys[arr[1]]
          img_arr[i,:,:] = img_dct[tot_key][arr[2]]
          nm_lst[i] = nm_dct[tot_key][arr[2]]
     return nm_lst, img_arr

def mask_test(nms, imgs):
     testing = True
     while testing:
        bad = True
        while bad:
            print("\nPlease don't misnput I will break 👉👈")
            try:
                im_idx = int(input("\nPlease input integer index of image " +
                                    f"(not greater than {len(nms) -1}): "))
                loops = int(input("\nPlease input number of loops (int): "))
                fig, ((ori_plt, new_img_plt, msk_plt))\
                      = plt.subplots(1, 3)
                ori_plt.imshow(imgs[im_idx,:,:])
                ori_plt.set_title(nms[im_idx])
                ori_plt.axis("off")
                ori_g = (imgs[im_idx,:,:] / np.max(imgs[im_idx,:,:]))\
                      ** 0.5
                ori_g_8 = cv.normalize(ori_g, None, 0, 255,\
                                       cv.NORM_MINMAX).astype('uint8')
                clh = cv.createCLAHE(clipLimit = 1.0, tileGridSize = (8,8))
                img = ori_g_8
                for _ in range(loops):
                    blur = cv.GaussianBlur(img,\
                                        (0,0),80)
                    img = cv.subtract(img,blur)
                    img= cv.normalize(img, None, 0, 255,\
                                        cv.NORM_MINMAX).astype('uint8')
                    img = clh.apply(img)
                    img = cv.equalizeHist(img)
                img = cv.GaussianBlur(img, (0,0),4)
                img = cv.fastNlMeansDenoising(img,None,5,7,21)
                new_img_plt.imshow(img)
                new_img_plt.set_title("Img Cleaned")
                new_img_plt.axis("off")
                _, mask = cv.threshold(img, 0, 255,\
                                       cv.THRESH_BINARY + cv.THRESH_OTSU)
                msk_plt.imshow(mask)
                msk_plt.set_title("Otsu Mask")
                msk_plt.axis("off")
                print("\nNOTE: You will have to close all image" + 
                      " windown before continuing")
                plt.show()
            except ValueError as e:
                print(f"\nValue Error: {e}")
                continue  
            except cv.error as e:
                # This block will execute if a cv2.error is raised
                print(f"\nOpenCV Error: {e}")
                continue
            bad = False
        bad = True
        while bad:
            go_on = input("\nContinue Testing? (Y/N): ")
            if go_on.capitalize() == "Y": #Open more images if yes
                bad = False
            elif go_on.capitalize() == "N":
                testing = False
                bad = False
            else:
                print("\nInvalid input, please try again.")

def pkl_upkl_dct(dct,fnm,bool):
    """if Bool is true, write, if false, read"""
    if bool:
        with open(fnm, "wb") as f: #pickle dicts
            pickle.dump(dct, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(fnm, "rb") as f: #pickle dicts
            return pickle.load(f)

def mask(img_in):
    img = img_in.copy()
    p99 = np.percentile(img, 99)
    p95 = np.percentile(img, 95)
    t = p99 - p95 #makes the gap the threshold for no signal
    if t > 80: #empirical threshold with a little give 
        img = (img / np.max(img))\
            ** 0.5
        img = cv.normalize(img, None, 0, 255,\
                        cv.NORM_MINMAX).astype('uint8')
        clh = cv.createCLAHE(clipLimit = 1.0, tileGridSize = (8,8))
        for _ in range(5):
            blur = cv.GaussianBlur(img,\
                                (0,0),80)
            img = cv.subtract(img,blur)
            img= cv.normalize(img, None, 0, 255,\
                                cv.NORM_MINMAX).astype('uint8')
            img = clh.apply(img)
            img = cv.equalizeHist(img)
        img = cv.GaussianBlur(img, (0,0),4)
        img = cv.fastNlMeansDenoising(img,None,5,7,21)
        _, mask = cv.threshold(img, 0, 255,\
                                cv.THRESH_BINARY + cv.THRESH_OTSU)
    else:
        mask = np.zeros(img.shape, dtype = np.uint8)
    return mask

def img_dct_2_msks(in_dct,nm_dct,sup_strs,sub_strs,delim):
    c_dir = os.getcwd()
    fld_dir = "msk_dcts"
    c_fld_dir = os.path.join(c_dir,fld_dir)
    if os.path.exists(c_fld_dir) & os.path.isdir(c_fld_dir):
        if len(os.listdir(c_fld_dir)) > 0: 
            #check if fls, prompt delete if so
            bad = True
            while bad: #ask if the user wants to delete mask dcts
                del_msks = input("\nWould you like to delete mask dicts? (Y/N): ")
                if del_msks.capitalize() == "Y": #Delete masks
                    print("\n")
                    for file in os.listdir(c_fld_dir): #delete msk dcts
                        t_dir = os.path.join(c_fld_dir,file)
                        os.remove(t_dir)
                        print(f"{file} deleted")
                    mk_msks = True
                    bad = False
                elif del_msks.capitalize() == "N": #Don't delete masks
                    mk_msks = False
                    bad = False
                else:
                    print("\nInvalid input, please try again.")
        else: #generate mask dcts
            mk_msks = True
            print(f"\nNo dicts found in {fld_dir} folder")
    else: #no folder detected
        print(f"\n No folder by name {fld_dir} found, check\n" + 
              "img_dct_2_msk directory vars and actual directory")

    if mk_msks:
        print("\nStarting Mask Creation")
        for sup_k in sup_strs: #gen sup_k dicts
            msk_dct = {}
            for sub_k in sub_strs:
                k = sup_k + delim + sub_k
                tmp_arr = np.zeros(in_dct[k].shape, dtype=np.uint8)
                for i in range(in_dct[k].shape[0]): #gen dict
                    tmp_arr[i,:,:] =\
                    mask(in_dct[k][i,:,:])
                msk_dct[k] = tmp_arr
            dct_nm = sup_k.replace(".lif",".pkl")
            t_dir = os.path.join(c_fld_dir,dct_nm)
            pkl_upkl_dct(msk_dct,t_dir,True)
            print(f"\n{dct_nm} Mask Dict Pickled")

def comp_msk(nm_dict, img_dict, delim_str):
    """Pared down the code in show_imgs to allow for just 
    index based image opening in subplot with img and mask"""
    sup_kys, sub_kys = get_key_parts(nm_dict, delim_str)
    table_1 = PrettyTable()
    table_1.field_names = ["Prompt Input", "First Prompt","Second Prompt"]
    for i in range(max(len(sup_kys), len(sub_kys))): #defines table
        if ((i < len(sup_kys)) & (i < len(sub_kys))):
            tbl_row = [i, sup_kys[i], sub_kys[i]]
        elif ((i < len(sup_kys)) & (i >= len(sub_kys))):
            tbl_row = [i,sup_kys[i],"N/A"]
        elif ((i >= len(sup_kys)) & (i < len(sub_kys))):
            tbl_row = [i,"N/A",sub_kys[i]] 
        table_1.add_row(tbl_row)
    more_imgs = True #set while loop
    sm_msk_b = False
    while more_imgs: #allow for the opening of multiple sets of images
        print("\nHere is a table with indexes associated with image files and transcript\n" +
            "categories of the images openable through this function.")
        print(table_1)
        print("\nPlease put input corresponding to\n" +
        "the prompt number and desired img key")
        sup_kys_idx, sub_kys_idx = req_two_idx(len(sup_kys),len(sub_kys))
        req_key = sup_kys[sup_kys_idx] + delim_str + sub_kys[sub_kys_idx]
        img_mns = img_dict[req_key].mean(axis=(1,2)) #gets mean for brightness
        if not(sm_msk_b): #get dict
            c_dir = os.getcwd()
            fld_dir = "msk_dcts"
            c_fld_dir = os.path.join(c_dir,fld_dir)
            msk_dct_nm = sup_kys[sup_kys_idx].replace('.lif','.pkl') #get mask name
            msk_dct = pkl_upkl_dct(None,os.path.join(c_fld_dir,msk_dct_nm),False)  
        msk_mns = msk_dct[req_key].mean(axis=(1,2))
        print("\nHere is a table with the index associated with,\n" +
                    "name of, and brightness of all slected images.")
        gap_arr = np.zeros(img_dict[req_key].shape[0])
        for i in range(img_dict[req_key].shape[0]): #generate gap array
            img = img_dict[req_key][i,:,:].copy()
            p99 = np.percentile(img, 99)
            p95 = np.percentile(img, 95)
            gap_arr[i] = p99 - p95
        table = PrettyTable()
        table.add_column("Index",np.arange(0,len(img_mns)))
        table.add_column("Image Name",nm_dict[req_key])
        table.add_column("Mean Brightness",img_mns)
        table.add_column("99-95 gap",gap_arr)
        table.add_column("Mask Mean",msk_mns)
        print(table)
        bad = True
        while bad: #prompt to open more images
            img_opn = input("\nWould You like to Open Images? (Y/N): ")
            if img_opn.capitalize() == "Y": #Open more images if yes
                show_imgs = True
                bad = False
            elif img_opn.capitalize() == "N": #open no images if no
                show_imgs = False
                bad = False
            else: #retry
                print("\nInvalid input, please try again.")
        if show_imgs:
            max_idx = len(nm_dict[req_key])
            bad = True
            while bad: #makes sure indexes are correct
                print("\nYou will be asked to input first and second indexes\n" +
                    "These will be the starting and ending integer indexes\n" +
                    "of images in the selsection you would like to open.\n" +
                    "The first index can thus equal but NOT exceed the second.\n" +
                    f"They also cannot be less than 0 or greater than {max_idx-1}.")
                idx_s, idx_e = req_two_idx(max_idx,max_idx)   
                if idx_s <= idx_e:
                    proc_q = input(f"\nThis will generate {idx_e-idx_s+1} imgs. Proceed? (Y/N): ")
                    if proc_q .capitalize() == "Y": #Open images is yes
                        print("\nNOTE: You will have to close all image windown before continuing")
                        bad = False
                    else:
                        print("\nWill not proceed, returning to indexes prompt.")
            img_sel = img_dict[req_key][np.arange(idx_s,idx_e+1)]
            msk_sel = msk_dct[req_key][np.arange(idx_s,idx_e+1)]
            for i in range(img_sel.shape[0]):
                fig, ((ori_plt, msk_plt)) = plt.subplots(1, 2)
                fig.canvas.manager.set_window_title(nm_dict[req_key][i+idx_s])  # real window title
                ori_plt.imshow(img_sel[i]) 
                ori_plt.axis("off")
                ori_plt.set_title(nm_dict[req_key][i+idx_s])
                msk_plt.imshow(msk_sel[i]) 
                msk_plt.axis("off")
                msk_plt.set_title("Mask")
            plt.show()
        bad = True
        while bad: #ask if the user wants to open more images
            go_on = input("\nContinue to look at masks? (Y/N): ")
            if go_on.capitalize() == "Y": #Open more images if yes
                sm_msk = input("\nWill you still be looking at\n" + 
                           f"{sup_kys[sup_kys_idx]}? (Y/N):")
                if sm_msk.capitalize() == "Y": #Open more images if yes
                    sm_msk_b = True
                    more_imgs = True
                    bad = False
                elif sm_msk_b.capitalize() == "N":
                    del(msk_dct)
                    sm_msk_b = False
                    more_imgs = False
                    bad = False
                else:
                    print("\nInvalid input, please try again.")
                bad = False
            elif go_on.capitalize() == "N":
                more_imgs = False
                bad = False
            else:
                print("\nInvalid input, please try again.")

def msk_dta_sum_tbls(nm_dict, img_dict, delim_str):
    """No functionality but printing the comparison info"""
    sup_kys, sub_kys = get_key_parts(nm_dict, delim_str)
    c_dir = os.getcwd()
    fld_dir = "msk_dcts"
    c_fld_dir = os.path.join(c_dir,fld_dir)
    for i in range(len(sup_kys)):
        msk_dct_nm = sup_kys[i].replace('.lif','.pkl') #get mask name
        msk_dct = pkl_upkl_dct(None,os.path.join(c_fld_dir,msk_dct_nm),False)
        dngn = True
        while dngn: #stop to inspect
            go_on = input(f"\nGenerate tables for {sup_kys[i]}? (Y/N): ")
            if go_on.capitalize() == "Y":
                dngn = False
            elif go_on.capitalize() == "N":
                return()
            else:
                print("\nIncorrect input. Please try again.")  
        for j in range(len(sub_kys)):
            k = sup_kys[i] + delim_str + sub_kys[j]
            msk_mns = msk_dct[k].mean(axis=(1,2))
            img_mns = img_dict[k].mean(axis=(1,2))
            gap_arr = np.zeros(img_dict[k].shape[0])
            print("\n")
            for l in range(img_dict[k].shape[0]): #generate gap array
                img = img_dict[k][l,:,:].copy()
                p99 = np.percentile(img, 99)
                p95 = np.percentile(img, 95)
                gap_arr[l] = p99 - p95
            table = PrettyTable()
            table.add_column("Index",np.arange(0,len(img_mns)))
            table.add_column("Image Name",nm_dict[k])
            table.add_column("Mean Brightness",img_mns)
            table.add_column("99-95 gap",gap_arr)
            table.add_column("Mask Mean",msk_mns)
            print(table)
            
img_Nms, imgs = use_open_lif()
delim_str = "#"
# lif_fls, trns_nms =\
#       get_key_parts(img_Nms, delim_str)
#img_dct_2_msks(imgs, img_Nms, 
            #    lif_fls, trns_nms, 
            #    delim_str)
comp_msk(img_Nms, imgs, delim_str)


# rep_img_sel_vec = \
# np.array([
#     [3,3,8],
#     [3,4,0],
#     [3,5,5],
#     [3,1,2],
#     [0, 1, 17],
#     [0, 4, 16],
#     [0, 4, 9],
#     [3, 7, 10],
#     [3, 7, 11],
#     [8, 2, 18],
#     [8, 2, 19]
# ])

#prompt_show_imgs(img_Nms, imgs, delim_str)
#sel_nm, sel_img = img_select(
    # lif_fls, trns_nms, 
    # delim_str, img_Nms, 
    # imgs, rep_img_sel_vec)
#mask_test(sel_nm, sel_img)

