import os #To interpret file paths
import pickle #To save the dicts
import numpy as np #to manipulate images
import matplotlib.pyplot as plt #plotting
import cv2 as cv
from prettytable import PrettyTable
from open_lif import use_open_lif
from img_dict_manip import show_imgs, get_key_parts

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

def pkl_dct(dct,fnm):
    with open(fnm, "wb") as f: #pickle dicts
        pickle.dump(dct, f, protocol=pickle.HIGHEST_PROTOCOL)

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
        print("\n Starting Mask Creation")
        for sup_k in sup_strs: #gen sup_k dicts
            msk_dct = {}
            for sub_k in sub_strs:
                k = sup_k + delim + sub_k
                tmp_arr = np.zeros(in_dct[k].shape, dtype=np.uint8)
                print("\n")
                for i in range(in_dct[k].shape[0]): #gen dict
                    tmp_arr[i,:,:] =\
                    mask(in_dct[k][i,:,:])
                    print(f"{nm_dct[k][i]} Mask Created")
                msk_dct[k] = tmp_arr
            dct_nm = sup_k.replace(".lif",".pkl")
            t_dir = os.path.join(c_fld_dir,dct_nm)
            pkl_dct(msk_dct,t_dir)
            print(f"{dct_nm} Mask Dict Pickled")

img_Nms, imgs = use_open_lif()
delim_str = "#"
lif_fls, trns_nms =\
      get_key_parts(img_Nms, delim_str)
img_dct_2_msks(imgs, img_Nms, 
               lif_fls, trns_nms, 
               delim_str)

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

