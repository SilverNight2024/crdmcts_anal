from open_lif import use_open_lif #to load in dicts
from img_dict_manip import show_imgs, get_key_parts
import numpy as np #to manipulate images
import matplotlib.pyplot as plt #plotting
import cv2 as cv

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
          
def edge_det_test(nms, imgs):
     testing = True
     while testing:
        bad = True
        while bad:
            print("\nPlease don't misnput I will break 👉👈")
            try:
                im_idx = int(input("\nPlease input integer index of image" +
                                    f"(not greater than {len(nms) -1}): "))
                bl_size = int(input("\nPlease input bilateral filter size" + 
                                     " (integer <10): "))
                bl_s_cl = float(input("\nPlease input bilateral filter sigma" + 
                                     " color value (float): "))
                bl_s_sp = int(input("\nPlease input bilateral filter sigma" + 
                                     " space size (integer): "))
                clp_lim = float(input("\nPlease input CLAHE clip limit (float): "))
                oc_k_sz = int(input("\nPlease input open/close kernel" + 
                                     " size (integer): "))
                fig, ((ori_plt, bl_8_plt), (clh_plt,msk_plt))\
                      = plt.subplots(2, 2)
                ori_plt.imshow(imgs[im_idx,:,:])
                ori_plt.set_title(nms[im_idx])
                ori_plt.axis("off")
                blur = cv.GaussianBlur(imgs[im_idx,:,:],\
                                       (21,21),5)
                blur_32 = blur.astype(np.float32) / 65535 #optimizes bilateral filter
                bl = cv.bilateralFilter(blur_32,d=bl_size,sigmaColor = bl_s_cl\
                                        ,sigmaSpace = bl_s_sp)
                bl_8= cv.normalize(bl, None, 0, 255,\
                                       cv.NORM_MINMAX).astype('uint8')
                #normalize linearly maps pixels from 0 --> 255 to preserve
                #gradients. .astype('uint8') allows cv to 
                #actually read the image by converting back to integers
                bl_8_plt.imshow(bl_8)
                bl_8_plt.set_title("8-bit BL Filter after G Blur")
                bl_8_plt.axis("off")
                clh = cv.createCLAHE(clipLimit = clp_lim, tileGridSize = (8,8))
                clahe = clh.apply(bl_8)
                clh_plt.imshow(clahe)
                clh_plt.set_title("8-bit CLAHE")
                clh_plt.axis("off")
                _, mask = cv.threshold(clahe, 0, 255,\
                                       cv.THRESH_BINARY + cv.THRESH_OTSU)
                k = np.ones((oc_k_sz,oc_k_sz), np.uint8)
                mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
                mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=2)
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

img_Nms, imgs = use_open_lif()
delim_str = "#"
#prompt_show_imgs(img_Nms, imgs, delim_str)
rep_img_sel_vec = np.array([[3,3,8],[3,4,0],[3,5,5],[3,1,2]])
lif_fls, trns_nms = get_key_parts(img_Nms, delim_str)
rep_nms, rep_imgs = img_select(lif_fls, trns_nms, delim_str,
                              img_Nms, imgs, rep_img_sel_vec)
edge_det_test(rep_nms, rep_imgs)

    

