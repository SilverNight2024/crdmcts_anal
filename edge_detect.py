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
          
def mask_test(nms, imgs):
     testing = True
     while testing:
        bad = True
        while bad:
            print("\nPlease don't misnput I will break 👉👈")
            try:
                im_idx = int(input("\nPlease input integer index of image " +
                                    f"(not greater than {len(nms) -1}): "))
                g_sig = int(input("\nPlease input gaussian bg" + 
                                     " estimation sigma (integer): "))
                clp_lim = float(input("\nPlease input CLAHE clip limit (float): "))
                gamma = float(input("\nPlease input gamma correction (float): "))
                loops = int(input("\nInput number of cleaning loops (int): "))
                fig, ((ori_plt, new_img_plt, msk_plt))\
                      = plt.subplots(1, 3)
                ori_plt.imshow(imgs[im_idx,:,:])
                ori_plt.set_title(nms[im_idx])
                ori_plt.axis("off")
                ori_g = (imgs[im_idx,:,:] / np.max(imgs[im_idx,:,:]))\
                      ** gamma
                ori_g_8 = cv.normalize(ori_g, None, 0, 255,\
                                       cv.NORM_MINMAX).astype('uint8')
                clh = cv.createCLAHE(clipLimit = clp_lim, tileGridSize = (8,8))
                img = ori_g_8
                for _ in range(loops):
                    blur = cv.GaussianBlur(img,\
                                        (0,0),g_sig)
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
                k = np.ones((5,5), np.uint8)
                # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k)
                # mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=2)
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
mask_test(rep_nms, rep_imgs)

    

