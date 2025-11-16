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
                k_size = int(input("\nPlease input Gaus kernel size (odd integer): "))
                sigma = float(input("\nPlease input sigma (float): "))
                nrm_str = float(input("\nPlease input normalization strength" +
                                        "(int > 0): ")) 
                nrm_tmp_sz = int(input("\nPlease input normalization template" + 
                                    " size (odd integer) (recommended --> 7)): "))
                nrm_ar_sz = int(input("\nPlease input normalization area" + 
                                    " size (odd integer (recommended --> 21)): "))
                print("\nNOTE: upper and lower Canny edge detection thresholds" + 
                        "\nshould have a ratio between 2:1 and 3:1")
                thresh_l = int(input("\nPlease input lower Canny threshold" +
                                        "(int > 0): "))
                thresh_h = int(input("\nPlease input upper Canny threshold" +
                                        "(int > 0): "))    
                fig, ((ori, blur_8), (blur_8_den,img_edges))\
                      = plt.subplots(2, 2)
                ori.imshow(imgs[im_idx,:,:])
                ori.set_title(nms[im_idx])
                ori.axis("off")
                blur = cv.GaussianBlur(imgs[im_idx,:,:],\
                                       (k_size,k_size),sigma)
                blur_8_temp = cv.normalize(blur, None, 0, 255,\
                                       cv.NORM_MINMAX).astype('uint8')
                #normalize linearly maps pixels from 0 --> 255 to preserve
                #gradients for canny. .astype('uint8') allows cv to 
                #actually read the image by converting back to integers
                blur_8.imshow(blur_8_temp)
                blur_8.set_title("8-bit Blur")
                blur_8.axis("off")
                blur_8_den_temp = cv.fastNlMeansDenoising\
                    (blur_8_temp,None,h = nrm_str,\
                     templateWindowSize = nrm_tmp_sz,\
                        searchWindowSize =nrm_ar_sz)
                blur_8_den_temp = cv.GaussianBlur(blur_8_den_temp,\
                                       (k_size,k_size),sigma)
                blur_8_den.imshow(blur_8_den_temp)
                blur_8_den.set_title("8-bit Blur Denoised")
                blur_8_den.axis("off")
                img_edges.imshow(cv.Canny(blur_8_den_temp, thresh_l, thresh_h))
                img_edges.set_title("Canny Edges")
                img_edges.axis("off")
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

    

