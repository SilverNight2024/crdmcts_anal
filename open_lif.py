"""
File contains major functions--
get_Lif_Imgs: open file directory and load all lifs into an image and imagenames struct
stand_Lif_Imgs: take a specific naming convention in a representative image and make 
sure all images in the image name and images dictionary only have names that meet that convention

File contains minor functions--
substr_Idxs: gets the start index of all instances of a substring in a string
get_Trns_Nms: takes a list of images and a delimiter string to generate the unique
file naming convention for stand_Lif_Imgs
"""
import os #To interpret file paths
import pickle #To save the dicts
from readlif.reader import LifFile #importing readlif lif storing object
import numpy as np #to manipulate images
#Define sub-functions

def substr_Idxs(main_str, sub_str): #find all indexes of a substring in a main string
    """
    main_str is the main string with the sub_str
    NOTE: if no indexes are found this function will error
    """
    idxs = [0]
    offset = 0
    idx = 0
    while idx != -1:
        idx = main_str.find(sub_str)
        main_str = main_str[idx + len(sub_str):]
        real_idx = offset + idx #adjusts for deletion
        idxs.append(real_idx)
        offset += idx + len(sub_str) #increases with each deletion
    idxs = idxs[1:-1] 
    return idxs

def get_Trns_Nms(nms_Dct, sub_str, template_Lif): #Get Unique Transcript Names
    """nms_Dct is the names dictionary that comes out of get_Lif_Imgs"""
    lif_File_Nms = list(nms_Dct.keys())
    trns_Nms = nms_Dct[lif_File_Nms[template_Lif]].copy() #makes sure trns_Nms is unique
    for i in range(len(trns_Nms)):
        idxs = substr_Idxs(trns_Nms[i], sub_str)
        trns_Nms[i] = trns_Nms[i][idxs[0] + 1:idxs[1]]
    return list(set(trns_Nms))

#Define Main Functions
def get_Lif_Imgs(lifFolderPath, img_w, img_l, img_thr): #get img and imgnms dicts
    """
    takes in a folder path with lif files, the dimensions of images in the file, 
    and the threshold for the lowest pixel value in the image
    NOTE this only works if the path presented is a directory with only lif images in it
    """
    if os.path.exists(lifFolderPath) & os.path.isdir(lifFolderPath):
        lif_File_Nms = os.listdir(path=lifFolderPath) #get list of files int the directory
        img_Nms = {}
        imgs = {}
        for i in range(len(lif_File_Nms)):
            lif_File = LifFile(lifFolderPath + "\\" + lif_File_Nms[i]) #get lif file object
            tmp_Img_Arr = np.zeros((lif_File.num_images, img_l ,img_w), dtype=np.uint16) #preallocate temp img arr
            tmp_Nm_Arr = [""] * lif_File.num_images #preallocate temp name array
            for j in range(lif_File.num_images):
                #Get image array and alter the LUT
                img = lif_File.get_image(j)
                img_Arr = np.array(img.get_frame(c=0))
                img_Arr[img_Arr <= img_thr] = 0 #set all values under the threshold to 0
                tmp_Img_Arr[j] = img_Arr
                #Format Image Name
                tmp_Nm = img.name
                idx = tmp_Nm.index("/")
                tmp_Nm = tmp_Nm[idx + 1:]
                tmp_Nm_Arr[j] = tmp_Nm
            img_Nms[lif_File_Nms[i]] = tmp_Nm_Arr #store names in dict
            imgs[lif_File_Nms[i]] = tmp_Img_Arr #store images in dict
            print(lif_File_Nms[i] + " Images Unpacked")
    return img_Nms, imgs

def stand_Lif_Imgs(raw_nms_dct, raw_imgs_dct, delim_str, template_Lif):
    """
    takes dictionaries of names and images and standardizes 
    them based on delimited standard names in a template lif file's image names

    In this case standardization is the generation of two dictionaries that add 
    the standard names to the keys of the original dictionaries delimited 
    by a #

    template_Lif: integer index of the file in the loaded dicts
    delim_str: whatever delimiter the naming convention is contained in

    NOTE: the image convenstion must be contained. This is also not terribly 
    generalizable past the cardiomyocyte project it was designed for
    """
    trns_Nms = get_Trns_Nms(raw_nms_dct, delim_str, template_Lif)
    lif_File_Nms = list(raw_nms_dct.keys())
    img_nm_dict = {}
    img_dict = {}
    for i in range(len(lif_File_Nms)):
        img_Nms = raw_nms_dct[lif_File_Nms[i]]
        for j in range(len(trns_Nms)): #find all the indexes in the names that have a template transcript name
            tmp_idxs = np.zeros(len(img_Nms), dtype=bool)
            for k in range(len(img_Nms)):
                tmp_idxs[k] = ((trns_Nms[j] + delim_str) in img_Nms[k]) #needs the extra underscore to be restrictive
            no_del_idxs = np.flatnonzero(tmp_idxs) #get non-zero elements
            img_nm_dict[lif_File_Nms[i]+"#"+trns_Nms[j]] = [raw_nms_dct[lif_File_Nms[i]][l] for l in no_del_idxs]
            img_dict[lif_File_Nms[i]+"#"+trns_Nms[j]] = raw_imgs_dct[lif_File_Nms[i]][no_del_idxs,:,:]
    print("\nImage Dicts Standardized")
    return img_nm_dict, img_dict

#Streamline Use

def load_save_dicts(pkl_fnm):
    """ Loads in and saves dicts to pkl_fnm"""
    #define loading vars
    lifFolderPath = r"C:\Users\Simon\Documents\Cardiomyocytes Files\lif files"
    img_w = 1280
    img_l = 1080
    img_thr = 1000
    #Get, format, and save dicts
    raw_img_Nms, raw_imgs = get_Lif_Imgs(lifFolderPath, img_w, img_l, img_thr) #Get image dicts
    img_Nms, imgs = stand_Lif_Imgs(raw_img_Nms, raw_imgs,"_",0) #Standardize image dicts
    with open(pkl_fnm, "wb") as f: #pickle dicts
        pickle.dump(img_Nms, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(imgs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("\nImage names dict and images dict have been generated and saved")
    return img_Nms, imgs

def use_open_lif():
    """
    -Generate Dicts and Save them if nonexixtent
    -Delete saved dicts if you want to regenerate
    -Load in saved dicts otherwise
    """
    c_dir = os.getcwd()
    pkl_fnm = "cmc_dcts.pkl" #define pickle filename
    #If picked dicts exist, ask user to delete them. 
    if os.path.exists(c_dir + "\\" + pkl_fnm):
        incr_inp = True
        while incr_inp:
            del_a = input("\nWould you like to delete pickled dicts? (Y/N): ")
            if del_a.capitalize() == "Y": #Regenerate and save if deleted, load in if not.
                os.remove(c_dir + "\\" + pkl_fnm) #delete pickled files
                img_Nms, imgs = load_save_dicts(pkl_fnm)
                incr_inp = False
            elif del_a.capitalize() == "N":
                with open(pkl_fnm, "rb") as f: #unpickle dicts
                    img_Nms = pickle.load(f)
                    imgs = pickle.load(f)
                print("\nImage names dict and images dict are loaded in")
                incr_inp = False
            else:
                print("\nInvalid input, please try again.")
    else: #if the dicts dont exist generate and save them
        img_Nms, imgs = load_save_dicts(pkl_fnm)
    return img_Nms, imgs