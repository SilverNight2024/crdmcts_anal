"""This file and its contents exist to hold and support show_imgs"""
import os #To interpret file path
import numpy as np #to manipulate images
from prettytable import PrettyTable
from PIL import Image #for image creation
import matplotlib.pyplot as plt #for image showing

def get_key_parts(in_dict, delim_str):
    """Takes a dictionary whos keys are split by a delimeter into a keys 
    that form a tree with the base of super keys and the branches of sub keys
    and returns an unordered list of unique sets of those keys"""
    all_keys = list(in_dict.keys())
    #preallocates the lists to have nothing and the length of all_keys
    sup_keys = [None] * len(all_keys) 
    sub_keys = [None] * len(all_keys)
    idx = 0
    for i, k in enumerate(all_keys): #enumerate gives the index and value of an iterable
        idx = k.find(delim_str)
        sup_keys[i] = k[0:idx]
        sub_keys[i] = k[idx+len(delim_str):]
    return list(sorted(set(sup_keys))), list(sorted(set(sub_keys)))

def show_imgs(nm_dict, img_dict, delim_str):
    """ With a dict of images and of their names where the keys follow the 
    convention discussed in get_key_parts, get a subset of those images by
    prompting the user and then further allow the user to only open images
    with a certain average brightness compared to other images in the same
    sup_str + delim_str + sub_str key
    """
    sup_kys, sub_kys = get_key_parts(nm_dict, delim_str)
    table = PrettyTable()
    print("When prompted, please put input corresponding\n" +
           "to the prompt number and desired img key")
    table.field_names = ["Prompt Input", "First Prompt","Second Prompt"]
    for i in range(max(len(sup_kys), len(sub_kys))):
        if ((i < len(sup_kys)) & (i < len(sub_kys))):
            tbl_row = [i, sup_kys[i], sub_kys[i]]
        elif ((i < len(sup_kys)) & (i >= len(sub_kys))):
            tbl_row = [i,sup_kys[i],"N/A"]
        elif ((i >= len(sup_kys)) & (i < len(sub_kys))):
            tbl_row = [i,"N/A",sub_kys[i]] 
        table.add_row(tbl_row)
    print(table)
    more_imgs = True #set while loop
    while more_imgs: #allow for the opening of multiple sets of images
        bad = True
        while bad: #make sure input 1 is valid
            sup_kys_idx = input("Prompt 1. Please Input Integer " +
                                "Representing First Key Component: ")
            try: #will break if no integer
                sup_kys_idx = int(sup_kys_idx) #input only returns strings
            except ValueError:
                print("Input is not an Integer!")
                continue
            if ((type(sup_kys_idx) is int) & (sup_kys_idx > -1) & (sup_kys_idx < len(sup_kys))):
                bad = False #Check for valid input
            else:
                print("Input is an invalid integer, please try again")
        bad = True
        while bad: #make sure input 2 is valid
            sub_kys_idx = input("Prompt 2. Please Input Integer " +
                            "Representing Second Key Component: ")
            try:
                sub_kys_idx = int(sub_kys_idx)
            except ValueError:
                print("Input is not an Integer!")
            if ((type(sub_kys_idx) is int) & (sub_kys_idx > -1) & (sub_kys_idx < len(sub_kys))):
                bad = False
            else:
                print("Input is an invalid integer, please try again")
                continue
        req_key = sup_kys[sup_kys_idx] + delim_str + sub_kys[sub_kys_idx]
        img_mns = img_dict[req_key].mean(axis=(1,2)) #gets mean for brightness
        table = PrettyTable()
        print("You will be asked to input a value. Images with average brightness\n" +
               "equal to or above that value in the selection will be opened.\n" +
              "Below is a summary table of brightness values in the selection: ")
        table.field_names = ["Mean", "Maximum","Minimum"]
        table.add_row([np.mean(img_mns),np.max(img_mns),np.min(img_mns)])
        print(table)
        bad = True
        while bad: #make sure input is valid
            img_thr = input("Please input image open threshold (float or int only): ")
            try:
                img_thr = float(img_thr)
            except ValueError:
                print("Input is not a float!")
                continue
            if ((type(img_thr) is float) & (img_thr <= np.max(img_mns))):
                thresh_vec = img_mns >= img_thr #get logical mask
                proc_q = input(f"This will generate {np.sum(thresh_vec)} imgs. Proceed? (Y/N): ")
                if proc_q .capitalize() == "Y": #Open images is yes
                    bad = False
                else:
                    print("Will not proceed, returning to threshold prompt.")
            else:
                print("Threshold will not open an image, please try again")
        for i,img in enumerate(img_dict[req_key][thresh_vec]):
            pil_img = Image.fromarray(img) #convert to pillow object
            fig = plt.figure()
            fig.canvas.manager.set_window_title(nm_dict[req_key][i])  # real window title
            plt.imshow(img) 
            plt.axis("off")
            plt.title(nm_dict[req_key][i])
        plt.show()
        bad = True
        while bad:
            go_on = input("Continue to open new images? (Y/N): ")
            if go_on.capitalize() == "Y": #Open more images if yes
                more_imgs = True
                bad = False
            elif go_on.capitalize() == "N":
                more_imgs = False
                bad = False
            else:
                print("Invalid input, please try again.")
        



