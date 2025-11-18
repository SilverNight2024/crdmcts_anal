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

def req_two_idx(idx_1_up,idx_2_up):
    """Takes in two index upper bounds (must be less than) and prompts indexes"""
    bad = True
    while bad: #make sure input 1 is valid
        idx_1= input("\nPrompt 1. Please Input Integer " +
                            "Representing First Index: ")
        try: #will break if no integer
            idx_1= int(idx_1) #input only returns strings
        except ValueError:
            print("\nInput is not an Integer!")
            continue
        if ((type(idx_1) is int) & (idx_1> -1) & (idx_1< idx_1_up)):
            bad = False #Check for valid input
        else:
            print("\nInput is an invalid integer, please try again")
    bad = True
    while bad: #make sure input 2 is valid
        idx_2 = input("\nPrompt 2. Please Input Integer " +
                        "Representing Second Index: ")
        try:
            idx_2 = int(idx_2)
        except ValueError:
            print("\nInput is not an Integer!")
            continue
        if ((type(idx_2) is int) & (idx_2 > -1) & (idx_2 < idx_2_up)):
            bad = False
        else:
            print("\nInput is an invalid integer, please try again")
    return idx_1, idx_2

def show_imgs(nm_dict, img_dict, delim_str):
    """ With a dict of images and of their names where the keys follow the 
    convention discussed in get_key_parts, get a subset of those images by
    prompting the user and then further allow the user to only open images
    with a certain average brightness compared to other images in the same
    sup_str + delim_str + sub_str key. The user can also optionally get 
    information on all the image brightnesses and percentile gaps
    in the selection and open based on indexes or the threshold and can also
    optionally not open images at all.
    """
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
    while more_imgs: #allow for the opening of multiple sets of images
        print("\nHere is a table with indexes associated with image files and transcript\n" +
              "categories of the images openable through this function.")
        print(table_1)
        print("\nPlease put input corresponding to\n" +
           "the prompt number and desired img key")
        sup_kys_idx, sub_kys_idx = req_two_idx(len(sup_kys),len(sub_kys))
        req_key = sup_kys[sup_kys_idx] + delim_str + sub_kys[sub_kys_idx]
        img_mns = img_dict[req_key].mean(axis=(1,2)) #gets mean for brightness
        print("\nHere is a table with a summary of information pertaining\n" +
               "to the brightness of the images in your selection.")
        table = PrettyTable()
        table.field_names = ["Minimum","Quarter 1","Median","Quarter 3","Maximum"]
        table.add_row([np.min(img_mns),np.percentile(img_mns,25),np.percentile(img_mns,50)\
                       ,np.percentile(img_mns,75),np.max(img_mns)])
        print(table)
        bad = True
        while bad: #get more information loop
            mr_inf = input("\nWould you like more information? (Y/N): ")
            if mr_inf.capitalize() == "Y": #Give Comprehensive table
                print("\nHere is a table with the index associated with,\n" +
                      "name of, and brightness of all slected images.")
                gap_arr = np.zeros((len(nm_dict[req_key]),5))
                gap_nms = ["99-50 gap",
                           "99-95 gap", 
                           "99-96 gap", 
                           "99-97 gap", 
                           "99-98 gap"]
                for id in range(len(nm_dict[req_key])): #generate gap array
                    img = img_dict[req_key][i,:,:].copy()
                    med = np.percentile(img, 50)
                    p99 = np.percentile(img, 99)
                    p95 = np.percentile(img, 95)
                    p96 = np.percentile(img, 96)
                    p97 = np.percentile(img, 97)
                    p98 = np.percentile(img, 98)
                    p99_p50_gap = p99 - med
                    p99_p95_gap = p99 - p95
                    p99_p96_gap = p99 - p96
                    p99_p97_gap = p99 - p97
                    p99_p98_gap = p99 - p98
                    gap_arr[id,:] = [
                                    int(p99_p50_gap),
                                    int(p99_p95_gap),
                                    int(p99_p96_gap),
                                    int(p99_p97_gap),
                                    int(p99_p98_gap)]
                table = PrettyTable()
                table.add_column("Index",np.arange(0,len(img_mns)))
                table.add_column("Image Name",nm_dict[req_key])
                table.add_column("Mean Brightness",img_mns)
                for idx in range(5): #set gap sections of colums
                    table.add_column(gap_nms[idx],gap_arr[:,idx])
                print(table)
                bad_op = True
                while bad_op: #prompt to open more images
                    img_opn = input("\nWould You like to Open Images? (Y/N): ")
                    if img_opn.capitalize() == "Y": #Open more images if yes
                        show_imgs = True
                        bad_in = True
                        while bad_in: #open images based on indices or on brightness threshold
                            idx_t = input("\nOpen images across range between two idxs (I)\n" + 
                                        "or open all images above a brightness threshold (T)? (I/T): ")
                            if idx_t.capitalize() == "I": #Open images bases on indices
                                idx_t = True
                                bad_in = False
                            elif idx_t.capitalize() == "T":
                                idx_t = False
                                bad_in = False
                            else:
                                print("\nInvalid input, please try again.")
                        bad_op = False
                    elif img_opn.capitalize() == "N": #open no images if no
                        show_imgs = False
                        bad_op = False
                    else: #retry
                        print("\nInvalid input, please try again.")
                bad = False
            elif mr_inf.capitalize() == "N": #give no comprehensive table
                bad_op = True
                while bad_op: #only open more images in threshold style if wanted
                    img_opn = input("\nWould You like to Open Images? (Y/N): ") 
                    if img_opn.capitalize() == "Y": #Open more images if yes
                        show_imgs = True
                        idx_t = False
                        bad_op = False
                    elif img_opn.capitalize() == "N":
                        show_imgs = False
                        bad_op = False
                    else:
                        print("\nInvalid input, please try again.")
                bad = False
            else: #try again / bad input
                print("\nInvalid input, please try again.")
        bad = True
        if show_imgs: #if the user wants to see inages
            if idx_t: #opens based on indexes
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
                for i,img in enumerate(img_dict[req_key][np.arange(idx_s,idx_e+1)]):
                    fig = plt.figure()
                    fig.canvas.manager.set_window_title(nm_dict[req_key][i+idx_s])  # real window title
                    plt.imshow(img) 
                    plt.axis("off")
                    plt.title(nm_dict[req_key][i+idx_s])
                plt.show()
            else: #opens based on brightness threshold
                while bad: #make sure input is valid
                    img_thr = input("\nPlease input brightness threshold. Images with mean\n" +
                                    "brightness above it will open (float or int only): ")
                    try:
                        img_thr = float(img_thr)
                    except ValueError:
                        print("\nInput is not a float!")
                        continue
                    if ((type(img_thr) is float) & (img_thr <= np.max(img_mns))):
                        thresh_vec = img_mns >= img_thr #get logical mask
                        proc_q = input(f"\nThis will generate {np.sum(thresh_vec)} imgs. Proceed? (Y/N): ")
                        if proc_q .capitalize() == "Y": #Open images is yes
                            print("\nNOTE: You will have to close all image windown before continuing")
                            bad = False
                        else:
                            print("\nWill not proceed, returning to threshold prompt.")
                    else:
                        print("\nThreshold will not open an image, please try again")
                thresh_idxs = np.flatnonzero(thresh_vec)
                for i,img in enumerate(img_dict[req_key][thresh_vec]):
                    fig = plt.figure()
                    fig.canvas.manager.set_window_title(nm_dict[req_key][thresh_idxs[i]])  # real window title
                    plt.imshow(img) 
                    plt.axis("off")
                    plt.title(nm_dict[req_key][thresh_idxs[i]])
                plt.show()
        bad = True
        while bad: #ask if the user wants to open more images
            go_on = input("\nContinue to open new images? (Y/N): ")
            if go_on.capitalize() == "Y": #Open more images if yes
                more_imgs = True
                bad = False
            elif go_on.capitalize() == "N":
                more_imgs = False
                bad = False
            else:
                print("\nInvalid input, please try again.")
        



