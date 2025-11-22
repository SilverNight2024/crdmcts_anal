import numpy as np #to manipulate images
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt #plotting
from open_lif import use_open_lif
from gen_data import get_dta_arrs, arr2prtytbl

def exp_fit_plt(arr,err,tm,trns_nms,colors,i,end,axs,r,c):
    def exp_f(x, a, b):
        return a * np.exp(-b * x)
    k0 = -(np.log(arr[i,end])\
    - np.log(mns_arr[i,0]))\
    / (tm[end] - tm[0]) 
    #estimating the - * -b through the slope 
    #of the graph of ln of y and a through
    #assuming it more or less starts decaying
    #at a
    p0 = [arr[i,0],k0]
    bounds = ([arr[i,0] / 10 , 0] , 
            [10 * arr[i,0], np.inf])
    #absolute sigma makes param_cov actually
    #meaningful for half life error
    prs, prcov = curve_fit(exp_f, 
                                tm, arr[i,:], 
                                p0=p0, 
                                sigma = err[i,:],
                                absolute_sigma = True, 
                                bounds = bounds)
    x = np.linspace(tm[0],tm[end],100)
    fit_arr = prs[0] * np.exp(-prs[1] * x)
    axs[r,c].errorbar(tm,
        arr[i,:],
        yerr = err[i,:],
        fmt = 'o', #dashed with markers
        ecolor = colors[i], #error bar color
        markerfacecolor = (0,0,0),
        markeredgecolor = (0,0,0),
        markersize = 2, # marker size
        elinewidth = 1, # error bar line thickness
        capsize = 3, #length of error caps
        capthick = 1,
        label = trns_nms[i] + " Data")
    axs[r,c].plot(x, fit_arr, '--', color = colors[i],
                linewidth = 1,
                label = trns_nms[i] + " Fit")

def gen_plt(mn_arr, er_arr, tm_pts, trns_nms, arr_s_idx, agg_idxs):
    arr = mn_arr.copy()
    arr = arr[:,arr_s_idx:]
    end = arr.shape[1] - 1
    err = er_arr.copy()
    err = err[:,arr_s_idx:]
    tm = tm_pts.copy()
    tm = tm[arr_s_idx:]
    agg_idxs = agg_idxs.copy()
    colors = [
            (0.00, 0.20, 0.70),   # Deep Blue
            (0.85, 0.10, 0.10),   # Strong Red
            (0.00, 0.60, 0.20),   # Dark Green
            (0.90, 0.50, 0.00),   # Deep Orange
            (0.50, 0.00, 0.70),   # Deep Purple
            (0.90, 0.80, 0.00),   # Mustard Yellow
            (0.00, 0.60, 0.65),   # Dark Cyan
            (0.70, 0.20, 0.00),   # Rust Brown
            ]
    title_lst = ["MDPV FLuc vs VEEV FLuc",
                   "MDPV VP35 vs VEEV VP35",
                   "MDPV NS1 vs VEEV NS1",
                   "Summary Plot"]
    def exp_f(x, a, b):
        return a * np.exp(-b * x)  
    fig, axs = plt.subplots(2, 2, layout="constrained")
    plt.rcParams['legend.fontsize'] = 8
    for i in range(len(agg_idxs)):
            r, c = divmod(i, 2) #spits out 0,0 0,1 1,0 1,1
            if i == 3: #statement for legend
                for _, idx in enumerate(agg_idxs[i]):
                    exp_fit_plt(arr,err,tm,trns_nms,colors,idx,end,axs,r,c)
            else: #all other plots
                for _, idx in enumerate(agg_idxs[i]):
                    exp_fit_plt(arr,err,tm,trns_nms,colors,idx,end,axs,r,c)
                axs[r,c].legend(loc = 'best')  
            axs[r, c].set_yscale('log')
            axs[r, c].set_xlabel("Time After Transfection (Hrs)")
            axs[r, c].set_ylabel("Log Denoised Integrated\nPixel Intensity")
            axs[r, c].set_title(title_lst[i])
    fig.suptitle("Data-Fit Plots")
    plt.show()

def gen_tHalf_dta(mn_arr, er_arr, tm_pts, trns_nms, arr_s_idx, idxs):
    arr = mn_arr.copy()
    arr = arr[:,arr_s_idx:]
    end = arr.shape[1] - 1
    err = er_arr.copy()
    err = err[:,arr_s_idx:]
    tm = tm_pts.copy()
    tm = tm[arr_s_idx:]
    thlf_arr = np.zeros((len(idxs),2))
    def exp_f(x, a, b):
        return a * np.exp(-b * x)  
    for idx, i in enumerate(idxs):
        k0 = -(np.log(arr[i,end])\
        - np.log(mns_arr[i,0]))\
        / (tm[end] - tm[0]) 
        #estimating the - * -b through the slope 
        #of the graph of ln of y and a through
        #assuming it more or less starts decaying
        #at a
        p0 = [arr[i,0],k0]
        bounds = ([arr[i,0] / 10 , 0] , 
                [10 * arr[i,0], np.inf])
        #absolute sigma makes param_cov actually
        #meaningful for half life error
        prs, prcov = curve_fit(exp_f, 
                                    tm, arr[i,:], 
                                    p0=p0, 
                                    sigma = err[i,:],
                                    absolute_sigma = True, 
                                    bounds = bounds)
        thlf_arr[idx,0] = np.log(2) / prs[1]
        prs_err = np.sqrt(np.diag(prcov))
        thlf_arr[idx,1] =\
            thlf_arr[idx,0] *\
                  (prs_err[1] / prs[1])
        #propogating the error
    trns_nms.pop(3)
    arr2prtytbl(thlf_arr,["Half Life","Error"],trns_nms,"")

img_Nms, imgs = use_open_lif()
delim_str = "#"
tm_pts = [18,26,50,80,100,123,148,176]
trns_nms, mns_arr, nrm_mns_arr, err_arr, nrm_err_arr\
      = get_dta_arrs(img_Nms, imgs, delim_str)
idxs = np.delete(np.arange(len(trns_nms)),3)
#trns_agg_idxs = [[0,4,7],[2,6,7],[1,5,7],idxs]
gen_tHalf_dta(mns_arr, err_arr, tm_pts, trns_nms, 2, idxs)


