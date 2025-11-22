import numpy as np #to manipulate images
import matplotlib.pyplot as plt #plotting
from open_lif import use_open_lif
from gen_data import get_dta_arrs, arr2prtytbl

def gen_plt(mn_arr, er_arr, tm_pts, trns_nms, agg_idxs, neg_bool, log_bool, norm_bool):
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
      norm_str = ""
      norm_nl = ""
      if norm_bool:
            norm_str = "Normalized "
            norm_nl = "\n"
      log_str = ""
      if log_bool:
            log_str = "log Scale "
      if neg_bool: #add or remove negative control
            for i in range(len(agg_idxs)-1):
                  agg_idxs[i].append(3)
      xtra_nl = ""
      if log_bool & ~norm_bool:
            xtra_nl = "\n"
      else:
            agg_idxs[3] = np.delete(agg_idxs[3],3)
      fig, axs = plt.subplots(2, 2, layout="constrained")
      plt.rcParams['legend.fontsize'] = 8
      for i in range(4):
            r, c = divmod(i, 2) #spits out 0,0 0,1 1,0 1,1
            if i == 3: #statement for legend
                  for _, idx in enumerate(agg_idxs[i]):
                        axs[r,c].errorbar(tm_pts,
                                    mn_arr[idx,:],
                                    yerr = er_arr[idx,:],
                                    fmt = '--o', #dashed with markers
                                    color = colors[idx], #line color
                                    ecolor = colors[idx], #error bar color
                                    markerfacecolor = (0,0,0),
                                    markeredgecolor = (0,0,0),
                                    linewidth = 1, # main line thickness
                                    markersize = 2, # marker size
                                    elinewidth = 1, # error bar line thickness
                                    capsize = 3, #length of error caps
                                    capthick = 1, #line thickness of caps
                                    label = trns_nms[idx])
                  axs[r,c].legend(loc = 'best')
            else: #all other plots
                  for _, idx in enumerate(agg_idxs[i]):
                        axs[r,c].errorbar(tm_pts,
                                    mn_arr[idx,:],
                                    yerr = er_arr[idx,:],
                                    fmt = '--o', #dashed with markers
                                    color = colors[idx], #line color
                                    ecolor = colors[idx], #error bar color
                                    markerfacecolor = (0,0,0),
                                    markeredgecolor = (0,0,0),
                                    linewidth = 1, # main line thickness
                                    markersize = 2, # marker size
                                    elinewidth = 1, # error bar line thickness
                                    capsize = 3, #length of error caps
                                    capthick = 1) #line thickness of caps
            if log_bool:
                  axs[r, c].set_yscale('log')
            axs[r, c].set_xlabel("Time After Transfection (Hrs)")
            axs[r, c].set_ylabel(log_str +  norm_str + 
                                    "Mean " +  norm_nl + 
                                    "Denoised " + xtra_nl + 
                                    "Integrated Pixel Intensity")
            axs[r, c].set_title(title_lst[i])
      
      plt.show()

img_Nms, imgs = use_open_lif()
delim_str = "#"
trns_nms, mns_arr, nrm_mns_arr, err_arr, nrm_err_arr\
      = get_dta_arrs(img_Nms, imgs, delim_str)
tm_pts_lbls = ["18 hrs","26 hrs", 
          "50 hrs","80 hrs", 
          "100 hrs","123 hrs", 
          "148 hrs","176 hrs"]
# arr2prtytbl(mns_arr,tm_pts_lbls,trns_nms,"")
# arr2prtytbl(err_arr/(mns_arr + 1e-16),tm_pts_lbls,trns_nms,"")
# arr2prtytbl(nrm_mns_arr,tm_pts_lbls,trns_nms,"")
# arr2prtytbl(nrm_err_arr/(nrm_mns_arr + 1e-16),tm_pts_lbls,trns_nms,"")
tm_pts = [18,26,50,80,100,123,148,176]
all_idxs = np.arange(mns_arr.shape[1])
trns_agg_idxs = [[0,4,7],[2,6,7],[1,5,7],all_idxs]
gen_plt(mns_arr, err_arr, tm_pts, 
        trns_nms, trns_agg_idxs, 
        True, False, False)
gen_plt(nrm_mns_arr, nrm_err_arr, 
        tm_pts, trns_nms, trns_agg_idxs, 
        True, False, True)
gen_plt(mns_arr, err_arr, tm_pts, 
        trns_nms, trns_agg_idxs, 
        False, True, False)
gen_plt(nrm_mns_arr, nrm_err_arr, 
        tm_pts, trns_nms, trns_agg_idxs, 
        False, True, True)

