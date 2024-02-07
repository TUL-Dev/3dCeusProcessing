# Give NIFTI mask and 4D CEUS image, returns NIFTI paramap of AUC, PE, TP, and MTT

# For now, paramap must be viewed on QuantUS. Will not render usable results in other NIFTI viewers
# due to formatting decisions made to minimize file size

import time
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from math import exp
import sys
from pathlib import Path
import os
import nibabel as nib
import scipy

def data_fit(TIC, normalizer, timeconst):
    #Fitting function
    #Returns the parameters scaled by normalizer
    #Beware - all fitting - minimization is done with data normalized 0 to 1. 
    #kwargs = {"max_nfev":5000}
    popt, pcov = curve_fit(bolus_lognormal, TIC[:,0], TIC[:,1], p0=(1.0,3.0,0.5,0.1),bounds=([0., 0., 0., -1.], [np.inf, np.inf, np.inf, 10.]),method='trf')#p0=(1.0,3.0,0.5,0.1) ,**kwargs
    popt = np.around(popt, decimals=1);
    auc = popt[0]; rauc=normalizer*popt[0]; mu=popt[1]; sigma=popt[2]; t0=popt[3]; mtt=timeconst*np.exp(mu+sigma*sigma/2);
    tp = timeconst*exp(mu-sigma*sigma); wholecurve = bolus_lognormal(TIC[:,0], popt[0], popt[1], popt[2], popt[3]); pe = np.max(wholecurve); # took out pe normalization
    rt0 = t0;# + tp;
    
    # Get error parameters
    residuals = TIC[:,1] - bolus_lognormal(TIC[:,0], popt[0], mu, sigma, t0);
    ss_res = np.sum(residuals[~np.isnan(residuals)]**2);# Residual sum of squares
    ss_tot = np.sum((TIC[:,1]-np.mean(TIC[:,1]))**2);# Total sum of squares
    r_squared = 1 - (ss_res / ss_tot);# R squared
    RMSE = (np.sum(residuals[~np.isnan(residuals)]**2)/(residuals[~np.isnan(residuals)].size-2))**0.5;#print('RMSE 1');print(RMSE);# RMSE
    rMSE = mean_squared_error(TIC[:,1], bolus_lognormal(TIC[:,0], popt[0], mu, sigma, t0))**0.5;#print('RMSE 2');print(rMSE);

    # Filters to block any absurb numbers based on really bad fits. 
    if tp > TIC[-1,0]: tp = TIC[-1,0]
    if mtt > TIC[-1,0]*2: mtt = TIC[-1,0]*2
    if rt0 > TIC[-1,0]: rt0 = TIC[-1,0]
    # if tp > 220: tp = 220; #pe = 0.1; rauc = 0.1; rt0 = 0.1; mtt = 0.1;
    # if rt0 > 160: rt0 = 160; #pe = 0.1; rauc = 0.1; tp = 0.1; mtt = 0.1;
    # if mtt > 2000: mtt = 2000; #pe = 0.1; rauc = 0.1; tp = 0.1; rt0 = 0.1;
    if pe > 1e+07: pe = 1e+07;
    if auc > 1e+08: auc = 1e+08;

    if RMSE > 0.3: raise RuntimeError

    params = np.array([auc, pe, tp, mtt, rt0]);

    return params, popt, RMSE;

# def data_fit(TIC,normalizer):
#     normalizedLogParams, normalizedLogParamCov = curve_fit(lognormal, TIC[:,0], TIC[:,1], p0=(1.0, 0.0,1.0),bounds=([0.,0., 0.], [np.inf, np.inf, np.inf]),method='trf')#p0=(1.0,3.0,0.5,0.1) ,**kwargs
#     popt = normalizedLogParams

#     auc = popt[0]
#     mu = popt[1]
#     sigma = popt[2]
#     mtt = np.exp(mu+(sigma**2/2))
#     wholeCurve = lognormal(TIC[:,0], auc, mu, sigma)
#     tp = np.exp(mu - (sigma**2))
#     pe = np.max(wholeCurve)

#     # Filters to block any absurd numbers based on really bad fits. 
#     if tp > TIC[-1,0] or mtt > TIC[-1,0]*2 or pe > 1 or auc > 1e+04: raise RuntimeError
    
#     params = np.array([auc, pe, tp, mtt])

#     wholeCurve *= normalizer;
#     return params, popt, wholeCurve;

def lognormal(x, auc, mu, sigma):      
    curve_fit=(auc/(2.5066*sigma*x))*np.exp((-1/2)*(((np.log(x)-mu)/sigma)**2)) 
    return np.nan_to_num(curve_fit)

def bolus_lognormal(x, auc, mu, sigma, t0):        
    curve_fit=(auc/(2.5066*sigma*(x-t0)))*np.exp(-1*(((np.log(x-t0)-mu)**2)/(2*sigma*sigma))) 
    return np.nan_to_num(curve_fit)

def generate_TIC(window, mask, times, compression, voxelscale):
    TICtime=times;TIC=[]; 
    bool_mask = np.array(mask, dtype=bool)
    for t in range(0,window.shape[3]):
        tmpwin = window[:,:,:,t];      
        TIC.append(np.around(np.exp(tmpwin[bool_mask]/compression).mean()/voxelscale, decimals=1));
        # TIC.append(np.exp(tmpwin[bool_mask]/compression).mean()*voxelscale);
        # TIC.append(np.around((tmpwin[bool_mask]/compression).mean()*voxelscale, decimals=1)); 
    TICz = np.array([TICtime,TIC]).astype('float64'); TICz = TICz.transpose();
    TICz[:,1]=TICz[:,1]-np.mean(TICz[0:2,1]);#Substract noise in TIC before contrast.
    if TICz[np.nan_to_num(TICz)<0].any():#make the smallest number in the TIC 0.
        TICz[:,1]=TICz[:,1]+np.abs(np.min(TICz[:,1]));
    else:
        TICz[:,1]=TICz[:,1]-np.min(TICz[:,1]);
    return TICz;

def generate_TIC_2d(window, mask, times, compression, voxelscale):
    TICtime=times;TIC=[]; 
    bool_mask = np.array(mask, dtype=bool)
    for t in range(0,window.shape[2]):
        tmpwin = window[:,:,t];      
        TIC.append(np.around(np.exp(tmpwin[bool_mask]/compression).mean()/voxelscale, decimals=1));
        # TIC.append(np.exp(tmpwin[bool_mask]/compression).mean()*voxelscale);
        # TIC.append(np.around((tmpwin[bool_mask]/compression).mean()*voxelscale, decimals=1)); 
    TICz = np.array([TICtime,TIC]).astype('float64'); TICz = TICz.transpose();
    TICz[:,1]=TICz[:,1]-np.mean(TICz[0:2,1]);#Substract noise in TIC before contrast.
    if TICz[np.nan_to_num(TICz)<0].any():#make the smallest number in the TIC 0.
        TICz[:,1]=TICz[:,1]+np.abs(np.min(TICz[:,1]));
    else:
        TICz[:,1]=TICz[:,1]-np.min(TICz[:,1]);
    return TICz;

# def generate_TIC_mc(window, bboxes, times, compression):
#     TICtime = []
#     TIC = []
#     areas = []
#     for t in range(0, window.shape[0]):
#         if bboxes[t] is not None:
#             tmpwin = window[t]
#             bool_mask = np.zeros(tmpwin.shape, dtype=bool)
#             x0, y0, x_len, y_len = bboxes[t]
#             if y0 + y_len >= bool_mask.shape[0]:
#                 y_len = bool_mask.shape[0] - y0 - 1
#             if x0 + x_len >= bool_mask.shape[1]:
#                 x_len = bool_mask.shape[0] - x0 - 1
#             bool_mask[y0 : y0 + y_len, x0 : x0 + x_len] = True
#             # for x in range(x_len):
#             #     bool_mask[y0,x] = True
#             #     bool_mask[y0+y_len, x] = True
#             # for y in range(y_len):
#             #     bool_mask[y, x0] = True
#             #     bool_mask[y, x0+x_len] = True
#             # bool_mask = binary_fill_holes(bool_mask)
#             numPoints = len(np.where(bool_mask > 0)[0])
#             TIC.append(np.exp(tmpwin[bool_mask] / compression).mean())
#             TICtime.append(times[t])
#             areas.append(numPoints)

#     TICz = np.array([TICtime, TIC]).astype("float64")
#     TICz = TICz.transpose()
#     TICz[:, 1] = TICz[:, 1] - np.mean(
#         TICz[0:2, 1]
#     )  # Subtract noise in TIC before contrast
#     if TICz[np.nan_to_num(TICz) < 0].any():  # make the smallest number in TIC 0
#         TICz[:, 1] = TICz[:, 1] + np.abs(np.min(TICz[:, 1]))
#     else:
#         TICz[:, 1] = TICz[:, 1] - np.min(TICz[:, 1])
#     return TICz, np.round(np.mean(areas), decimals=2)

def generate_TIC_2d_MC(window, mask, times, compression):
    TICtime = []
    TIC = []
    areas = []
    summed_window = np.transpose(np.sum(np.squeeze(window), axis=3))
    for t in range(0, mask.shape[2]):
        tmpwin = summed_window[t]
        bool_mask = np.array(mask[t]).astype(bool)
        numPoints = len(np.where(bool_mask > 0)[0])
        if numPoints == 0:
            continue
        TIC.append(np.exp(tmpwin[bool_mask] / compression).mean())
        TICtime.append(times[t])
        areas.append(numPoints)

    TICz = np.array([TICtime, TIC]).astype("float64")
    TICz = TICz.transpose()
    TICz[:, 1] = TICz[:, 1] - np.mean(
        TICz[0:2, 1]
    )  # Subtract noise in TIC before contrast
    if TICz[np.nan_to_num(TICz) < 0].any():  # make the smallest number in TIC 0
        TICz[:, 1] = TICz[:, 1] + np.abs(np.min(TICz[:, 1]))
    else:
        TICz[:, 1] = TICz[:, 1] - np.min(TICz[:, 1])
    return TICz


# def generate_TIC(window, times, compression,voxelscale):
#     TICtime=[];TIC=[];
#     for t in range(0,times.shape[0]):
#         TICtime.append(times[t]); 
#         tmpwin = window[t,:,:,:];       
#         TIC.append(np.around(np.exp(tmpwin[~np.isnan(tmpwin)]/compression).mean()/voxelscale, decimals=1));
#     TICz = np.array([TICtime,TIC]).astype('float64'); TICz = TICz.transpose();
#     TICz[:,1]=TICz[:,1]-np.mean(TICz[0:2,1]);#Substract noise in TIC before contrast.
#     if TICz[np.nan_to_num(TICz)<0].any():#make the smallest number in the TIC 0.
#         TICz[:,1]=TICz[:,1]+np.abs(np.min(TICz[:,1]));
#     else:
#         TICz[:,1]=TICz[:,1]-np.min(TICz[:,1]);
#     return TICz;

# def organize_points_by_time(pointsPlotted: np.array) -> list:
#     cur_time = pointsPlotted[0,2]
#     organized_points = []
#     cur_time_points = []
#     for point in pointsPlotted:
#         if point[2] == cur_time:
#             cur_time_points.append(point)
#         else:
#             organized_points.append(cur_time_points)
#             cur_time = point[2]
#             cur_time_points = [point]
#     organized_points.append(cur_time_points)
#     return organized_points

# def get_bboxes(pointsPlotted: np.array) -> list:
#     organized_points = organize_points_by_time(pointsPlotted)

#     bboxes = []
#     for time_sector in organized_points:
#         x0 = np.min(time_sector[:,0])
#         w = np.max(time_sector[:,0]) - x0
#         y0 = np.min(time_sector[:,1])
#         h = np.max(time_sector[:,1]) - y0
#         bboxes.append((x0, y0, w, h))

#     return bboxes

def get_bbox(x_coords: np.array, y_coords: np.array, windSize_x: int, windSize_y: int) -> np.array:
    x0 = np.min(x_coords)
    y0 = np.min(y_coords)
    w = np.max(x_coords)-x0
    h = np.max(y_coords)-y0

    pix_x0s = np.arange(x0, x0+w, windSize_x)[:-1]
    pix_y0s = np.arange(y0, y0+h, windSize_y)[:-1]
    pix_bboxes = np.transpose(np.meshgrid(pix_x0s, pix_y0s))
    pix_bboxes = np.pad(pix_bboxes, [(0,0), (0,0), (0,2)], mode='constant', constant_values=0)
    pix_bboxes[:,:,2] = windSize_x
    pix_bboxes[:,:,3] = windSize_y
    return pix_bboxes

def paramap2d(img, mask, res, time, tf, compressfactor, windSize_x, windSize_y, mc):
    # windSize_x = 1; windSize_y = 1; windSize_z = 1
    print('*************************** Starting Parameteric Map *****************************')
    # print('Prep For Loop:');print(str(datetime.now()));
    # start_time = datetime.now()
    #1a. Windowing and image info
    global windSize, voxelscale, compression, imgshape, timeconst, times, xlist, ylist, zlist, typefit;
    windSize = (windSize_x, windSize_y);
    voxelscale = res[0]*res[1]*res[2];
    compression = compressfactor; 
    imgshape = img.shape;
    typefit = tf;
    #img = img - np.mean(img[:,0:4,:,:,:,:],axis=1);img[img < 1]=0;

    # Make expected calculation time

    #1b. Creat time point and position lists
    timeconst = time;#time/(img.shape[1]+1);
    times = [i*time for i in range(1, img.shape[2]+1)];

    if mc:
        bbox_shape_x = 0
        bbox_shape_y = 0
        pixel_bboxes = []
        for t in range(mask.shape[2]):
            xmask, ymask = np.where(mask[:,:,t]>0)
            if len(xmask):
                pixel_bboxes.append(get_bbox(xmask, ymask, windSize_x, windSize_y))
                if not bbox_shape_x:
                    bbox_shape_x = pixel_bboxes[-1].shape[0]
                    bbox_shape_y = pixel_bboxes[-1].shape[1]
            else:
                pixel_bboxes.append(None)

        final_map = np.zeros((img.shape[0], img.shape[1], img.shape[2], 5))
        for x in range(bbox_shape_x):
            for y in range(bbox_shape_y):
                segMask = np.zeros((img.shape[2], img.shape[1], img.shape[0]))
                for t, bbox in enumerate(pixel_bboxes):
                    if bbox is not None:
                        x0, y0, x_len, y_len = bbox[x,y]
                        segMask[t, y0 : y0 + y_len, x0 : x0 + x_len] = 1

                cur_TIC = generate_TIC_2d_MC(img, segMask, times, compression)
                normalizer = np.max(cur_TIC[:,1]);
                cur_TIC[:,1] = cur_TIC[:,1]/normalizer;

                # Bunch of checks
                if np.isnan(np.sum(cur_TIC[:,1])):
                    print('STOPPED:NaNs in the VOI')
                    return;
                if np.isinf(np.sum(cur_TIC[:,1])):
                    print('STOPPED:InFs in the VOI')
                    return;

                # Do the fitting
                try:
                    params, popt, wholecurve = data_fit(cur_TIC,normalizer, timeconst);
                    index_points = np.transpose(np.where(np.transpose(segMask)>0))
                    final_map[index_points] = params
                except RuntimeError:
                    # params = np.array([-1, np.max(cur_TIC[:,1]), -1, -1])
                    pass
                
        print('Paraloop ended:')
        return final_map
    
    try:
        xmask, ymask, _ = np.where(mask>0)
        xlist = np.arange(min(xmask), max(xmask)+windSize_x, windSize_x)
        ylist = np.arange(min(ymask), max(ymask)+windSize_y, windSize_y)
    except:
        print("Voxel dimensions too small! Try larger values")
        exit(1)
    final_map = np.zeros([img.shape[0], img.shape[1], 5]).astype(np.double)
    summed_img = np.sum(np.squeeze(img), axis=3)
    for x_base in range(len(xlist)):
        for y_base in range(len(ylist)):
            cur_mask = np.zeros([img.shape[0], img.shape[1]])
            indices = []
            for x in range(windSize[0]):
                cur_index = []
                cur_index.append(xlist[x_base]+x)
                for y in range(windSize[1]):
                    cur_index.append(ylist[y_base]+y)
                    indices.append(cur_index.copy())
                    cur_index.pop()
                cur_index.pop()
            sig_indices = False
            for i in indices:
                if max(summed_img[i[0],i[1]]) != 0:
                    cur_mask[i[0],i[1]] = 1
                    sig_indices = True
            if not sig_indices:
                continue
            cur_TIC = generate_TIC_2d(summed_img, cur_mask, times, 24.9,  voxelscale)
            normalizer = np.max(cur_TIC[:,1]);
            cur_TIC[:,1] = cur_TIC[:,1]/normalizer;

            # Bunch of checks
            if np.isnan(np.sum(cur_TIC[:,1])):
                print('STOPPED:NaNs in the VOI')
                return;
            if np.isinf(np.sum(cur_TIC[:,1])):
                print('STOPPED:InFs in the VOI')
                return;

            # Do the fitting
            try:
                params, popt, wholecurve = data_fit(cur_TIC,normalizer, timeconst);
                for i in indices:
                    final_map[i[0],i[1]] = params
            except RuntimeError:
                # params = np.array([-1, np.max(cur_TIC[:,1]), -1, -1])
                pass

    print('Paraloop ended:')#;print(str(datetime.now()));
    return final_map;


def paramap(img, xmask, ymask, zmask, res, time, tf, compressfactor, windSize_x, windSize_y, windSize_z):
    # windSize_x = 1; windSize_y = 1; windSize_z = 1
    print('*************************** Starting Parameteric Map *****************************')
    # print('Prep For Loop:');print(str(datetime.now()));
    # start_time = datetime.now()
    #1a. Windowing and image info
    global windSize, voxelscale, compression, imgshape, timeconst, times, xlist, ylist, zlist, typefit;
    windSize = (windSize_x, windSize_y, windSize_z);
    voxelscale = res[0]*res[1]*res[2];
    compression = compressfactor; 
    imgshape = img.shape;
    typefit = tf;
    #img = img - np.mean(img[:,0:4,:,:,:,:],axis=1);img[img < 1]=0;

    # Make expected calculation time

    #1b. Creat time point and position lists
    timeconst = time;#time/(img.shape[1]+1);
    times = [i*time for i in range(1, img.shape[3]+1)];

    try:
        xlist = np.arange(min(xmask), max(xmask)+windSize[0], windSize[0])
        ylist = np.arange(min(ymask), max(ymask)+windSize[1], windSize[1])
        zlist = np.arange(min(zmask), max(zmask)+windSize[2], windSize[2])
    except:
        print("Voxel dimensions too small! Try larger values")
        exit(1)
    final_map = np.zeros([img.shape[0], img.shape[1], img.shape[2], 5]).astype(np.double)
    for x_base in range(len(xlist)):
        for y_base in range(len(ylist)):
            for z_base in range(len(zlist)):
                cur_mask = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
                indices = []
                for x in range(windSize[0]):
                    cur_index = []
                    cur_index.append(xlist[x_base]+x)
                    for y in range(windSize[1]):
                        cur_index.append(ylist[y_base]+y)
                        for z in range(windSize[2]):
                            cur_index.append(zlist[z_base]+z)
                            indices.append(cur_index.copy())
                            cur_index.pop()
                        cur_index.pop()
                    cur_index.pop()
                sig_indices = False
                for i in indices:
                    if max(img[i[0],i[1],i[2]]) != 0:
                        cur_mask[i[0],i[1],i[2]] = 1
                        sig_indices = True
                if not sig_indices:
                    continue
                cur_TIC = generate_TIC(img, cur_mask, times, 24.9,  voxelscale)
                normalizer = np.max(cur_TIC[:,1]);
                cur_TIC[:,1] = cur_TIC[:,1]/normalizer;

                # Bunch of checks
                if np.isnan(np.sum(cur_TIC[:,1])):
                    print('STOPPED:NaNs in the VOI')
                    return;
                if np.isinf(np.sum(cur_TIC[:,1])):
                    print('STOPPED:InFs in the VOI')
                    return;

                # Do the fitting
                try:
                    params, popt, wholecurve = data_fit(cur_TIC,normalizer, timeconst);
                    for i in indices:
                        final_map[i[0],i[1],i[2]] = params
                except RuntimeError:
                    # params = np.array([-1, np.max(cur_TIC[:,1]), -1, -1])
                    pass

    print('Paraloop ended:')#;print(str(datetime.now()));
    return final_map;


# if __name__ == "__main__":
    # if (argsCount := len(sys.argv)) != 7:
    #     print(f"Six arguments expected, got {argsCount - 1}")
    #     raise SystemExit(2)

    # imPath = sys.argv[1] # string
    # maskPath = sys.argv[2] # string
    # windowHeightValue = sys.argv[3] # float
    # windowWidthValue = sys.argv[4] # float
    # windowDepthValue = sys.argv[5] # float
    # destinationPath = Path(sys.argv[6]) # string

def main2d(imPath, maskPath, windowHeightValue, windowWidthValue, destinationPath, bmodeSeparate=0, mc=0, manual_fps=-1):
    start = time.time()
    if not os.path.exists(imPath):
        print("The image file doesn't exist")
        raise SystemExit(1)
    if not (imPath.endswith('.nii') or imPath.endswith('.nii.gz')):
        print("CEUS image must be in NIFTI format")
        raise SystemExit(1)
    
    if not os.path.exists(maskPath):
        print("The mask file doesn't exist")
        raise SystemExit(1)
    if not (maskPath.endswith('.nii') or maskPath.endswith('.nii.gz')):
        print("CEUS image must be in NIFTI format")
        raise SystemExit(1)
    
    nibImg = nib.load(imPath, mmap=False)
    header = nibImg.header['pixdim'] # [dims, voxel dims (3 vals), timeconst, 0, 0, 0]
    dataNibImg = nibImg.get_fdata()
    image = dataNibImg.astype(np.uint8)

    nibMask = nib.load(maskPath, mmap=False)
    dataNibMask = nibMask.get_fdata()
    mask = dataNibMask.astype(np.uint8)
    if bmodeSeparate:
        mask = mask[:,:image.shape[1]]

    xlist, ylist, fitted_times = np.where( mask > 0)
    pointsPlotted = np.transpose([xlist, ylist, fitted_times])

    compressValue = 24.9 # hardcoded for now
    if manual_fps == -1:
        fps = header[4]
    else:
        fps = (1/manual_fps)

    masterParamap = paramap2d(image, mask, header[1:4], fps, 'BolusLognormal', compressValue, int(windowHeightValue/header[1]), int(windowWidthValue/header[2]), mc)
    # maxAuc = 0
    # minAuc = 99999999
    # maxPe = 0
    # minPe = 99999999
    # maxTp = 0
    # minTp = 99999999
    # maxMtt = 0
    # minMtt = 99999999
    # for i in range(len(pointsPlotted)):
    #     if masterParamap[pointsPlotted[i][0], pointsPlotted[i][1]][3] != 0:
    #         if masterParamap[pointsPlotted[i][0], pointsPlotted[i][1]][0] > maxAuc:
    #             maxAuc = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][0]
    #         if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][0] < minAuc:
    #             minAuc = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][0]
    #         if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][1] > maxPe:
    #             maxPe = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][1]
    #         if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][1] < minPe:
    #             minPe = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][1] 
    #         if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][2] > maxTp:
    #             maxTp = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][2]
    #         if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][2] < minTp:
    #             minTp = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][2]
    #         if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][3] > maxMtt:
    #             maxMtt = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][3]
    #         if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][3] < minMtt:
    #             minMtt = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1]][3]


    affine = np.eye(4)
    niiarray = nib.Nifti1Image(masterParamap, affine, dtype=np.double)
    if os.path.exists(destinationPath):
        os.remove(destinationPath)
    nib.save(niiarray, destinationPath)
    print("Total time taken (sec):", time.time() - start)
    # print([minAuc, maxAuc, minPe, maxPe, minTp, maxTp, minMtt, maxMtt])

def main(imPath, maskPath, windowHeightValue, windowWidthValue, windowDepthValue, destinationPath):
    start = time.time()
    if not os.path.exists(imPath):
        print("The image file doesn't exist")
        raise SystemExit(1)
    if not (imPath.endswith('.nii') or imPath.endswith('.nii.gz')):
        print("CEUS image must be in NIFTI format")
        raise SystemExit(1)
    
    if not os.path.exists(maskPath):
        print("The mask file doesn't exist")
        raise SystemExit(1)
    if not (maskPath.endswith('.nii') or maskPath.endswith('.nii.gz')):
        print("CEUS image must be in NIFTI format")
        raise SystemExit(1)
    
    nibImg = nib.load(imPath, mmap=False)
    header = nibImg.header['pixdim'] # [dims, voxel dims (3 vals), timeconst, 0, 0, 0]
    dataNibImg = nibImg.get_fdata()
    image = dataNibImg.astype(np.uint8)

    nibMask = nib.load(maskPath, mmap=False)
    dataNibMask = nibMask.get_fdata()
    mask = dataNibMask.astype(np.uint8)

    xlist, ylist, zlist, _ = np.where( mask > 0)
    pointsPlotted = np.transpose([xlist, ylist, zlist, _])

    compressValue = 24.9 # hardcoded for now

    masterParamap = paramap(image, xlist, ylist, zlist, header[1:4], header[4], 'BolusLognormal', compressValue, int(windowHeightValue/header[1]), int(windowWidthValue/header[2]), int(windowDepthValue/header[3]))
    maxAuc = 0
    minAuc = 99999999
    maxPe = 0
    minPe = 99999999
    maxTp = 0
    minTp = 99999999
    maxMtt = 0
    minMtt = 99999999
    for i in range(len(pointsPlotted)):
        if masterParamap[pointsPlotted[i][0], pointsPlotted[i][1],pointsPlotted[i][2]][3] != 0:
            if masterParamap[pointsPlotted[i][0], pointsPlotted[i][1],pointsPlotted[i][2]][0] > maxAuc:
                maxAuc = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][0]
            if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][0] < minAuc:
                minAuc = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][0]
            if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][1] > maxPe:
                maxPe = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][1]
            if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][1] < minPe:
                minPe = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][1] 
            if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][2] > maxTp:
                maxTp = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][2]
            if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][2] < minTp:
                minTp = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][2]
            if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][3] > maxMtt:
                maxMtt = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][3]
            if masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][3] < minMtt:
                minMtt = masterParamap[pointsPlotted[i][0],pointsPlotted[i][1],pointsPlotted[i][2]][3]


    affine = np.eye(4)
    niiarray = nib.Nifti1Image(masterParamap, affine, dtype=np.double)
    if os.path.exists(destinationPath):
        os.remove(destinationPath)
    nib.save(niiarray, destinationPath)
    print("Total time taken (sec):", time.time() - start)
    print([minAuc, maxAuc, minPe, maxPe, minTp, maxTp, minMtt, maxMtt])


# main("/Volumes/CREST Data/David_S_Data/RRX Data/2017D000-m005.nii.gz", "/Volumes/CREST Data/David_S_Data/RRX Data/nifti_segmentation_QUANTUS/2017D000-m005_segmentation.nii.gz", 6, 6, 6, "/Volumes/CREST Data/David_S_Data/RRX Data/nifti_segmentation_QUANTUS/2017D000-m005_paramap_LARGER_VOXELS_NEW.nii.gz")
# print(sys.argv[1])


# Argument 1 --> Absolute path to Original 4D Nifti File
# Argument 2 --> Absolute path to Segmentation File saved in NIFTI format
# Argument 3 --> X Voxel Dim (default value 4) (mm --> assumes pixdim in NIFTI header uses mm units)
# Argument 4 --> Y Voxel Dim (default value 4) (mm --> assumes pixdim in NIFTI header uses mm units)
# Argument 5 --> Z Voxel Dim (default value 4) (mm --> assumes pixdim in NIFTI header uses mm units)
# Argument 6 --> Absolute path of paramap destination file (in NIFTI format)
# main(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6])
main2d("/Volumes/CREST Data/David_S_Data/Prostate Sample Data/85300606_20230814_093203_0000_CEUS.nii.gz", "/Volumes/CREST Data/David_S_Data/Prostate Sample Data/nifti_segmentation_QUANTUS/const_85300606_20230814_093203_0000_CEUS.nii.gz", 6, 6, "/Volumes/CREST Data/David_S_Data/Prostate Sample Data/nifti_segmentation_QUANTUS/paramap_85300606_20230814_093203_0000_CEUS.nii.gz", 1, 0, 30)