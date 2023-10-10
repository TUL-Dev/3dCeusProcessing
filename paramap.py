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

def data_fit(TIC,normalizer):
    normalizedLogParams, normalizedLogParamCov = curve_fit(lognormal, TIC[:,0], TIC[:,1], p0=(1.0, 0.0,1.0),bounds=([0.,0., 0.], [np.inf, np.inf, np.inf]),method='trf')#p0=(1.0,3.0,0.5,0.1) ,**kwargs
    popt = normalizedLogParams

    auc = popt[0]
    mu = popt[1]
    sigma = popt[2]
    mtt = np.exp(mu+(sigma**2/2))
    wholeCurve = lognormal(TIC[:,0], auc, mu, sigma)
    tp = np.exp(mu - (sigma**2))
    pe = np.max(wholeCurve)

    # Filters to block any absurd numbers based on really bad fits. 
    if tp > TIC[-1,0] or mtt > TIC[-1,0]*2 or pe > 1 or auc > 1e+04: raise RuntimeError
    
    params = np.array([auc, pe, tp, mtt])

    wholeCurve *= normalizer;
    return params, popt, wholeCurve;

def lognormal(x, auc, mu, sigma):      
    curve_fit=(auc/(2.5066*sigma*x))*np.exp((-1/2)*(((np.log(x)-mu)/sigma)**2)) 
    return np.nan_to_num(curve_fit)

def generate_TIC(window, mask, times, compression, voxelscale):
    TICtime=times;TIC=[]; 
    bool_mask = np.array(mask, dtype=bool)
    for t in range(0,window.shape[3]):
        tmpwin = window[:,:,:,t];      
        TIC.append(np.exp(tmpwin[bool_mask]/compression).mean()*voxelscale);
        # TIC.append(np.around((tmpwin[bool_mask]/compression).mean()*voxelscale, decimals=1)); 
    TICz = np.array([TICtime,TIC]).astype('float64'); TICz = TICz.transpose();
    TICz[:,1]=TICz[:,1]-np.mean(TICz[0:2,1]);#Substract noise in TIC before contrast.
    if TICz[np.nan_to_num(TICz)<0].any():#make the smallest number in the TIC 0.
        TICz[:,1]=TICz[:,1]+np.abs(np.min(TICz[:,1]));
    else:
        TICz[:,1]=TICz[:,1]-np.min(TICz[:,1]);
    return TICz;

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

    xlist = np.arange(min(xmask), max(xmask)+windSize[0], windSize[0])
    ylist = np.arange(min(ymask), max(ymask)+windSize[1], windSize[1])
    zlist = np.arange(min(zmask), max(zmask)+windSize[2], windSize[2])
    final_map = np.zeros([img.shape[0], img.shape[1], img.shape[2], 4]).astype(np.double)
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
                    params, popt, wholecurve = data_fit(cur_TIC,normalizer);
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

    masterParamap = paramap(image, xlist, ylist, zlist, header[1:4], header[4], 'BolusLognormal', compressValue, int(windowHeightValue*header[1]), int(windowWidthValue*header[2]), int(windowDepthValue*header[3]))
    maxAuc = 0
    minAuc = 9999
    maxPe = 0
    minPe = 9999
    maxTp = 0
    minTp = 9999
    maxMtt = 0
    minMtt = 9999
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


main("/Volumes/CREST Data/David_S_Data/RRX Data/2017D000-m005.nii.gz", "/Volumes/CREST Data/David_S_Data/RRX Data/nifti_segmentation_QUANTUS/2017D000-m005_segmentation.nii.gz", 4, 4, 4, "/Volumes/CREST Data/David_S_Data/RRX Data/nifti_segmentation_QUANTUS/2017D000-m005_paramap.nii.gz")