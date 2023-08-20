
from os.path import exists
import FeatureTemplates as FT
import PoseDetector
import cv2
from os.path import exists
from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import signal
import sys

def signal_handler(sig, frame):
    print('\nExiting...')
    visualise(data, maxima_tolerance, minima_tolerance, g_max, g_min, close=False)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def strToList(st):
    if st == '[]':
        return []
    factor = -1
    for ch in st:
        if ch != '[':
            break
        factor += 1
    if factor == 0:
        return [float(x) for x in st.split("[")[1].split("]")[0].split(", ")]
    
    sList = [x+("]"*factor) if x[len(x) - 1] != ']' else x for x in st[1:len(st)-1].split("]"*factor + ", ")]
    lst = []
    for s in sList:
        lst.append(strToList(s))
    return lst

def smoothen(data):
    #apply a Savitzky-Golay filter
    smooth = savgol_filter(data, window_length = min(len(data), 41), polyorder = min(min(len(data), 41) - 1, 4))
    #smooth = savgol_filter(smooth, window_length = min(len(smooth), 91), polyorder = min(min(len(smooth), 91) - 1, 4))
    #smooth = savgol_filter(smooth, window_length = min(len(smooth), 81), polyorder = min(min(len(smooth), 81) - 1, 4))
    #smooth = savgol_filter(smooth, window_length = min(len(smooth), 71), polyorder = min(min(len(smooth), 71) - 1, 4))
    #smooth = savgol_filter(smooth, window_length = min(len(smooth), 61), polyorder = min(min(len(smooth), 61) - 1, 4))
    #smooth = savgol_filter(smooth, window_length = min(len(smooth), 51), polyorder = min(min(len(smooth), 51) - 1, 4))
    #smooth = savgol_filter(smooth, window_length = min(len(smooth), 41), polyorder = min(min(len(smooth), 41) - 1, 4))
    return smooth

def FindPeaks(data):
    idx, ret2 = find_peaks(data, prominence = 1)
    idx = np.append(idx, len(data)-1)
    return idx, ret2

def filterMinMax(min_idx, max_idx, smooth,  maxima_tolerance, minima_tolerance, g_max, g_min):
    if(len(min_idx) >= 1 and len(max_idx) >= 1):
        if(len(min_idx) >= 2):
            i=0
            while(i < len(min_idx)):
                failure = False
                try:
                    old_min_idx = min_idx[:i]
                    old_mins = smooth[old_min_idx]
                    min_min = min(old_mins)

                    old_max_idx = max_idx[max_idx < min_idx[i]]
                    old_maxs = smooth[old_max_idx]
                    largest_min_max = max(old_maxs) - min(old_mins)

                except ValueError:
                    i += 1
                    failure = True
                
                if failure == False:
                    print(str(smooth[min_idx[i]])+" > "+"min("+str((minima_tolerance/100)*min_min)+", "+str(min_min + ((g_min/100)*largest_min_max))+")")
                    if smooth[min_idx[i]] > min_min + ((g_min/100)*largest_min_max):
                        min_idx = np.delete(min_idx, [i])
                    else:
                        i += 1
                
        if(len(max_idx) >= 2):
            i=0
            while(i < len(max_idx)):
                failure = False
                try:
                    old_max_idx = max_idx[:i]
                    #print("OldMaxIdx: "+str(old_max_idx))
                    old_maxs = smooth[old_max_idx]
                    #print("OldMaxs: "+str(old_maxs))
                    max_max = max(old_maxs)

                    old_min_idx = min_idx[min_idx < max_idx[i]]
                    old_mins = smooth[old_min_idx]
                    largest_min_max = max(old_maxs) - min(old_mins)
                    
                except ValueError:
                    i += 1
                    failure = True
                
                if failure == False:
                    #print(str(smooth[max_idx[i]])+" < "+"max("+str((maxima_tolerance/100)*max_max)+", "+str(max_max - ((g_max/100)*largest_min_max))+")")
                    if smooth[max_idx[i]] < max((maxima_tolerance/100)*max_max, max_max - ((g_max/100)*largest_min_max)):
                        max_idx = np.delete(max_idx, [i])
                    else:
                        i += 1
    return min_idx, max_idx

def filterMinMaxReverse(min_idx, max_idx, smooth,  maxima_tolerance, minima_tolerance, g_max, g_min):
    min_flag = False
    max_flag = False
    if(len(min_idx) >= 1 and min_idx[0] == len(smooth) - 1):
        min_idx = np.delete(min_idx, [0])
        min_flag = True
    if(len(max_idx) >= 1 and max_idx[0] == len(smooth) - 1):
        max_idx = np.delete(max_idx, [0])
        max_flag = True
        
    if(len(min_idx) >= 1 and len(max_idx) >= 1):
        if(len(min_idx) >= 2):
            i=0
            while(i < len(min_idx)):
                failure = False
                try:
                    old_min_idx = min_idx[:i]
                    old_mins = smooth[old_min_idx]
                    min_min = min(old_mins)

                    old_max_idx = max_idx[max_idx > min_idx[i]]
                    old_maxs = smooth[old_max_idx]
                    largest_min_max = max(old_maxs) - min(old_mins)

                except ValueError:
                    i += 1
                    failure = True
                
                if failure == False:
                    if smooth[min_idx[i]] > min((minima_tolerance/100)*min_min, min_min + ((g_min/100)*largest_min_max)):
                        min_idx = np.delete(min_idx, [i])
                    else:
                        i += 1
                
        if(len(max_idx) >= 2):
            i=0
            while(i < len(max_idx)):
                failure = False
                try:
                    old_max_idx = max_idx[:i]
                    #print("OldMaxIdx: "+str(old_max_idx))
                    old_maxs = smooth[old_max_idx]
                    #print("OldMaxs: "+str(old_maxs))
                    max_max = max(old_maxs)

                    old_min_idx = min_idx[min_idx > max_idx[i]]
                    old_mins = smooth[old_min_idx]
                    largest_min_max = max(old_maxs) - min(old_mins)
                    
                except ValueError:
                    i += 1
                    failure = True
                
                if failure == False:
                    #print(str(smooth[max_idx[i]])+" < "+"max("+str((maxima_tolerance/100)*avg_max)+", "+str(avg_max - ((g_max/100)*largest_min_max))+")")
                    if smooth[max_idx[i]] < max((maxima_tolerance/100)*max_max, max_max - ((g_max/100)*largest_min_max)):
                        max_idx = np.delete(max_idx, [i])
                    else:
                        i += 1
    
    if max_flag:
        max_idx = np.insert(max_idx, [0], [len(smooth) - 1])
    if min_flag:
        min_idx = np.insert(min_idx, [0], [len(smooth) - 1])
        
    return min_idx, max_idx 

def altVerify(min_idx, max_idx):
    idx_type = {}
    for idx in min_idx:
        idx_type[idx] = "min"
    for idx in max_idx:
        idx_type[idx] = "max"
        
    idx_lst = []
    for idx in min_idx:
        idx_lst.append(idx)
    for idx in max_idx:
        idx_lst.append(idx)    
    idx_lst.sort()
    
    new_idx_lst = []
    i = 0
    j = 0
    typ = ""
    while(j < len(idx_lst)):
        if typ == "":
            typ = idx_type[idx_lst[j]]
            
        if typ != idx_type[idx_lst[j]]: #Put average idx in new_idx_lst
            avg_idx = int(sum(idx_lst[i:j])/(j-i))
            idx_type[avg_idx] = typ
            new_idx_lst.append(avg_idx)
            i = j            
            typ = idx_type[idx_lst[j]]
        
        if typ == idx_type[idx_lst[j]]:
            j += 1
    
    if typ != "":
        avg_idx = int(sum(idx_lst[i:j])/(j-i))
        idx_type[avg_idx] = typ
        new_idx_lst.append(avg_idx)
    new_idx_lst.sort()
    
    min_idx = np.array([], dtype=np.int64)
    max_idx = np.array([], dtype=np.int64)
    
    for idx in new_idx_lst:
        if idx_type[idx] == "min":
            min_idx = np.append(min_idx, idx)
        if idx_type[idx] == "max":
            max_idx = np.append(max_idx, idx)
        
    return min_idx, max_idx

def visualise(data, maxima_tolerance, minima_tolerance, g_max, g_min, close=True):
    #Data visualisation:
    df = pd.DataFrame([])
    df['y'] = data
    df['x'] = range(len(data))
    plt.plot(df['x'], df['y'], label='Original Angle')

    smooth = smoothen(data)

    min_idx, _ = FindPeaks(-1*smooth)
    max_idx, _ = FindPeaks(smooth)
    
    #Filtering minimas and maximas:
    min_idx, max_idx = filterMinMax(min_idx, max_idx, smooth,  maxima_tolerance, minima_tolerance, g_max, g_min)
    
    #Alternating Min-Max verification:
    #min_idx, max_idx = altVerify(min_idx, max_idx)
    
    # min_idx = np.flip(min_idx)
    # max_idx = np.flip(max_idx)
    
    # #Filtering minimas and maximas from opposite direction:
    # min_idx, max_idx = filterMinMaxReverse(min_idx, max_idx, smooth,  maxima_tolerance, minima_tolerance, g_max, g_min)
    # min_idx = np.sort(min_idx)
    # max_idx = np.sort(max_idx)
    
    # #Alternating Min-Max verification:
    # min_idx, max_idx = altVerify(min_idx, max_idx)

    plt.xlabel('Frame Number')
    plt.ylabel('Angle Value')


    plt.plot(df['x'], smooth, label='smoothed')

    #plot them
    plt.scatter(df.x.values[max_idx], smooth[max_idx], s = 55,
                c = 'green', label = 'max')
    plt.scatter(df.x.values[min_idx], smooth[min_idx], s = 55,
                c = 'black', label = 'min')
    plt.legend(loc='upper left')
    
    if close:
        plt.show(block=True)
        plt.pause(0.001)
        plt.close()
    else:
        plt.show()

x = np.array(range(700), dtype=np.float64)
data = np.sin(x/10)

#Settings:
maxima_tolerance = 5
minima_tolerance = 500
g_max = 100
g_min = 100

visualise(data, maxima_tolerance, minima_tolerance, g_max, g_min, close=True)