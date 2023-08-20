#Improved smoothening 
#Improved pause tolerance
#Tackles end issues
#Tackles start issues
#Miscellaneous improvements...further...
#Bug fixes
#Modified filtering algorithm

from scipy.signal import savgol_filter, find_peaks
import numpy as np

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

def filterMinMax(min_idx, max_idx, smooth,  g_max, g_min):
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
                    if smooth[max_idx[i]] < max_max - ((g_max/100)*largest_min_max):
                        max_idx = np.delete(max_idx, [i])
                    else:
                        i += 1
    return min_idx, max_idx

def filterMinMaxReverse(min_idx, max_idx, smooth,  g_max, g_min):
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

                    old_min_idx = min_idx[min_idx > max_idx[i]]
                    old_mins = smooth[old_min_idx]
                    largest_min_max = max(old_maxs) - min(old_mins)
                    
                except ValueError:
                    i += 1
                    failure = True
                
                if failure == False:
                    if smooth[max_idx[i]] < max_max - ((g_max/100)*largest_min_max):
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

def getRepCount(data, init_dir, g_max, g_min, control):
    smooth = smoothen(data)
    rep_idx = []
        
    #visualise(data, g_max, g_min, close=True)
    
    min_idx, _ = FindPeaks(-1*smooth)
    max_idx, _ = FindPeaks(smooth)
    
    #Filtering minimas and maximas:
    min_idx, max_idx = filterMinMax(min_idx, max_idx, smooth,  g_max, g_min)
    
    #Alternating Min-Max verification:
    min_idx, max_idx = altVerify(min_idx, max_idx)
    
    min_idx = np.flip(min_idx)
    max_idx = np.flip(max_idx)
    
    #Filtering minimas and maximas from opposite direction:
    min_idx, max_idx = filterMinMaxReverse(min_idx, max_idx, smooth,  g_max, g_min)
    min_idx = np.sort(min_idx)
    max_idx = np.sort(max_idx)
    
    #Alternating Min-Max verification:
    min_idx, max_idx = altVerify(min_idx, max_idx)
    
    control = False
    
    #Further checks:
    if(len(set(min_idx) - set([len(smooth) - 1])) >= 1 and len(set(max_idx) - set([len(smooth) - 1])) >= 1):
        if init_dir == "Decreasing" or init_dir == "MaxFirst_Decreasing":
            if(max_idx[0] <= min_idx[0]):
                max_idx = np.delete(max_idx, [0])
        if init_dir == "Increasing" or init_dir == "MinFirst_Increasing":
            if(min_idx[0] <= max_idx[0]):
                min_idx = np.delete(min_idx, [0])
    
    if(len(set(min_idx) - set([len(smooth) - 1])) >= 2 or len(set(max_idx) - set([len(smooth) - 1])) >= 2):
        control = True
    
    if(len(set(min_idx) - set([len(smooth) - 1])) >= 1 and len(set(max_idx) - set([len(smooth) - 1])) >= 1):
        if min_idx[0] < max_idx[0]: #Min first
            init_diff = data[0] - smooth[min_idx[0]]
            if(init_diff <= (10/100)*(smooth[max_idx[0]] - smooth[min_idx[0]])):
                init_dir = "MinFirst_Increasing"
            else:
                init_dir = "Decreasing"
                control = True
                
        else: #Max first
            init_diff = smooth[max_idx[0]] - data[0]
            if(init_diff <= (10/100)*(smooth[max_idx[0]] - smooth[min_idx[0]])):
                init_dir = "MaxFirst_Decreasing"
            else:
                init_dir = "Increasing"
                control = True
    
    if init_dir == "MaxFirst_Decreasing":
        if(len(set(min_idx) - set([len(smooth) - 1])) >= 1 and len(set(max_idx) - set([len(smooth) - 1])) >= 1):
            if((len(set(max_idx) - set([len(smooth) - 1])) >= 2) or (len(set(max_idx) - set([len(smooth) - 1])) >= 1 and max_idx[0] >= min_idx[0])):
                control = True
                init_dir = "Decreasing"
    if init_dir == "MinFirst_Increasing":
        if(len(set(min_idx) - set([len(smooth) - 1])) >= 1 and len(set(max_idx) - set([len(smooth) - 1])) >= 1):
            if((len(set(min_idx) - set([len(smooth) - 1])) >= 2) or (len(set(min_idx) - set([len(smooth) - 1])) >= 1 and min_idx[0] >= max_idx[0])):
                control = True
                init_dir = "Increasing"
    
    if control:
        if(init_dir == "Increasing"):
            rep_idx = min_idx
            print("Initially increasing")
        elif(init_dir == "Decreasing"):
            rep_idx = max_idx
            print("Initially decreasing")

        return len(rep_idx), init_dir, control
    
    print(init_dir)
    return 0, init_dir, control

def resize(offset, data): #Needs to be created to avoid exceeding memory limit
    return offset, data