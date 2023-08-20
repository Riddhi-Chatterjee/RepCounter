from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt

visThreshold = 0.0

essential_features = [
                    "2, A, 13, 11, 23, d"
                    ]

features = []

for line in essential_features:
    line = line.split("\n")[0].lower()
    components = line.split(", ")
    featureType = components[0] + components[1]
    parameters = [int(x) if x.isdigit() else x.lower() for x in components[2:]]
    features.append(object_dispatcher[featureType](parameters, True, visThreshold))

repCount = 0

#Settings:
g_max = 15
g_min = 30

def computeRepCount(pl, o_fps, data, init_dir, control):
    if control == "false":
        control = False
    else:
        control = True
        
    data = list(data)
    data = [float(x) for x in data]
    
    frameKeypoints = pl
    validFrame = True
    frameFeatures = []
    for feature in features:
        feature.loadData(frameKeypoints)
        feature.calculate(0, o_fps)
        if feature.isEssential == True and validFrame == True:
            for v in feature.value:
                if v == "None":
                    validFrame = False
                    break
        #if not validFrame:
        #    break

        s = ""
        for p in feature.original_parameters:
            s += ", "
            s += str(p)

        descriptor = feature.type[0]+", "+feature.type[1:]+s
        frameFeatures.append([descriptor, feature.value])
    if validFrame:
        #print(frameFeatures[0][1][0])
        data.append(frameFeatures[0][1][0])
        count = 0
        if(len(data) > 1):
            count, init_dir, control = getRepCount(data, init_dir, g_max, g_min, control)
        repCount = count
        
    #print(data)
    #print(repCount)
    ret_val = [data, repCount, init_dir, control]
    
    return str(ret_val)
