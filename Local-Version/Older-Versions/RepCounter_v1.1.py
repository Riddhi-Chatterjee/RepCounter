from os.path import exists
import FeatureTemplates as FT
import PoseDetector
import cv2
from os.path import exists
from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt
import pandas as pd
import matplotlib.pyplot as plt

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

def getRepCount(data, init_dir): #Returns maxima indices
    #apply a Savitzky-Golay filter
    smooth = savgol_filter(data, window_length = min(len(data), 101), polyorder = min(min(len(data), 101) - 1, 4))
    peaks_idx_max = []
    if(smooth[0] <= smooth[1] and init_dir == "None"): #Initially increasing data
        init_dir = "Increasing"
    elif(init_dir == "None"): #Initially decreasing data
        init_dir = "Decreasing"
        

    if(init_dir == "Increasing"):
        smooth = -1*smooth
        peaks_idx_max, _ = find_peaks(smooth, prominence = 0.5)
        print("Initially increasing")
    else:
        peaks_idx_max, _ = find_peaks(smooth, prominence = 0.5)
        print("Initially decreasing")
    
    return len(peaks_idx_max), init_dir

def resize(offset, data): #Needs to be created to avoid exceeding memory limit
    return offset, data

init_dir = "None"
visThreshold = 0.5
object_dispatcher = FT.object_dispatcher

features = []

if exists("EssentialFeatures.csv"):
    with open("EssentialFeatures.csv", "r") as ef:
        for line in ef:
            line = line.split("\n")[0].lower()
            components = line.split(", ")
            featureType = components[0] + components[1]
            parameters = [int(x) if x.isdigit() else x.lower() for x in components[2:]]
            features.append(object_dispatcher[featureType](parameters, True, visThreshold))


data = []
offset = 0

o_fps = 5 #Dummy initialisation
pdt = PoseDetector.poseDetector("V")
cap = cv2.VideoCapture("V5.mp4")

# Finding OpenCV version:
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver)  < 3 :
    o_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS) 
else :
    o_fps = cap.get(cv2.CAP_PROP_FPS)
    
    
#pTime = 0
while True:
    success, img = cap.read()
    width  = cap.get(3)  # float `width`
    height = cap.get(4)  # float `height`
    if(str(img) == "None"):
        break
    img = pdt.findPose(img)
    lmList = pdt.findPosition(img)
    #print(lmList)
    #cTime = time.time()
    #fps = 1/(cTime-pTime)
    #pTime = cTime
    #cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3) 
    #cv2.imshow("Video", img)
    #cv2.waitKey(10)
    
    if(len(lmList) == 0):
        continue

    frameKeypoints = lmList
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
        data.append(frameFeatures[0][1][0])
        offset, data = resize(offset, data)
        extra_count = 0
        if(len(data) > 1):
            extra_count, init_dir = getRepCount(data, init_dir)
        repCount = offset + extra_count
        cv2.putText(img, "RepCount: "+str(repCount), (int(width/20), int(height/15)), cv2.FONT_HERSHEY_PLAIN, (height)/300, (0, 0, 255), 6) 
        cv2.imshow("Video", img)
        cv2.waitKey(10)
        
#Final data visualisation:
df = pd.DataFrame([])
df['y'] = data
df['x'] = range(len(data))
plt.plot(df['x'], df['y'], label='Original Height')

#apply a Savitzky-Golay filter
smooth = savgol_filter(df.y.values, window_length = 101, polyorder = 4)

#find the maximums
peaks_idx_max, _ = find_peaks(smooth, prominence = 0.5)
print(len(peaks_idx_max))

#reciprocal, so mins will become max
smooth_rec = -1*smooth

#find the mins now
peaks_idx_mins, _ = find_peaks(smooth_rec, prominence = 0.5)

plt.xlabel('Distance')
plt.ylabel('Height')


plt.plot(df['x'], smooth, label='smoothed')

#plot them
plt.scatter(df.x.values[peaks_idx_max], smooth[peaks_idx_max], s = 55,
            c = 'green', label = 'max')
plt.scatter(df.x.values[peaks_idx_mins], smooth[peaks_idx_mins], s = 55,
            c = 'black', label = 'min')
plt.legend(loc='upper left')
plt.show()
