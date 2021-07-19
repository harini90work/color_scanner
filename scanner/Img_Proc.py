import cv2
import cvzone
import numpy as np
import os
#%% Subfunctions

def blankimg(height, width, rgb_color=(0, 255, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def print_coordinate(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)

def printxy(img):
    window_name = "Find Coordinate"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, print_coordinate)
    display(img,window_name)
    
    
def display(img, name = 'Preview', wait_sec = 0):
    cv2.imshow(name, img)
    if cv2.waitKey(wait_sec) & 0xFF == ord('q'):
        pass
    cv2.destroyAllWindows()     
    

def hsv_callback(x):
    global H_low,H_high,S_low,S_high,V_low,V_high
    #assign trackbar position value to H,S,V High and low variable
    H_low = cv2.getTrackbarPos('low H','controls')
    H_high = cv2.getTrackbarPos('high H','controls')
    S_low = cv2.getTrackbarPos('low S','controls')
    S_high = cv2.getTrackbarPos('high S','controls')
    V_low = cv2.getTrackbarPos('low V','controls')
    V_high = cv2.getTrackbarPos('high V','controls')
    
def hsv_mask_area(img,hsv_low,hsv_high):
    hsv_low = np.array(hsv_low, dtype = np.uint8)
    hsv_high = np.array(hsv_high, dtype = np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[...,1] = hsv[...,1]*0.8
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    res = cv2.bitwise_and(img, img, mask=mask)
    contors,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    box = (0,0,0,0)
    for cnt in contors:
        area += cv2.contourArea(cnt)
        box = cv2.boundingRect(cnt)
    
    return mask, res, area, box


def hsv_value_detect(img):
    
    #create a seperate window named 'controls' for trackbar
    cv2.namedWindow('controls',2)
    cv2.resizeWindow("controls", 550,200);

    #create trackbars for high,low H,S,V 
    cv2.createTrackbar('low H','controls',0,179,hsv_callback)
    cv2.createTrackbar('high H','controls',179,179,hsv_callback)

    cv2.createTrackbar('low S','controls',0,255,hsv_callback)
    cv2.createTrackbar('high S','controls',255,255,hsv_callback)

    cv2.createTrackbar('low V','controls',0,255,hsv_callback)
    cv2.createTrackbar('high V','controls',255,255,hsv_callback)
    
    while True:
        hsv_low = np.array([H_low, S_low, V_low], np.uint8)
        print(hsv_low)
        #print(hsv_low)
        hsv_high = np.array([H_high, S_high, V_high], np.uint8)
        #making mask for hsv range
        mask, res, area, box = hsv_mask_area(img, hsv_low, hsv_high)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return (mask,area,hsv_low,hsv_high )


def resize_image(img, ref_img = None, color = (0,0,0), fx = None, fy = None, interpolation = None):
    if(ref_img is None):
        ref_img = img.copy()
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        
    ht, wd, cc= img.shape
    try:
        hh,ww = ref_img.shape
    except:
        hh,ww,_ = ref_img.shape
    
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2
    if(interpolation is None):
        image_final = np.full((hh,ww,cc), color, dtype=np.uint8)
        image_final[yy:yy+ht, xx:xx+wd] = img
    else:
        if((fx is None) or (fy is None)):
            image_final = cv2.resize(img, (ww, hh), interpolation = interpolation)
        else:
            image_final = cv2.resize(img, (0, 0), fx = fx, fy = fy, interpolation = interpolation)
    
    return image_final




def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports
#%%
### 2 New functions to be created
#Resize image
def readimg(filename):
    if(os.path.isfile(filename)):
        img = cv2.imread(filename)
    else:
        img = None
    return img

def readmask(filename):
    img = readimg(filename)
    if(img is not None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
#Comibing images horizontal 


#Comibing images vertically
def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)



def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb

H_low = 0
S_low = 0
V_low = 0

H_high = 179
S_high = 255
V_high = 255