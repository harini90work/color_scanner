import cv2
import cvzone
import numpy as np
import os

#Image Functions used in the project
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

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


def blankimg(height, width, rgb_color=(0, 255, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image
    
#%% Subfunctions
