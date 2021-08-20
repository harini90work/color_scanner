import cv2
import cvzone
import numpy as np
import os

#Image Functions used in the project
def display(img, name = 'Preview', wait_sec = 0):
    '''
    Preview the Image

    Parameters
    ----------
    img : Image
        Input Image.
    name : Str, optional
        Name of the window. The default is 'Preview'.
    wait_sec : int, optional
        How much time the window needs to be displayed. The default is 0.

    Returns
    -------
    None.

    '''
    cv2.imshow(name, img)
    if cv2.waitKey(wait_sec) & 0xFF == ord('q'):
        pass
    cv2.destroyAllWindows()   
    
    
def resize_image(img, ref_img = None, color = (0,0,0), fx = None, fy = None, interpolation = None):
    '''
    Resize the Image

    Parameters
    ----------
    img : Numpy Image
        Input Imge.
    ref_img : Numpy Image, optional
        Reference Image to considered to resize the image. The default is None.
    color : Tuple, optional
        RGB color to be filled in the remaining area. The default is (0,0,0).
    fx : float, optional
        Number of time the image to be exanded in X-axis. The default is None.
    fy : float, optional
         Number of time the image to be exanded in y-axis. The default is None.
    interpolation : str, optional
        Interpolation Method to be used while resizing the image. The default is None.

    Returns
    -------
    image_final : Numpy Image
        Resized Image.

    '''
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
        ymin = min(img.shape[0],image_final.shape[0])
        xmin = min(img.shape[1],image_final.shape[1]) 
        roi = img[0:ymin,0:xmin]
        
        image_final[0:ymin,0:xmin] =  roi
        #display(roi)
        #print(img.shape, image_final.shape)
        #print(yy,ht, xx, wd)
        #image_final[yy:yy+ht, xx:xx+wd] = img
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
    '''
    Function to read the Mask File

    Parameters
    ----------
    filename : str
        Full path where the mask file is availabe.

    Returns
    -------
    img : Numpy Image
        Mask Image in Grayscale mode.

    '''
    img = readimg(filename)
    if(img is not None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def rgb_to_hex(rgb):
    '''
    Conver the RGB value to Hex code

    Parameters
    ----------
    rgb : TYPE
        RGB Color.

    Returns
    -------
    Str
        Hex equivalent of RGB.

    '''
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
