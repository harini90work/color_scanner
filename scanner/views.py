import cv2
from django.http import JsonResponse
import numpy as np
import base64
from .color_extraction import *
try:
    from . import Img_Proc as im
except:
    import Img_Proc as im
    
def decode_img(img):
    '''
    Convert the Base 64 to Image

    Parameters
    ----------
    img : Base 64 Image
        Input Image in base 64 format.

    Returns
    -------
    img1 : 
        Decoded base 64 image to numpy image

    '''
    header, encoded = img.split(",", 1)
    img_b64decode = base64.b64decode(encoded)
    img_array = np.fromstring(img_b64decode,np.uint8)
    img1=cv2.imdecode(img_array,cv2.COLOR_BGR2RGB)
    return img1


def encode_img(img1):
    '''
    Convert NUmpy image to Base 64 Image

    Parameters
    ----------
    img1 : Numpy Image
        Input Image.

    Returns
    -------
    img1 : Base 64 String
        COnverted Image.

    '''
    retval, buffer_img= cv2.imencode('.jpg', img1)
    img1 = base64.b64encode(buffer_img)
    img1 = u'data:img/jpeg;base64,'+img1.decode('utf-8')
    return img1
    
    


def get_videofeed(request):
    '''
    Ajax Function to receive the Image frame from HTML SPACE

    Parameters
    ----------
    request : TYPE
        DESCRIPTION.

    Returns
    -------
    Dict
        Input data required to display in HTML Ajax function.

    '''
    img = request.POST.get('image')
    img1 = decode_img(img)
    orientation = request.POST.get('orientation')
    server_feed = request.POST.get('serverfeed')
    try:
        innerthreshold = int(request.POST.get('innerthreshold'))
    except:
        innerthreshold = 160
    try:
        outterthreshold = int(request.POST.get('outterthreshold'))
    except:
        outterthreshold = 160
    
    try:
        shadesbin = int(request.POST.get('shadesbin'))
    except:
        shadesbin = 14
        
    try:
        saturationcutoff = int(request.POST.get('saturationcutoff'))
    except:
        saturationcutoff = 10
        
    try:
        valuecutoff = int(request.POST.get('valuecutoff'))
    except:
        valuecutoff = 50
        
    print('received threshold', shadesbin,saturationcutoff,valuecutoff, type(valuecutoff))
    
    output, df, status, inner_circle_intensity, outercircle_intensity = col_detect_main(img1, orientation, innerthreshold, outterthreshold, shadesbin,saturation_cutoff=saturationcutoff,value_cutoff=valuecutoff )
    output = im.resize_image(output, fx=0.4, fy=0.4, interpolation = cv2.INTER_CUBIC)
    ip_img = encode_img(output)
    
    if(status):
        scanned = 'Finished'
        df = df.to_html(classes = 'table', index=False)
        #print(df)
    else:
        scanned = 'notyet'
        df = 'No Data Extracted'
    data = {
        'scanned': scanned,
        #'image': ip_img,
        'df' : df,
        'inner_circle': inner_circle_intensity, 
        'outter_circle': outercircle_intensity,
        
    }
    if(server_feed == 'true'):
        data['image'] = ip_img
    else:
        data['image'] = None
    return JsonResponse(data)
