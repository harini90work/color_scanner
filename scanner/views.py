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
    header, encoded = img.split(",", 1)
    img_b64decode = base64.b64decode(encoded)
    img_array = np.fromstring(img_b64decode,np.uint8)
    img1=cv2.imdecode(img_array,cv2.COLOR_BGR2RGB)
    return img1


def encode_img(img1):
    retval, buffer_img= cv2.imencode('.jpg', img1)
    img1 = base64.b64encode(buffer_img)
    img1 = u'data:img/jpeg;base64,'+img1.decode('utf-8')
    return img1
    
    


def get_videofeed(request):
    img = request.POST.get('image')
    img1 = decode_img(img)
    orientation = request.POST.get('orientation')
    server_feed = request.POST.get('serverfeed')
    print(server_feed,type(server_feed))
    
    output, df, status, inner_circle_intensity, outercircle_intensity = col_detect_main(img1, orientation )
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
