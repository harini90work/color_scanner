# -*- coding: utf-8 -*-
"""
    Last Modified on Mon Jul 12 11:31:21 2021
    
        
    Description:
    -----------
        * This is the main alogirthm file to scan the color
        *
    
    Objective:
    ---------
    1.  : Mon Jul 12 11:31:21 2021
    2.  : Mon Jul 12 11:31:21 2021
    
    Revision History:
    ----------------
        1. Rev.0.0  -   Base script created to complete the objective no.(1,) :Mon Jul 12 11:31:21 2021
                        Carried out by Harini K
    
    Script Written by : Harini K on Mon Jul 12 11:31:21 2021
"""
#% import required librar

import colorsys
import pandas as pd
import numpy as np
try:
    from . import Img_Proc as im
except:
    import Img_Proc as im
import cv2


#% Project Specific Functions

def extract_circle(img, r = 50, xoffset = 0):
    '''
     This function is used to extract the sub circle color information if dodle is correctly matched

    Parameters
    ----------
    img : 3 Channel Image
        Dodule corrected image.
    r : int, optional
        Radius of small circle. The default is 50.
    xoffset : int, optional
        Hyper Parameter to adjust the small circle size, if required to change through configuration. The default is 0.

    Returns
    -------
    circle_list : List
        Append proper circle list.

    '''
    circle_list = [(480+xoffset,200,r,'Grid 1;3'),
                   (630+xoffset,200,r,'Grid 1;4'),
                   (780+xoffset,200,r,'Grid 1;5'),
                   
                   (480+xoffset,345,r,'Grid 2;3'),
                   (630+xoffset,345,r,'Grid 2;4'),
                   (780+xoffset,345,r,'Grid 2;5'),
                   
                   (180+xoffset,490,r,'Grid 3;1'),
                   (330+xoffset,490,r,'Grid 3;2'),
                   (480+xoffset,490,r,'Grid 3;3'),
                   (630+xoffset,490,r,'Grid 3;4'),
                   (780+xoffset,490,r,'Grid 3;5'),
                   
                   (180+xoffset,635,r,'Grid 4;1'),
                   (330+xoffset,635,r,'Grid 4;2'),
                   (480+xoffset,635,r,'Grid 4;3'),
                   (630+xoffset,635,r,'Grid 4;4'),
                   (780+xoffset,635,r,'Grid 4;5'),
                   
                   (180+xoffset,790,r,'Grid 5;1'),
                   (330+xoffset,790,r,'Grid 5;2'),
                   (480+xoffset,790,r,'Grid 5;3'),
                   (630+xoffset,790,r,'Grid 5;4'),
                   (780+xoffset,790,r,'Grid 5;5'),
                   
                   ]
    return circle_list


def extract_small_circle(cropped):
    '''
    This function is used to extract color information from the given small circle list
        1. It will detect whether the sub small circle is in circle shape
        2. It will draw bounding rectangle
    Parameters
    ----------
    cropped : numpy Image
        Small circle cropped image.

    Returns
    -------
    cropped : numpy Image
        Cropped Image.
    x : int
        x.
    y : int
        y.
    w : int
        width.
    h : int
        height.

    '''
    original = cropped.copy()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(9,9),2)
    edge = cv2.Canny(blur, 1,24)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    last_min = None
    for contour in contours:
        cur_min = np.min(contour, axis = 0)
        if(last_min is None):
            last_min = cur_min
        cur_array = np.array([last_min, cur_min])
    #    print(cur_array)
        last_min = np.min(cur_array, axis = 0)
    if(last_min is None):
        last_min = np.array([[0,0]])

    last_max = None
    for contour in contours:
        cur_min = np.max(contour, axis = 0)
        
        if(last_max is None):
            last_max = cur_min
        cur_array = np.array([last_max, cur_min])
    #    print(cur_array)
        last_max = np.max(cur_array, axis = 0)
    if(last_max is None):
        last_max = np.array([[cropped.shape[0],cropped.shape[1]]])

    cv2.boundingRect(np.array([last_max,last_min]))
    x, y, w, h = cv2.boundingRect(np.array([last_max,last_min]))
    offset = 15
    x = x + offset
    h = h - 2*offset
    if(h<0):
        x = x - offset
        h = h + 2*offset
        
        
    y = y + offset
    w = w - 2*offset
    if(w<0):
        y = y - offset
        w = w + 2*offset

    
    cropped = original[y:y+w,x:x+h]
    return cropped, x,y,w,h


def rgb_to_hsv(r, g, b):
    '''
    This function is used to convert RGB code to equalent HSV code. This function is used to convert hex to number process.

    Parameters
    ----------
    r : int
        Red value.
    g : int
        Green Value.
    b : int
        Blue Value.

    Returns
    -------
    h : int
        Hue value.
    s : int
        Satuaration value.
    v : int
        Value.

    '''
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h, s, v

def rgb_to_digit(r, g, b, shades_bin = 12, max_shades = 14, brightness_bin = 10, saturation_cutoff = 10, value_cutoff = 50 ):
    '''
    This function is used to convert RGB color to Digit

    Parameters
    ----------
    r : int
        Red color intensity.
    g : int
        Green Color Intensity.
    b : int
        Blue color Intensity.
    shades_bin : int, optional
        Number of shades we need to group. The default is 12.
    max_shades : int, optional
        Maximum threshold of shades. The default is 14.
    brightness_bin : int, optional
        Number of brightness of the color to be splitted. The default is 10.
    saturation_cutoff : int, optional
        Maximum threshold of Saturation cutoff. The default is 10.
    value_cutoff : int, optional
        Value cutoff. The default is 50.

    Returns
    -------
    final_txt : str
        Final converted digit.
    rgb : tuple
        RGB color.

    '''
    h,s,v = rgb_to_hsv(r, g, b)
    v = v-0.01
    #print(h,s,v)
    #shades_bin = 12 #Maximum 14
    #max_shades = 14
    #brightness_bin = 10
    #saturation_cutoff = 10
    #value_cutoff = 50
    
    #h = 344
    #s = 9.9
    #v = 51
    
    shades_bin = min(shades_bin, max_shades)
    brightness_bin = min(brightness_bin, 10)
    degree = 360/shades_bin
    offset = degree/2
    hue_bin = int((h+offset)/degree) % shades_bin 
    hue_txt = str((hue_bin)).zfill(2)
    val_bin = int(v/brightness_bin) % brightness_bin 
    val_txt = str(val_bin).zfill(1)
    #print(val_txt,'asdf')
    if(s<saturation_cutoff):
        if(v>value_cutoff):
            hue_txt = '00'
            sat_txt = '0'
        else:
            hue_txt = '00'
            sat_txt = '0'
            val_txt = '0'
    else:
        sat_txt = '1'
    
    final_txt = hue_txt + sat_txt + val_txt
    
    hue = int(final_txt[0:2])
    sat = int(final_txt[2])
    val = int(final_txt[3])
    
    shades_bin = min(shades_bin, max_shades)
    brightness_bin = min(brightness_bin, 10)
    
    hue = (hue * shades_bin)/360
    val = val+1
    val = (val/brightness_bin)
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    rgb = tuple([int(a*255) for a in rgb])
    return final_txt, rgb


    
def detect_color(main_img, original_img = None, factor = None, shades_bin = 12, max_shades = 14, brightness_bin = 10, saturation_cutoff = 10, value_cutoff = 50):
    '''
    Main subfunction for detecting color

    Parameters
    ----------
    main_img : numpy Image
        Input Image.
    original_img : numpy Image, optional
        Reference Image. The default is None.
    factor : tuple, optional
        Set the Factor. The default is None.
    shades_bin : int, optional
        Number of shades we need to group. The default is 12.
    max_shades : int, optional
        Maximum threshold of shades. The default is 14.
    brightness_bin : int, optional
        Number of brightness of the color to be splitted. The default is 10.
    saturation_cutoff : int, optional
        Maximum threshold of Saturation cutoff. The default is 10.
    value_cutoff : int, optional
        Value cutoff. The default is 50.

    Returns
    -------
    output : numpy image
        Processed Image.
    final_result : Pandas Data Frame
        Processed Data Table.

    '''
    if(factor is None):
        factor = np.array([1,1,1])
    final_result = pd.DataFrame()
    if(original_img is None):
        original_img = main_img.copy()
        
    op_img_final = im.resize_image(original_img,fx=2,fy=2, interpolation = cv2.INTER_CUBIC)
    #%
    circles = extract_circle(op_img_final)
    cv2.imwrite('./input_image_considered.png',op_img_final)
    
    output = op_img_final.copy()
    detected_circle = 0
        
    if circles is not None:
        detected_circle = len(circles)
                        
        if(len(circles)==21):
            detected_circle = len(circles)
            output = op_img_final.copy() 
            final_result = pd.DataFrame()
            count = 1
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r, grid_id) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                #print(x,y,r)
                #if(r>40):
                #    r = 40
                cropped = output[y-r:y+r,x-r:x+r]
                cropped1 = cropped.copy()
                #im.display(cropped)
                cropped, x1, y1, w1, h1 = extract_small_circle(cropped)
                #im.display(cropped)

                cropped = factor*cropped
                cropped[cropped>255] = 255
                cropped = cropped.astype('uint8')
                cropped = cv2.blur(cropped, (13,13))
                
                cropped1 = factor*cropped1
                cropped1[cropped1>255] = 255
                cropped1 = cropped1.astype('uint8')
                cropped1 = cv2.blur(cropped1, (13,13))
                output[y-r:y+r,x-r:x+r] = cropped1
                #im.display(output)
                #
                rgb = tuple(np.median(cropped,axis=1)[0].astype(int))
                rgb = rgb[::-1]
                
                final_txt, rgb_ref = rgb_to_digit(rgb[0],rgb[1],rgb[2], shades_bin = shades_bin, max_shades = max_shades, brightness_bin = brightness_bin, saturation_cutoff = saturation_cutoff, value_cutoff=value_cutoff)
                hex_color = im.rgb_to_hex(rgb)
                #print(rgb, rgb_ref, 'asdfasdf')
                hex_color_ref = im.rgb_to_hex(rgb_ref)
                row = {
                'Circle No' : count,
                'Grid ID': grid_id,
                #'x' : x,
                #'y' : y,
                #'r' : r,
                'Hex Value': hex_color,
                'Ref hex' : hex_color_ref,
                'Number' : final_txt,
                
               # 'res' : res,
                }
                final_result = final_result.append(row,ignore_index=True)
                count += 1
                cropped = cv2.putText(output,hex_color,(x-30,y+5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0))
    
    if(len(circles)>0):
        for (x, y, r, grid_id) in circles:
            if(r>40):
                r = 40
            cv2.circle(output, (x, y), r, (0, 255, 0), 3)
                    
    output = cv2.putText(output, f'Total detected Circles:{detected_circle}',(10,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(64,127,255))
    return output, final_result


def colordetector(ip_img, orientation,  innerthreshold = 157, outterthreshold = 37, shades_bin = 12, max_shades = 14, brightness_bin = 10, saturation_cutoff = 10, value_cutoff = 50):
    '''
    This is the Main function to extract the color information

    Parameters
    ----------
    ip_img : Numpy Image
        Input Image to process.
    orientation : str
        Camera Orientation.
    innerthreshold : int, optional
        Inner Circle Threshold. The default is 157.
    outterthreshold : int, optional
        Outter Circle Threshold. The default is 37.
    shades_bin : int, optional
        Number of shades we need to group. The default is 12.
    max_shades : int, optional
        Maximum threshold of shades. The default is 14.
    brightness_bin : int, optional
        Number of brightness of the color to be splitted. The default is 10.
    saturation_cutoff : int, optional
        Maximum threshold of Saturation cutoff. The default is 10.
    value_cutoff : int, optional
        Value cutoff. The default is 50.

    Returns
    -------
    op_img : Numpy Image
        Processed Image.
    final_result : Pandas Data Frame
        Data Table.
    innercircle_intensity : int
        Inner Circle Intensity.
    outercircle_intensity : int
        OuterCircle Intensity .

    '''
    #
    if(orientation == 'P'):
        mask1file = 'mask1-p.png'
        mask2file = 'mask2-p.png'
        mask3file = 'mask3-p.png'
    else:
        mask1file = 'mask1.png'
        mask2file = 'mask2.png'
        mask3file = 'mask3.png'
    original_img = ip_img.copy()
    final_result = pd.DataFrame()
    mask1 = im.readmask(f'./static/assets/img/masks/{mask1file}')
    mask2 = im.readmask(f'./static/assets/img/masks/{mask2file}')
    mask3 = im.readmask(f'./static/assets/img/masks/{mask3file}')
    mask4 = cv2.bitwise_xor(mask2, mask3)
    #print('./static/assets/img/masks/{mask1file}')
    main_img = cv2.bitwise_and(ip_img,ip_img,mask=mask1)
    outer_circle = cv2.bitwise_and(ip_img,ip_img,mask=mask2)
    inner_circle = cv2.bitwise_and(ip_img,ip_img,mask=mask3)

    #Check Inner Circle Median Color
    contours, hierarchy = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if(len(contours)>0):
        rect = cv2.boundingRect(contours[0])
        test = inner_circle[rect[1]+10:rect[1]+rect[3]-10,rect[0]+10:rect[0]+rect[2]-10]
        test1 = np.mean(test,axis=1).mean()
        print('Inner Circle Median color', test1)
    else:
        test1 = 0

    innercircle_intensity = int(test1)
    if(test1>=innerthreshold): #157
        test1 = True
        factor = np.median(np.divide(255,test).mean(axis=1),axis=0)
        print('Before filter factor', factor)
        factor[(factor>1.4) & (factor<3)] = 1.4
        factor[factor>3] = 1
        print('After filter factor', factor)
    else:
        test1 = False
    outercircle_intensity  = 255

    #Check Outer Circle Median Color
    if(test1):
        contours, hierarchy = cv2.findContours(mask4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rect = cv2.boundingRect(contours[0])
        test = outer_circle[rect[1]+10:rect[1]+rect[3]-10,rect[0]+10:rect[0]+rect[2]-10]
        overlay = mask4[rect[1]+10:rect[1]+rect[3]-10,rect[0]+10:rect[0]+rect[2]-10]
        test = cv2.bitwise_and(test,test,mask=overlay)
        testhsv = cv2.cvtColor(test,cv2.COLOR_BGR2HSV)
        test2 = np.mean(testhsv[:,:,2],axis=1).mean()
        print('Outer Circle Median color', test2)
        outercircle_intensity  = test2

        if(test2<=outterthreshold): #37
            test2 = True
        else:
            test2 = False
    else:
        test2 = False
    op_img = ip_img.copy()
    
    if(test2):
        print('Final Factor used', factor)
        op_img, final_result = detect_color(main_img, ip_img, factor, shades_bin = shades_bin, max_shades = max_shades, brightness_bin = brightness_bin, saturation_cutoff = saturation_cutoff, value_cutoff=value_cutoff)
        if(final_result.shape[0]>0):
            final_result = final_result.sort_values('Grid ID')
    else:
        cv2.putText(op_img,'Keep Image within frame',(10,20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
        cv2.putText(op_img,f'Inner Circle Intensity: {innercircle_intensity}',(10,60),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
        cv2.putText(op_img,f'Outter Circle Intensity: {outercircle_intensity}',(10,90),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
        
    op_img = im.resize_image(op_img, original_img, interpolation = cv2.INTER_CUBIC)
    #op_img = im.resize_image(op_img, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
    return op_img, final_result, innercircle_intensity, outercircle_intensity



#%
def col_detect_main(ip_img, orientation, innerthreshold, outterthreshold, shades_bin = 14, max_shades = 99, brightness_bin = 10, saturation_cutoff = 10, value_cutoff = 50):
    '''
    Master function to process the entire Image

    Parameters
    ----------
    ip_img : Numpy Image
        Input Image to process.
    orientation : str
        Camera Orientation.
    innerthreshold : int, optional
        Inner Circle Threshold. The default is 157.
    outterthreshold : int, optional
        Outter Circle Threshold. The default is 37.
    shades_bin : int, optional
        Number of shades we need to group. The default is 12.
    max_shades : int, optional
        Maximum threshold of shades. The default is 14.
    brightness_bin : int, optional
        Number of brightness of the color to be splitted. The default is 10.
    saturation_cutoff : int, optional
        Maximum threshold of Saturation cutoff. The default is 10.
    value_cutoff : int, optional
        Value cutoff. The default is 50.

    Returns
    -------
    output : Numpy Image
        Processed Image.
    df : Pandas Data Frame
        Final Result DataFrame.
    status : Bool
        Whether 21 circles extracted.
    innercircle_intensity : int
        Inner Circle Intensity.
    outercircle_intensity : int
        OuterCircle Intensity .


    '''
    if(orientation == 'P'):
        mask4file = 'mask4-p.png'
        mask5file = 'mask5-p.png'
    else:
        mask4file = 'mask4.png'
        mask5file = 'mask5.png'
    status = False
    mask4 = im.readimg(f'./static/assets/img/masks/{mask4file}')
    mask5 = im.readimg(f'./static/assets/img/masks/{mask5file}')
    
    ip_img = im.resize_image(ip_img,mask4,interpolation = cv2.INTER_CUBIC)
    mask6 = cv2.add(mask4, mask5)
    mask6 = cv2.cvtColor(mask6, cv2.COLOR_BGR2GRAY)
    overlay = im.blankimg(mask6.shape[0], mask6.shape[1])
    overlay = cv2.bitwise_and(overlay, overlay, mask = mask6)
    frame = ip_img.copy()
    output, df, inner_circle_intensity, outercircle_intensity = colordetector(frame, orientation,  innerthreshold, outterthreshold, shades_bin = shades_bin, max_shades = max_shades, brightness_bin = brightness_bin, saturation_cutoff = saturation_cutoff, value_cutoff=value_cutoff)

    output = cv2.bitwise_xor(overlay,output)
    if(df.shape[0] == 21):
        status = True
    return output, df, status, inner_circle_intensity, outercircle_intensity
    
print('All Functions Loaded successfully')

#img = im.readimg('./input_image_considered.png')
#output4, df4, status4, inner_circle, outter_circle = col_detect_main(img,'L', innerthreshold =  150, outterthreshold=40)
#im.display(output4)
#%%

#%%
#original_size = img.copy()
#original_size = im.resize_image(img,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
#im.display(original_size)
#output2, df2, status2 = col_detect_main(original_size)

#reduced_size= original_size.copy()
#reduced_size = im.resize_image(reduced_size, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

#sr = cv2.dnn_superres.DnnSuperResImpl_create()
#path = "ESPCN_x4.pb"
#sr.readModel(path)
#sr.setModel("espcn", 4)
#result = sr.upsample(reduced_size)
#result = im.resize_image(result, fx=2, fy = 2, interpolation=cv2.INTER_CUBIC)
#output2, df2, status2 = col_detect_main(original_size)
#%%




#increased_size = im.resize_image(reduced_size, fx=8, fy = 8, interpolation=cv2.INTER_NEAREST)
#im.display(increased_size)
#ref = im.readimg('./static/assets/img/masks/mask4.png')
#output, df, status = col_detect_main(increased_size)
#output1, df1, status1 = col_detect_main(img)


#%%
