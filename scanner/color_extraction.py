# -*- coding: utf-8 -*-
"""
    Last Modified on Mon Jul 12 11:31:21 2021
    
        
    Description:
    -----------
        *
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


import pandas as pd
import numpy as np
try:
    from . import Img_Proc as im
except:
    import Img_Proc as im
import cv2
#%% Resize image
mask_name = 'mask1'
file = f'./static/assets/img/masks/{mask_name}.png'
ofile = f'./static/assets/img/masks/{mask_name}-p.png'
Ref_potrat = './static/assets/img/masks/Ref_potrat.png'
file1 = im.readimg(file)
ref = im.readimg(Ref_potrat)
file2 = im.resize_image(file1, ref)
cv2.imwrite(ofile, file2)
#shift the image
#file = './static/assets/img/masks/mask1.png'
#file1 = im.readimg(file)
#file1[:,0:640-63] =  file1[:,63:]
#cv2.imwrite(file, file1)
#%%

#% Project Specific Functions

def extract_circle(img, r = 60, xoffset = -63):
    circle_list = [(585+xoffset,200,r,'Grid 1;3'),
                   (730+xoffset,200,r,'Grid 1;4'),
                   (870+xoffset,200,r,'Grid 1;5'),
                   
                   (585+xoffset,340,r,'Grid 2;3'),
                   (730+xoffset,340,r,'Grid 2;4'),
                   (870+xoffset,340,r,'Grid 2;5'),
                   
                   (300+xoffset,480,r,'Grid 3;1'),
                   (440+xoffset,480,r,'Grid 3;2'),
                   (585+xoffset,480,r,'Grid 3;3'),
                   (730+xoffset,480,r,'Grid 3;4'),
                   (870+xoffset,480,r,'Grid 3;5'),
                   
                   (300+xoffset,615,r,'Grid 4;1'),
                   (440+xoffset,615,r,'Grid 4;2'),
                   (585+xoffset,615,r,'Grid 4;3'),
                   (730+xoffset,615,r,'Grid 4;4'),
                   (870+xoffset,615,r,'Grid 4;5'),
                   
                   (300+xoffset,755,r,'Grid 5;1'),
                   (440+xoffset,755,r,'Grid 5;2'),
                   (585+xoffset,755,r,'Grid 5;3'),
                   (730+xoffset,755,r,'Grid 5;4'),
                   (870+xoffset,755,r,'Grid 5;5'),
                   
                   ]
    return circle_list


def extract_small_circle(cropped):
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

    
def detect_color(main_img, original_img = None, factor = None):
    if(factor is None):
        factor = np.array([1,1,1])
    final_result = pd.DataFrame()
    if(original_img is None):
        original_img = main_img.copy()
        
    op_img_final = im.resize_image(original_img,fx=2,fy=2, interpolation = cv2.INTER_CUBIC)
    #%
    circles = extract_circle(op_img_final)
    #cv2.imwrite('./input_image_considered.png',op_img_final)
    
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
                hex_color = im.rgb_to_hex(rgb)
                row = {
                'Circle No' : count,
                'Grid ID': grid_id,
                #'x' : x,
                #'y' : y,
                #'r' : r,
                'Hex Value': hex_color,
                
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


def colordetector(ip_img):
    #
    original_img = ip_img.copy()
    final_result = pd.DataFrame()
    mask1 = im.readmask('./static/assets/img/masks/mask1.png')
    mask2 = im.readmask('./static/assets/img/masks/mask2.png')
    mask3 = im.readmask('./static/assets/img/masks/mask3.png')
    mask4 = cv2.bitwise_xor(mask2, mask3)
    
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
    if(test1>=160):
        test1 = True
        factor = np.median(np.divide(255,test).mean(axis=1),axis=0)
        print('Before filter factor', factor)
        factor[(factor>1.8) & (factor<3)] = 1.8
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

        if(test2<=37):
            test2 = True
        else:
            test2 = False
    else:
        test2 = False
    op_img = ip_img.copy()
    
    if(test2):
        print('Final Factor used', factor)
        op_img, final_result = detect_color(main_img, ip_img, factor)
        if(final_result.shape[0]>0):
            final_result = final_result.sort_values('Grid ID')
    else:
        cv2.putText(op_img,'Keep Image within frame',(10,20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
        cv2.putText(op_img,f'Inner Circle Intensity: {innercircle_intensity}',(10,60),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
        cv2.putText(op_img,f'Outter Circle Intensity: {outercircle_intensity}',(10,90),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
        
    op_img = im.resize_image(op_img, original_img, interpolation = cv2.INTER_CUBIC)
    #op_img = im.resize_image(op_img, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
    return op_img, final_result



#%
def col_detect_main(ip_img):
    status = False
    mask4 = im.readimg('./static/assets/img/masks/mask4.png')
    mask5 = im.readimg('./static/assets/img/masks/mask5.png')
    ip_img = im.resize_image(ip_img,mask4,interpolation = cv2.INTER_CUBIC)
    mask6 = cv2.add(mask4, mask5)
    mask6 = cv2.cvtColor(mask6, cv2.COLOR_BGR2GRAY)
    overlay = im.blankimg(mask6.shape[0], mask6.shape[1])
    overlay = cv2.bitwise_and(overlay, overlay, mask = mask6)
    frame = ip_img.copy()
    output, df = colordetector(frame)

    output = cv2.bitwise_xor(overlay,output)
    if(df.shape[0] == 21):
        status = True
    return output, df, status
    
print('All Functions Loaded successfully')

#img = im.readimg('./input_image_considered.png')
#output4, df4, status4 = col_detect_main(img)

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
