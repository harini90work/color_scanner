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
import datetime as dt
import matplotlib.pyplot as plt
#%%
#% Project Specific Functions
def process_circle(circles, offset1 = 0, offset2 = 0):
    final_circle = []
    for (x, y, r) in circles:
        valid = True
        grid_id = None
        if((x>560+offset1) & (x<660-offset2)):
            if((y>130+offset1) & (y<230-offset2)):
                grid_id = 'Grid 1;3'
        if((x>730+offset1) & (x<830-offset2)):
            if((y>130+offset1) & (y<230-offset2)):
                grid_id = 'Grid 1;4'
        if((x>890+offset1) & (x<990-offset2)):
            if((y>130+offset1) & (y<230-offset2)):
                grid_id = 'Grid 1;5'
        if((x>560+offset1) & (x<660-offset2)):
            if((y>290+offset1) & (y<390-offset2)):
                grid_id = 'Grid 2;3'
        if((x>730+offset1) & (x<830-offset2)):
            if((y>290+offset1) & (y<390-offset2)):
                grid_id = 'Grid 2;4'
        if((x>890+offset1) & (x<990-offset2)):
            if((y>290+offset1) & (y<390-offset2)):
                grid_id = 'Grid 2;5'
        if((x>250+offset1) & (x<350-offset2)):
            if((y>440+offset1) & (y<540-offset2)):
                grid_id = 'Grid 3;1'
        if((x>410+offset1) & (x<510-offset2)):
            if((y>440+offset1) & (y<540-offset2)):
                grid_id = 'Grid 3;2'
        if((x>560+offset1) & (x<660-offset2)):
            if((y>440+offset1) & (y<540-offset2)):
                grid_id = 'Grid 3;3'
                
        if((x>730+offset1) & (x<830-offset2)):
            if((y>440+offset1) & (y<540-offset2)):
                grid_id = 'Grid 3;4'
                
        if((x>890+offset1) & (x<990-offset2)):
            if((y>440+offset1) & (y<540-offset2)):
                grid_id = 'Grid 3;5'
                
        if((x>250+offset1) & (x<350-offset2)):
            if((y>610+offset1) & (y<710-offset2)):
                grid_id = 'Grid 4;1'
                
        if((x>410+offset1) & (x<510-offset2)):
            if((y>610+offset1) & (y<710-offset2)):
                grid_id = 'Grid 4;2'
                
        if((x>560+offset1) & (x<660-offset2)):
            if((y>610+offset1) & (y<710-offset2)):
                grid_id = 'Grid 4;3'
        if((x>730+offset1) & (x<830-offset2)):
            if((y>610+offset1) & (y<710-offset2)):
                grid_id = 'Grid 4;4'
                
        if((x>890+offset1) & (x<990-offset2)):
            if((y>610+offset1) & (y<710-offset2)):
                grid_id = 'Grid 4;5'
                
        if((x>250+offset1) & (x<350-offset2)):
            if((y>770+offset1) & (y<870-offset2)):
                grid_id = 'Grid 5;1'
                
        if((x>410+offset1) & (x<510-offset2)):
            if((y>770+offset1) & (y<870-offset2)):
                grid_id = 'Grid 5;2'
                
        if((x>560+offset1) & (x<660-offset2)):
            if((y>770+offset1) & (y<870-offset2)):
                grid_id = 'Grid 5;3'
                
        if((x>730+offset1) & (x<830-offset2)):
            if((y>770+offset1) & (y<870-offset2)):
                grid_id = 'Grid 5;4'
                
        if((x>890+offset1) & (x<990-offset2)):
            if((y>770+offset1) & (y<870-offset2)):
                grid_id = 'Grid 5;5'

                
        if(grid_id is not None):
            final_circle.append((x,y,r,grid_id))
    return final_circle
def extract_circle(img, r = 35):
    circle_list = [(585,200,r,'Grid 1;3'),
                   (730,200,r,'Grid 1;4'),
                   (870,200,r,'Grid 1;5'),
                   
                   (585,340,r,'Grid 2;3'),
                   (730,340,r,'Grid 2;4'),
                   (870,340,r,'Grid 2;5'),
                   
                   (300,480,r,'Grid 3;1'),
                   (440,480,r,'Grid 3;2'),
                   (585,480,r,'Grid 3;3'),
                   (730,480,r,'Grid 3;4'),
                   (870,480,r,'Grid 3;5'),
                   
                   (300,615,r,'Grid 4;1'),
                   (440,615,r,'Grid 4;2'),
                   (585,615,r,'Grid 4;3'),
                   (730,615,r,'Grid 4;4'),
                   (870,615,r,'Grid 4;5'),
                   
                   (300,755,r,'Grid 5;1'),
                   (440,755,r,'Grid 5;2'),
                   (585,755,r,'Grid 5;3'),
                   (730,755,r,'Grid 5;4'),
                   (870,755,r,'Grid 5;5'),
                   
                   ]
    return circle_list
    
def detect_color(main_img, original_img = None, factor = None):
    if(factor is None):
        factor = np.array([1,1,1])
    final_result = pd.DataFrame()
    if(original_img is None):
        original_img = main_img.copy()
        
    #img_final1 = main_img.copy()
    #img_final1 = im.resize_image(img_final1,fx=2,fy=2, interpolation = cv2.INTER_CUBIC) #cv2.resize(img_final1, (0,0), fx=2, fy=2,interpolation = cv2.INTER_CUBIC)
    op_img_final = im.resize_image(original_img,fx=2,fy=2, interpolation = cv2.INTER_CUBIC)
    #%
    circles = extract_circle(op_img_final)
    cv2.imwrite('./input_image_considered.png',op_img_final)
    #extract_circle(op_img_final)
    #gray = cv2.cvtColor(img_final1, cv2.COLOR_BGR2GRAY)
    
    
    # detect circles in the image
    
    # ensure at least some circles were found
    #hsv_low = [  0,   0, 255]
    #hsv_high = [0 , 0, 255]
    #mask, res, area, box = im.hsv_mask_area(img_final1, hsv_low, hsv_high)
    #count = 1
    
    output = op_img_final.copy()
    detected_circle = 0
    #for res in np.arange(1.3,2,0.1):
         
    #    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, res, 80)
        
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        #circles = np.round(circles[0, :]).astype("int")
        #circles = process_circle(circles,0,0)
        #print(len(circles))
        detected_circle = len(circles)
                        
        if(len(circles)==21):
            detected_circle = len(circles)
            old_output = output.copy()
            output = op_img_final.copy() 
            final_result = pd.DataFrame()
            count = 1
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r, grid_id) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                #print(x,y,r)
                if(r>40):
                    r = 40
                #im.display(output)
         #       cv2.rectangle(output, (x - r, y - r), (x + r, y + r), (0, 128, 255), -1)
                cropped = output[y-r+20:y+r-20,x-r+20:x+r-20]
                cropped = factor*cropped
                cropped[cropped>255] = 255
                cropped = cropped.astype('uint8')
                cropped = cv2.blur(cropped, (13,13))
                #im.display(cropped1)
                output[y-r+20:y+r-20,x-r+20:x+r-20] = cropped
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
                #print(hex_color)
                #im.display(cropped)
            #if(max(final_result['r']) > 42):
            #    output = old_output.copy()
            #else:
                #break
            #    pass
            # show the output image
            #im.display(output)
            #output = cv2.cvtColor(output,cv2.qCOLOR_RGB2BGR)
    if(len(circles)>0):
        for (x, y, r, grid_id) in circles:
            if(r>40):
                r = 40
            cv2.circle(output, (x, y), r, (0, 255, 0), 3)
                    
    output = cv2.putText(output, f'Total detected Circles:{detected_circle}',(10,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(64,127,255))
    return output, final_result


def colordetector(ip_img):
    #
    final_result = pd.DataFrame()
    original_img = ip_img.copy()
    #ip_img = im.readimg('20210714_092722.jpg')
    mask1 = im.readmask('./static/assets/img/masks/mask1.png')
    mask2 = im.readmask('./static/assets/img/masks/mask2.png')
    mask3 = im.readmask('./static/assets/img/masks/mask3.png')
    mask4 = cv2.bitwise_xor(mask2, mask3)
    
    #ip_img = im.resize_image(ip_img,mask1,interpolation = cv2.INTER_CUBIC)
    #im.display(mask3)
    main_img = cv2.bitwise_and(ip_img,ip_img,mask=mask1)
    outer_circle = cv2.bitwise_and(ip_img,ip_img,mask=mask2)
    inner_circle = cv2.bitwise_and(ip_img,ip_img,mask=mask3)
    #Check Inner Circle Median Color
    contours, hierarchy = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print(contours)
    if(len(contours)>0):
        rect = cv2.boundingRect(contours[0])
        test = inner_circle[rect[1]+10:rect[1]+rect[3]-10,rect[0]+10:rect[0]+rect[2]-10]
    
    
        #im.display(test)
        test1 = np.mean(test,axis=1).mean()
        print('Inner Circle Median color', test1)
    else:
        test1 = 0
    mark = 0
    mark = int(((test1/165)/2)*100)
    if(mark>50):
        mark = 50
    print(test1)
    if(test1>165):
        test1 = True
        factor = np.median(np.divide(255,test).mean(axis=1),axis=0)
        print('Before filter factor', factor)
        factor[(factor>1.8) & (factor<3)] = 1.8
        factor[factor>3] = 1
        print('After filter factor', factor)
    else:
        test1 = False
        
    #Check Outer Circle Median Color
    #test1 = True
    if(test1):
        contours, hierarchy = cv2.findContours(mask4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rect = cv2.boundingRect(contours[0])
        test = outer_circle[rect[1]+10:rect[1]+rect[3]-10,rect[0]+10:rect[0]+rect[2]-10]
        overlay = mask4[rect[1]+10:rect[1]+rect[3]-10,rect[0]+10:rect[0]+rect[2]-10]
        test = cv2.bitwise_and(test,test,mask=overlay)
        #im.display(test)
        testhsv = cv2.cvtColor(test,cv2.COLOR_BGR2HSV)
        test2 = np.mean(testhsv[:,:,2],axis=1).mean()
        print('Outer Circle Median color', test2)
        if(test2>0):
            temp = 30/test2
            temp = int((temp/2) *100)
            if(temp>50):
                temp = 50
            mark = mark + temp
        if(test2<20):
            test2 = True
        else:
            test2 = False
    else:
        test2 = False
    op_img = ip_img.copy()
    #print(test1, test2)
    
        
    
    if(test2):
        print('Final Factor used', factor)
        op_img, final_result = detect_color(main_img, ip_img, factor)
        if(final_result.shape[0]>0):
            final_result = final_result.sort_values('Grid ID')
        #print(final_result)
        
    else:
        cv2.putText(op_img,'Keep Image within frame',(10,20),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
        cv2.putText(op_img,f'Camera Position Accuracy: {mark}',(10,60),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0))
        
        #im.display(op_img)
    #im.display(main_img)
    #im.display(outer_circle)
    #im.display(inner_circle)
    op_img = im.resize_image(op_img, original_img, interpolation = cv2.INTER_CUBIC)
    return op_img, final_result


#% Test bench
#ip_file = './input_image_considered'
#ip_img = im.readimg(f'./{ip_file}.png')
#ip_img = im.resize_image(ip_img,fx = 0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
#op_img, final_result = colordetector(ip_img)
#im.display(op_img)
#ip_file = 'ip_img-r3'
#ip_img = im.readimg(f'./{ip_file}.png')
#print(dt.datetime.now())
#test = colordetector(ip_img)
#im.display(test[0])
#cv2.imwrite(f'./{ip_file}-op.png', test[0])
#test[1].to_csv(f'./{ip_file}-op.csv')

#test1 = cv2.cvtColor(test,cv2.COLOR_BGR2HSV)
#plt.boxplot(np.mean(test[:,:,:],axis=1))
#%
#plt.boxplot(np.mean(test1[:,:,2],axis=1))

#%%
def col_detect_main(ip_img):
    #video = cv2.VideoCapture(0)
    #%
    status = False
    mask4 = im.readimg('./static/assets/img/masks/mask4.png')
    mask5 = im.readimg('./static/assets/img/masks/mask5.png')
    ip_img = im.resize_image(ip_img,mask4,interpolation = cv2.INTER_CUBIC)
    #print(mask4)
    mask6 = cv2.add(mask4, mask5)
    mask6 = cv2.cvtColor(mask6, cv2.COLOR_BGR2GRAY)
    overlay = im.blankimg(mask6.shape[0], mask6.shape[1])
    overlay = cv2.bitwise_and(overlay, overlay, mask = mask6)
    frame = ip_img.copy()
    #while True:
    #    status, frame = video.read()
        
        #print(status)
    #    if(status):
    test = cv2.bitwise_xor(overlay,frame)
    output, df = colordetector(frame)
    output = cv2.bitwise_xor(overlay,output)
    if(df.shape[0] == 21):
        status = True
            #print(df.shape)
    #        cv2.imshow('color detector', test)
    #        cv2.imshow('output', output)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break
        
    #cv2.destroyAllWindows()
    #im.display(output)
    return output, df, status
    
print('All Functions Loaded successfully')
