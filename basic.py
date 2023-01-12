
# # Python code to read image
# import cv2
# from cv2 import destroyAllWindows
 
# # To read image from disk, we use
# # cv2.imread function, in below method,
# img = cv2.imread("elon.jpg", cv2.IMREAD_COLOR)
# # print(img)
# # Creating GUI window to display an image on screen
# # first Parameter is windows title (should be in string format)
# # Second Parameter is image array
# cv2.imshow("image", img)
# cv2.waitKey(5000)  #this function is used to decide the time of window
# cv2.destroyAllWindows()

# cv2.imwrite('elon_copy.png',img)

# python code for capturing videos on camera
from sys import implementation
from tkinter import font
import cv2
from cv2 import cvtColor
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import adaptiveThreshold

# cap = cv2.VideoCapture(0);

# while(True):
#     ret, frame = cap.read()
    
#     cv2.imshow('frame',frame)
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     if cv2.waitKey(1)  & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # capturing of saved videos

# import cv2
# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

# print(cap.isOpened())
# while(cap.isOpened()):
#     ret, img = cap.read()
#     if ret == True:
#         print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
#         out.write(img)

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('frame', gray)
    
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#          break
#     else:
#         break
# cap.release()
# out.release()
# cv2.destroyAllWindows()
    

# # drawing geometric shapes on picture 
# import cv2
# from cv2 import LINE_AA

# import numpy as np
# from pandas import Int8Dtype

# # for giving background color
# np.zeros((512,512,3),np.uint8)

# img = cv2.imread('elon.jpg',0)
# # for line 
# img = cv2.line(img,(0,0),(255,255),(255,0,0),5)
# # for rectangle
# img = cv2.rectangle(img,(0,0),(300,300),(0,0,255),3)
# # if we use -1 instead of thickness then it will color whole part

# font = cv2.FONT_HERSHEY_SIMPLEX
# img = cv2.putText(img,"opencv",(10,100),font,color=(255,0,0),thickness=2,lineType=LINE_AA)
# cv2.imshow('musk',img)

# cv2.waitKey()
# cv2.destroyAllWindows()

# # for putting date &time 
# import cv2
# import datetime
# cap = cv2.VideoCapture(0)
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# cap.set(3,3000)
# cap.set(4,3000)

# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
    #     fontc = cv2.FONT_HERSHEY_SIMPLEX
    #     text = "hieght:" + str(cap.get(3)) + "width" + str(cap.get(4))
    #     clock = str(datetime.datetime.now())
    #     frame= cv2.putText(frame,clock,(10,50),fontc,1,(255,0,0),2,cv2.LINE_AA)
    #     cv2.imshow("frame",frame)

    #     if cv2.waitKey(1) & 0xFFF == ord("q"):
    #         break
    # else:
#         break

# cap.release()
# cv2.destroyAllWindows()
       
# # handling mouse events

# import cv2
# from cv2 import LINE_AA
# import numpy as np


# def click_event(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x,',',y)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         text = str(x) + ',' +str(y)
#         img = cv2.putText(img,text,(x,y),font,1,(255,255,0),2)
#         cv2.imshow('img',img)
        
# img = np.zeros((512,512,3),np.uint8)
# cv2.imshow('img',img)
# cv2.setMouseCallback('img',click_event)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
# # split ,merge,resize,add etc images

# import cv2
# from cv2 import imread
# import numpy as np

# img = cv2.imread('elon.jpg')
# img2= cv2.imread('pass.jpg')

# print(img.shape)    #return the number of rows and column in tuple 
# print(img.size)     # return the number of pixels assecible in image
# print(img.dtype)    #return the image data type obtained
# b,g,r = cv2.split(img)
# img = cv2.merge((b,g,r))
# # now we will use add method
# dst = cv2.add(img,img2);      #for using this method size of the image should be same
# # for weighted image we use addwieghted method
# fst = cv2.addWeighted(img,0.9,img2,0.1,0)    #here 0.9 and 0.1 is weight and 0 is scalar

# cv2.imshow('img',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # bitwise operations

# import cv2
# from cv2 import imshow
# import numpy as np

# img1 = np.zeros((250,500,3),np.uint8)
# img1 = cv2.rectangle(img1,(200,0),(300,100),(255,255,255),-1)
# img2 = np.zeros((250,500,3),np.uint8)
# img2 = cv2.rectangle(img2,(250,0),(500,250),(255,255,255),-1)

# # for bitoperation function are as follows
# bitAnd = cv2.bitwise_and(img2,img1)
# bitOr = cv2.bitwise_or(img2,img1)
# bitNot = cv2.bitwise_not(img2,img1)
# bitXor = cv2.bitwise_xor(img2,img1)

# cv2.imshow('img1',img1)
# cv2.imshow('img2',img2)
# cv2.imshow('result',bitAnd)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # to bind the trackbar in opencv
# import cv2
# import numpy as np

# def nothing():
#     pass

# # create a window
# img = np.zeros((350,512,3),np.uint8)
# cv2.namedWindow('image')

# cv2.createTrackbar('R','image',0,255,nothing)
# cv2.createTrackbar('B','image',0,255,nothing)
# cv2.createTrackbar('G','image',0,255,nothing)

# switch = '0 : OFF\n 1 : ON'
# cv2.createTrackbar(switch ,'image',0,1,nothing)


# while(True):
#     cv2.imshow('image',img)
    
#     if cv2.waitKey(1) & 0xFF  == 27:
#         break
    
#     r = cv2.getTrackbarPos('R','image')
#     b = cv2.getTrackbarPos('B','image')
#     g = cv2.getTrackbarPos('G','image')
#     s = cv2.getTrackbarPos(switch,'image')
#     if switch == 0:
#         img[:] = 0
#     else:
#         img[:] = (r,b,g)
# cv2.destroyAllWindows()

# #note : here if we have to convert bgr to gray color then we have to use 
# #if switch == 0:
# #     pass
# #else:
# #     img = cv2.cvt.Color(img, str(pos),(50,150))

# #HSV(hue ,saturation ,value)
# #hue - differrent color
# #saturation - it is a amount of of color 
# #value = it is the amount of brightness


# import cv2
# import numpy as np

# def nothing ():
#     pass
# while True:
#     frame = cv2.imread('someimage.jpg')
    
#     hsv= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
#     lowerboundations = np.arange((100,50,29))
#     upperboundations  = np.arange((130,255,500)) #these are found by tracking 
    
#     mask  =  cv2.inRange(frame, lowerboundations,upperboundations)
#     res = cv2.bitwise_and(frame,frame,mask=mask)
    
#     cv2.imshow("frame",frame)
#     cv2.imshow("mask",mask)
#     cv2.imshow("res",res)

#     cv2.waitKey()
    
# # THRESHOLDING
# # thresholding means it is comparision of a color from the background or image.
# # it divides the image into 2 categories 1 is lower threshold intesity and other
# # is higher threshold intensity
# # code of adaptiveThreshold
# # adaptiveThreshold(src, dst, maxValue, adaptiveMethod, thresholdType, blockSize, C)
# import cv2
# from cv2 import ADAPTIVE_THRESH_MEAN_C
# import matplotlib
# import numpy as np
# img  = cv2.imread('black&whiteimage',0)
# _,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# cv2.imshow("image",img)
# cv2.imshow('thres',th1)

# cv2.waitKey(0)
# cv2.destroyAllWindows  


# # ADAPTIVE THRESHOLDING
# import cv2
# import numpy as np
# img  = cv2.imread('black&whiteimage',0)
# _,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# -,th2 = cv2.threshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

# cv2.imshow("image",img)
# cv2.imshow('thres',th1)

# cv2.waitKey(0)
# cv2.destroyAllWindows() 

# # opencv with matplotlib
# import cv2
# import matplotlib.pyplot as plt

# img = cv2.imread('elon.jpg',-1)
# cv2.imshow('image',img)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# plt.imshow(img)
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# MORPHOLOGICAL TRANSFORMATIONS
# It is the simple transformation of image shape
# there are 2 things required for this 1.origanal image 2.kernel 
# kernel tells how to change the value of sny given pixel by combinig with neighbour pixel
import cv2
# import matplotlib.pyplot as plt
from matplotlib import image, pyplot as plt
import numpy as np

img = cv2.imread('smarties.png',cv2.IMREAD_GRAYSCALE)
_,mask =  cv2.threshold(img,220,250,cv2.THRESH_BINARY_INV)
kernel = np.ones((2,2),np.uint8)
dilation  = cv2.dilate(mask,kernel,iterations=2)
erosion = cv2.erode(mask,kernel,iterations=1)
opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
titles = ['image','mask','dilation','erosion','opening']
images = [img,mask,dilation,erosion,opening]

for i in range(5):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
'''Basics of Erosion: 
    Erodes away the boundaries of the foreground object
    Used to diminish the features of an image.

Working of erosion: 
    A kernel(a matrix of odd size(3,5,7) is convolved with the image.
    A pixel in the original image (either 1 or 0) will be considered 1 only if all the pixels under the kernel are 1, otherwise, it is eroded (made to zero).
    Thus all the pixels near the boundary will be discarded depending upon the size of the kernel.
    So the thickness or size of the foreground object decreases or simply the white region decreases in the image.
'''
'''Basics of dilation: 

    Increases the object area
    Used to accentuate features   
A kernel(a matrix of odd size(3,5,7) is convolved with the image
    A pixel element in the original image is ‘1’ if at least one pixel under the kernel is ‘1’.
    It increases the white region in the image or the size of the foreground object increases '''