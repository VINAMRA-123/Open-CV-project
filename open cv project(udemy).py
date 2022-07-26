import cv2, glob

from cv2 import CascadeClassifier
from cv2 import imread
all_images = glob.glob("*.jpg")
detect = CascadeClassifier("haarcascade_facefrontal")
for images in all_images():
   img = imread(images)
   gray_image =cv2.cvtColour(img,cv2.COLOR_BGR2GRAY)
   faces = detect.detectMultiScalar(gray_image, 1.1, 5)
   
for (x,y,w,z) in faces:
    final_images = cv2.rectangle(img, (x,y),(x+w,y+z), (0,250,0)
    ,5)
    
    cv2.imshow("smart",final_images)
    cv2.waitKey(0)
    cv2.destroyALLWindows
    
