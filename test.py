import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow 


cap= cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("/final/Model/keras_model.h5","/final/Model/labels.txt")


offset=20 
imgSize=300

#folder = "Data/Aa"
#counter =0
labels=["Aa","Bb","Cc","Dd","hello","ilu","yes","yo"]
while True:
    success, img = cap.read() 
    imgOutput = img.copy()
    hands, img=detector.findHands(img)

    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']

        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop=img[y-offset: y+h+offset , x-offset: x+w+offset]
        imgCropShape=imgCrop.shape
      

        aspectRatio=h/w
        if aspectRatio>1:
            k = imgSize/h #constant = size of image / height
            wCal= math.ceil(k*w) #width calculated = constant*previous width math.ceil round of the value
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))# imgcrop(width,height)
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap]=imgResize
            prediction, index =classifier.getPrediction(imgWhite)
            print(prediction,index)
            
        else:    
            k = imgSize/w #constant = size of image / width
            hCal= math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:]=imgResize
            prediction, index =classifier.getPrediction(imgWhite, draw=False)
        

        cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+90,y-offset-50+50),(255,0,255),cv2.FILLED)

        cv2.putText(imgOutput, labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)


        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)

    cv2.imshow("Image",imgOutput)
    key= cv2.waitKey(1)
    if key == ord("E") or key == ord("e"):
        print("Exiting the program...")
        break
#cap.release()
#cv2.destroyAllWindows()


