import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap= cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset=20 
imgSize=300

folder = "Data\yo"
counter =0
while True:                         
    sucess, img = cap.read()
    hands, img=detector.findHands(img) 
    cv2.putText(img, "Press Q or q to capture dataset", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 0,0), 2, cv2.LINE_AA)
    cv2.putText(img, "Press E or e to exit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 0, 0), 2, cv2.LINE_AA)
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
        else:    
            k = imgSize/w #constant = size of image / width
            hCal= math.ceil(k*h) #width calculated = constant*previous width math.ceil round of the value
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))# imgcrop(width,height)
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:]=imgResize

        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)

    cv2.imshow("Image",img)
    key= cv2.waitKey(1)
    if key == ord("Q")or key == ord("q"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
    elif key == ord("E") or key == ord("e"):
        print("Exiting the program...")
        break
