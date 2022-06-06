# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:08:44 2021

@author: Andy
"""
import cv2
import mediapipe as mp
import time

thres = 0.5

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=False,
                      max_num_hands=100,
                      min_detection_confidence=0.75,
                      min_tracking_confidence=0.5)
mpdraw = mp.solutions.drawing_utils

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    classIds, confs, bbox = net.detect(img, confThreshold = thres)
    #print(classIds, bbox)
    save_picture = 1
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classNames[classId-1] == "cell phone" and results.multi_hand_landmarks :
                for handlms in results.multi_hand_landmarks:
                    #print(classNames[classId-1])
                    #mpdraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)
                    cv2.rectangle(img, box, color = (0, 0, 255), thickness = 2)
                    cv2.putText(img, "CAUTION!!!!", (box[0], box[1]-20,), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255),3)
                    cv2.imwrite("cheating/"+str(save_picture)+".jpg", img)
                    #cv2.putText(img, str(confidence), (box[0]+50, box[1]+30,), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
                    save_picture+=1
                
    cv2.imshow('img', img)
    kk = cv2.waitKey(1)
    if kk == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()


