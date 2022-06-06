# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 17:18:57 2021

@author: user
"""

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.2)

mp_face_detection = mp.solutions.face_detection

mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

thres = 0.5
mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=False,
                      max_num_hands=100,
                      min_detection_confidence=0.75,
                      min_tracking_confidence=0.5)


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

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
head = ['左轉頭', '右轉頭', '向前看', '低頭']
x_faces = []
y_faces = []

def cv2ImgAddText(image, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(image, np.ndarray)):  #判斷是否OpenCV圖片類型
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((boundbox[0]+15, boundbox[1]-60), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)






while cap.isOpened():
        
    success, image = cap.read()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    face_mesh_results = face_mesh.process(image)
    face_detection_results = face_detection.process(image)
    hands_results = hands.process(image)
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    
    classIds, confs, bbox = net.detect(image, confThreshold = thres)
    
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []    
    show_faces = []
    fac = 0
    face1 = []
    face2 = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classNames[classId-1] == "cell phone" and hands_results.multi_hand_landmarks :
                for handlms in hands_results.multi_hand_landmarks:
                    #print(classNames[classId-1])
                    #mpdraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)
                    cv2.rectangle(image, box, color = (0, 0, 255), thickness = 2)
                    cv2.putText(image, "CAUTION!!!!", (box[0], box[1]-20,), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255),3)
                    #cv2.imwrite("cheating/"+str(save_picture)+".jpg", img)
                    #cv2.putText(img, str(confidence), (box[0]+50, box[1]+30,), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
                    #save_picture+=1
    
    if face_mesh_results.multi_face_landmarks or face_detection_results.detections:
            for id, detection in enumerate(face_detection_results.detections):
                facebbox = detection.location_data.relative_bounding_box
                h,w,c = image.shape
                boundbox = int(facebbox.xmin * w),int(facebbox.ymin * h),int(facebbox.width * w),int(facebbox.height * h)
                cv2.rectangle(image,boundbox,(255, 255, 255), 2)
                fac = fac+1
                cv2.putText(image, str(fac), (boundbox[0]+60, boundbox[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                #print(len(face_landmarks))
                
                for idx, lm in enumerate(face_landmarks.landmark):
                    #print(lm)
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                                #if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * img_c)
                                    
                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                #cv2.putText(image, str(test), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                # Get the 2D Coordinates
                                face_2d.append([x, y])
                                
                                # Get the 3D Coordinates
                                face_3d.append([x, y, lm.z])       
                        
                        # Convert it to the NumPy array
                    np_face_2d = np.array(face_2d, dtype=np.float64)
                        
                        # Convert it to the NumPy array
                    np_face_3d = np.array(face_3d, dtype=np.float64)
            
                        # The camera matrix
                    focal_length = 1 * img_w
            
                    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                                [0, focal_length, img_w / 2],
                                                [0, 0, 1]])
            
                        # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
            
                        # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(np_face_3d, np_face_2d, cam_matrix, dist_matrix)
                #print(trans_vec)
                        # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)
                #p = cv2.Rodrigues(success)
                        # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                
                
                        # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                #print(y)
                
                    #x_faces.append([i, x])
                    #y_faces.append([i, y])
                    #print(x_faces[i][1])
                    
                        # See where the user's head tilting
                        
            
                        # Display the nose direction
                    #nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            
                    #p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    #p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                    #cv2.line(image, p1, p2, (255, 0, 0), 2)
                    #if i == 0:
                #for id, detection in enumerate(results2.detections):
                if fac >= 1:
                   if y < -5 :
                       image = cv2ImgAddText(image, "左轉頭", 140, 60, (255, 255, 0), 60)
                   elif y > 5 :
                       image = cv2ImgAddText(image, "右轉頭", 140, 60, (255, 255, 0), 60)
                   elif x < -5 :
                       image = cv2ImgAddText(image, "低頭", 140, 60, (255, 255, 0), 60)
                   elif x > 5  :
                       image = cv2ImgAddText(image, "抬頭", 140, 60, (255, 255, 0), 60)
                            #elif y > -10 and y < 10 and x > -10 and x < 10:
                             #   image = cv2ImgAddText(image, "向前看", 140, 60, (255, 255, 0), 60)
                    
                        
                
    '''if results2.detections:
   
        for id, detection in enumerate(results2.detections):
            #mp_drawing.draw_detection(image, detection)
            bbox = detection.location_data.relative_bounding_box
            h,w,c = image.shape
            boundbox = int(bbox.xmin * w),int(bbox.ymin * h),int(bbox.width * w),int(bbox.height * h)
            for detection in results2.detections:
                mp_drawing.draw_detection(image, detection)
                                
                    #cv2.line(image, p1, p2, (255, 0, 0), 2)
                if y < -10:
                    image = cv2ImgAddText(image, "左轉頭", 140, 60, (255, 255, 0), 60)
                elif y > 10:
                    image = cv2ImgAddText(image, "右轉頭", 140, 60, (255, 255, 0), 60)
                elif x < -10:
                    image = cv2ImgAddText(image, "低頭", 140, 60, (255, 255, 0), 60)
                elif x > 10:
                    image = cv2ImgAddText(image, "抬頭", 140, 60, (255, 255, 0), 60)
                else:
                    image = cv2ImgAddText(image, "向前看", 140, 60, (255, 255, 0), 60)'''
                    # Add the text on the image
    #cv2.putText(image, "test", (boundbox[0], boundbox[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        cv2.destroyAllWindows()
        break

cap.release()