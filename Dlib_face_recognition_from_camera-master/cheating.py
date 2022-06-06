import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from timeit import default_timer as timer
import time
import os
import requests

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

now = time.localtime()
dtString = time.strftime('%Y/%m/%d %H:%M:%S',now)

start_time =0.0

cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)

#start_time = 0.00

def cv2ImgAddText(image, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(image, np.ndarray)):  #判斷是否OpenCV圖片類型
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    draw.text((boundbox[0]+10, boundbox[1]-60), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def show_mes(image, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(image, np.ndarray)):  #判斷是否OpenCV圖片類型
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    draw.text((20, 100), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)




def main():
    img_count = 1
    show_time = 3.5
    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image.flags.writeable = False
        
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
        # To improve performance
        
        hands_results = hands.process(image)
        classIds, confs, hands_bbox = net.detect(image, confThreshold = thres)
        # Get the result
        face_mesh_results = face_mesh.process(image)
        face_detection_results = face_detection.process(image)
        # To improve performance
        image.flags.writeable = True
        
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []    
        global boundbox
        
        img_count = str(img_count)
            
        imagePath = "./cheating_pic/"
        imageName = "cheating_img"+img_count+".jpg"
        imageFileName = imagePath + imageName
            
        img_count = int(img_count)
        
        
        if len(classIds) != 0:
            
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), hands_bbox):
                if classNames[classId-1] == "cell phone" and hands_results.multi_hand_landmarks :
                    for handlms in hands_results.multi_hand_landmarks:
                        #print(classNames[classId-1])
                        #mpdraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)
                        cv2.rectangle(image, box, color = (0, 0, 255), thickness = 2)
                        cv2.putText(image, "CAUTION!!!!", (box[0], box[1]-20,), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255),3)
                        cv2.putText(image, 'Time:'+str(dtString), (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        #cv2.imwrite("cheating/"+str(save_picture)+".jpg", image)
                        cv2.imwrite(imageFileName, image)
                        recordTime = [time.strftime("%Y%m%d%H%M%S", time.localtime())]
                        raw_data = {'Action': 'Upload', 'ImageName':imageFileName, 'RecordTime':recordTime}
                        #df = pd.DataFrame(raw_data)
                        #df.to_csv('data.csv',mode='a',index=False,header=False)
                        
                        url = 'http://dgdhdrh.000webhostapp.com'
                        files = {'fileToUpload': open(imagePath+imageName, 'rb')}
                        r = requests.post(url, files=files, data=raw_data)
                        print(r.text)
                        img_count+=1
                        
        
        if face_detection_results.detections:
            for detection in face_detection_results.detections:
                mp_drawing.draw_detection(image, detection)
                bbox = detection.location_data.relative_bounding_box
                h,w,c = image.shape
                boundbox = int(bbox.xmin * w),int(bbox.ymin * h),int(bbox.width * w),int(bbox.height * h)
                #return boundbox
        
        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            #print(lm)
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * img_c)
                            #print(type(nose_2d))
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                            
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
        
                    # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)
        
                    # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
                    # Get the y rotation degree
                face_x = angles[0] * 360
                face_y = angles[1] * 360
            
            '''img_count = str(img_count)
            
            imagePath = "./cheating_pic/"
            imageName = "cheating_img"+img_count+".jpg"
            imageFileName = imagePath + imageName
            
            img_count = int(img_count)'''
                #print(y)
            if face_y > -7 and face_y < 7 and face_x > -7 and face_x < 7:
                global start_time
                start_time = timer()
                
                #image = cv2ImgAddText(image, "向前看", 140, 60, (255, 255, 0), 60)
            # See where the user's head tilting
            if face_y < -7:
                current_time = timer()
                cout_time = current_time - start_time
                cout_time = round(cout_time, 2)
                cv2.putText(image, 'Past Time:'+str(cout_time), (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, 'Time:'+str(dtString), (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                image = cv2ImgAddText(image, "左轉頭", 140, 60, (255, 255, 0), 60)
                image = show_mes(image, "疑似作弊", 140, 60, (255, 0, 0), 60)
                if cout_time >= show_time and cout_time<=4.5:
                    # 讀取圖片
                    cv2.imwrite(imageFileName, image)
                    recordTime = [time.strftime("%Y%m%d%H%M%S", time.localtime())]
                    raw_data = {'Action': 'Upload', 'ImageName':imageFileName, 'RecordTime':recordTime}
                    #df = pd.DataFrame(raw_data)
                    #df.to_csv('data.csv',mode='a',index=False,header=False)
                    
                    url = 'http://dgdhdrh.000webhostapp.com'
                    files = {'fileToUpload': open(imagePath+imageName, 'rb')}
                    r = requests.post(url, files=files, data=raw_data)
                    print(r.text)
                    img_count+=1
                
            elif face_y > 7:
                current_time = timer()
                cout_time = current_time - start_time
                cout_time = round(cout_time, 2)
                cv2.putText(image, 'Past Time:'+str(cout_time), (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, 'Time:'+str(dtString), (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                image = cv2ImgAddText(image, "右轉頭", 140, 60, (255, 255, 0), 60)
                image = show_mes(image, "疑似作弊", 140, 60, (255, 0, 0), 60)
                if cout_time >= show_time and cout_time<=4.5:
                    # 讀取圖片
                    cv2.imwrite(imageFileName, image)
                    recordTime = [time.strftime("%Y%m%d%H%M%S", time.localtime())]
                    raw_data = {'Action': 'Upload', 'ImageName':imageFileName, 'RecordTime':recordTime}
                    #df = pd.DataFrame(raw_data)
                    #df.to_csv('data.csv',mode='a',index=False,header=False)
                    
                    url = 'http://dgdhdrh.000webhostapp.com'
                    files = {'fileToUpload': open(imagePath+imageName, 'rb')}
                    r = requests.post(url, files=files, data=raw_data)
                    print(r.text)
                    img_count+=1
            elif face_x < -7:
                image = cv2ImgAddText(image, "低頭", 140, 60, (255, 255, 0), 60)
                image = show_mes(image, "疑似作弊", 140, 60, (255, 0, 0), 60)
                
            elif face_x > 7:
                current_time = timer()
                cout_time = current_time - start_time
                cout_time = round(cout_time, 2)
                cv2.putText(image, 'Past Time:'+str(cout_time), (250, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, 'Time:'+str(dtString), (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                image = cv2ImgAddText(image, "抬頭", 140, 60, (255, 255, 0), 60)
                image = show_mes(image, "疑似作弊", 140, 60, (255, 0, 0), 60)
                if cout_time >= show_time and cout_time<=4.5:
                    # 讀取圖片
                    cv2.imwrite(imageFileName, image)
                    recordTime = [time.strftime("%Y%m%d%H%M%S", time.localtime())]
                    raw_data = {'Action': 'Upload', 'ImageName':imageFileName, 'RecordTime':recordTime}
                    #df = pd.DataFrame(raw_data)
                    #df.to_csv('data.csv',mode='a',index=False,header=False)
                    
                    url = 'http://dgdhdrh.000webhostapp.com'
                    files = {'fileToUpload': open(imagePath+imageName, 'rb')}
                    r = requests.post(url, files=files, data=raw_data)
                    print(r.text)
                    img_count+=1
            
            cv2.putText(image, "Q: Quit", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #else:
             #   image = cv2ImgAddText(image, "向前看", 140, 60, (255, 255, 0), 60)
            #cv2.rectangle(image, int(lm.x * img_w), int(lm.y * img_h), (0, 255, 0), 2)
                    # Display the nose direction
                    #nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        
                    #p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    #p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                    
                    #cv2.line(image, p1, p2, (255, 0, 0), 2)
        
                    # Add the text on the image
                    #cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)'''
    
    
        
        cv2.imshow('Head Pose Estimation', image)
    
        if cv2.waitKey(5) == ord('q'):
            cv2.destroyAllWindows()
            break
    
    cap.release()


if __name__ == '__main__':
    main()