import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
from PIL import Image, ImageDraw, ImageFont
import serial
import pymysql
import csv
# Dlib 正向人臉檢測器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

face_path = "data/data_dlib/shape_predictor_68_face_landmarks.dat"
# Dlib 人臉 landmark 特徵點檢測器 / Get face landmarks
predictor = dlib.shape_predictor(face_path)

# Dlib Resnet 人臉辨識模型, 抓取 128D 的特徵矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

#開啟Arduino連接埠
ser = serial.Serial('COM4',9600)
if not ser.isOpen():
    ser.open()
print('COM4 is open :', ser.isOpen())


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        self.font_chinese = ImageFont.truetype("simsun.ttc", 30)
        # For FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        #當前人臉數
        self.i = 0
        self.faces = []
        self.img_rd2 = []
        # cnt for frame
        self.frame_cnt = 0
        #畫面中的人臉
        self.k = 0
        # 用來存放所有存入人臉特稱的數組 / Save the features of faces in the database
        self.face_features_known_list = []
        self.realtemp = []
        # 儲存人臉名字 / Save the name of faces in the database
        self.face_name_known_list = []

        # 用來儲存上一幀和當前幀 ROI 的質心座標 / List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # 用來儲存上一幀和当前幀檢測出目標的名字 / List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        # 上一幀和當前幀中人臉數的計數器 / cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # 用來存放進行識別時候對比的歐式距離 / Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # 儲存當前webcam中捕捉到的所有人臉的坐標名字 / Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        # 儲存當前webcam中捕捉到的人臉特徵 / Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        # 如果辨識出 "unknown" 的臉, 將在 reclassify_interval_cnt 計數到 reclassify_interval 後, 對人臉進行重新辨識
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10
        self.similar_person_num = 0

    # 从 "features_all.csv" 讀取存入人臉特徵 / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                for j in range(0, 128):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
                self.face_name_known_list.append("Person_" + str(i + 1))
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    # 獲取處理之後 stream 的幀数 / Get the fps of video stream
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # 計算兩個128D向量間的歐式距離 / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # 利用質心追蹤來識別人臉 / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            # 对于当前帧中的人脸1, 和上一帧中的 人脸1/2/3/4/.. 进行欧氏距离计算 / For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    # 生成的 cv2 window 上面添加說明文字 / putText on cv2 window
    def draw_note(self, img_rd):
        # 添加说明 / Add some info on windows
        cv2.putText(img_rd, "Face Recognizer with OT", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font,
                                 0.8, (255, 190, 0),
                                 1,
                                 cv2.LINE_AA)
    
    
    def draw_name(self):
        # 在人臉框下寫人名 / Write names under ROI
        logging.debug(self.current_frame_face_name_list)
        img = Image.fromarray(cv2.cvtColor(self.img_rd2, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        draw.text(xy=self.current_frame_face_position_list[0], text=self.current_frame_face_name_list[0], font=self.font_chinese,
                  fill=(255, 0, 0))
        self.img_rd2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return self.img_rd2

    def show_chinese_name(self):
        if self.current_frame_face_cnt >= 1:
            logging.debug(self.face_name_known_list)
            # 修改存入的人臉姓名 / Modify names in face_name_known_list to chinese name
            self.face_name_known_list[0] ='黃湟鈞'.encode('utf-8').decode()
            self.face_name_known_list[1] ='潘進昇'.encode('utf-8').decode()
            self.face_name_known_list[2] ='邱智清'.encode('utf-8').decode()
            self.face_name_known_list[3] ='林郁勝'.encode('utf-8').decode()
            self.face_name_known_list[4] ='林奕褕'.encode('utf-8').decode()
            self.face_name_known_list[5] ='陳家瀅'.encode('utf-8').decode()
            self.face_name_known_list[6] ='周映澤'.encode('utf-8').decode()
            self.face_name_known_list[7] ='林洛平'.encode('utf-8').decode()
            self.face_name_known_list[8] ='李柏賢'.encode('utf-8').decode()
            self.face_name_known_list[9] ='廖威瑀'.encode('utf-8').decode()
            self.face_name_known_list[10] ='林宜貞'.encode('utf-8').decode()
            self.face_name_known_list[11] ='張國御'.encode('utf-8').decode()
            self.face_name_known_list[12] ='戴嘉霈'.encode('utf-8').decode()
            self.face_name_known_list[13] ='李咨穎'.encode('utf-8').decode()
            self.face_name_known_list[14] ='黃尹蓁'.encode('utf-8').decode()
            self.face_name_known_list[15] ='林育賢'.encode('utf-8').decode()
    #讀取溫度
    def get_temp(self):
        temp=ser.readline().decode('utf-8')
        self.realtemp = temp[9:14] + chr(176)+'C'
        return self.realtemp
        
    def test_def1(self):
        self.current_frame_face_position_list = []
        if "不認識" in self.current_frame_face_name_list:
            logging.debug("  有未知人臉, 開始進行 reclassify_interval_cnt 計數")
            self.reclassify_interval_cnt += 1

        if self.current_frame_face_cnt != 0:
            for k, d in enumerate(self.faces):
                self.current_frame_face_position_list.append(tuple(
                    [self.faces[k].left(), int(self.faces[k].bottom() + (self.faces[k].bottom() - self.faces[k].top()) / 4)]))
                self.current_frame_face_centroid_list.append(
                                [int(self.faces[k].left() + self.faces[k].right()) / 2,
                                 int(self.faces[k].top() + self.faces[k].bottom()) / 2])

                self.img_rd2 = cv2.rectangle(self.img_rd2,tuple([d.left(), d.top()]),tuple([d.right(), d.bottom()]),(255, 255, 255), 2)
                #self.img_rd2 = self.draw_name(self.img_rd2)
                
                    # 如果当前帧中有多个人脸, 使用质心追踪 / Multi-faces in current frame, use centroid-tracker to track
        if self.current_frame_face_cnt != 1:
            self.centroid_tracker()

        for i in range(self.current_frame_face_cnt):
            # 6.2 Write names under ROI
            #self.img_rd2 = cv2.putText(self.img_rd2, self.current_frame_face_name_list[i],self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,cv2.LINE_AA)
            
            temp23 = self.get_temp()
            #self.show_chinese_name()
            #self.draw_name(self.img_rd2)
            self.img_rd2 = self.draw_name()
            cv2.putText(self.img_rd2,' Temp='+ str(temp23) ,
                                      self.current_frame_face_position_list[i], self.font, 1.5, (0, 255, 0), 3,
                                       cv2.LINE_AA)
            
        
    def test_def2(self):
        self.current_frame_face_name_list = []
        for i in range(len(self.faces)):
            shape = predictor(self.img_rd2, self.faces[i])
            self.current_frame_face_feature_list.append(face_reco_model.compute_face_descriptor(self.img_rd2, shape))
            self.current_frame_face_name_list.append("不認識")
            # 6.2.2.1 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
        for self.k in range(len(self.faces)):
            logging.debug("  For face %d in current frame:", self.k + 1)
            self.current_frame_face_centroid_list.append([int(self.faces[self.k].left() + self.faces[self.k].right()) / 2,int(self.faces[self.k].top() + self.faces[self.k].bottom()) / 2])
            self.current_frame_face_X_e_distance_list = []
            # 6.2.2.2 每个捕获人脸的名字坐标 / Positions of faces captured
            self.current_frame_face_position_list.append(tuple([self.faces[self.k].left(), int(self.faces[self.k].bottom() + (self.faces[self.k].bottom() - self.faces[self.k].top()) / 4)]))
                            # 6.2.2.3 对于某张人脸, 遍历所有存储的人脸特征
                            # For every faces detected, compare the faces in the database
            
            for i in range(len(self.face_features_known_list)):
                # 如果 q 数据不为空
                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[self.k],
                                        self.face_features_known_list[i])
                                    logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                else:
                    # 空数据 person_X
                    self.current_frame_face_X_e_distance_list.append(999999999)
                    # 6.2.2.4 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
            self.similar_person_num = self.current_frame_face_X_e_distance_list.index(min(self.current_frame_face_X_e_distance_list))
            

                        # 7. 生成的窗口添加说明文字 / Add note on cv2 window
        #self.img_rd2 = self.draw_name(self.img_rd2)
        self.draw_note(self.img_rd2)
        
    '''def body_show(self, img_rd):
        imgRGB = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img_rd, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img_rd.shape
                print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img_rd, (cx, cy), 5, (255,0,0), cv2.FILLED)'''

    # 處理從webcam獲取的影像, 進行人臉辨識 / Face detection and recognition wit OT from input video stream
    def process(self, stream):
        # 1. 讀取存放所有人臉特徵的 csv / Get faces known from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                #開啟資料庫
                db_settings = {
                    "host": "127.0.0.1",
                    "port": 3306,
                    "user": "root",
                    "password": "",
                    "db": "student_imformation",
                    "charset": "utf8mb4"}
                conn = pymysql.connect(**db_settings)# 建立Connection物件
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, self.img_rd2 = stream.read()
                kk = cv2.waitKey(1)
                # 2. 檢測人臉 / Detect faces for frame X
                self.faces = detector(self.img_rd2, 0)

                # 3. 更新人臉計數器 / Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(self.faces)

                # 4. 更新上一幀中的人臉list / Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5. 更新上一幀和当前幀的質心list / update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1 如果當前幀和上一幀人臉數没變化 / if cnt not changes
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("scene 1: 當前幀和上一幀相比没有發生人臉數變化 / No face cnt changes in this frame!!!")
                    self.test_def1()
                    #self.current_frame_face_position_list = []

                #如果當前幀和上一幀人臉數發生變化 / If cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    logging.debug("scene 2: 當前幀和上一幀相比人臉數發生變化 / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    #人臉數减少 / Face cnt decreases: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        logging.debug("  scene 2.1 人臉消失, 當前幀中没有人臉 / No faces in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                    # 人臉數增加 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        logging.debug("  scene 2.2 出現人臉, 進行人臉辨識 / Get faces in this frame and do face recognition")
                        #self.current_frame_face_name_list = []
                        self.test_def2()
                        if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                self.show_chinese_name()
                                self.current_frame_face_name_list[self.k] = self.face_name_known_list[self.similar_person_num]
                                self.img_rd2 = self.draw_name()
                                #self.save_imformation()
                                logging.debug("  Face recognition result: %s",
                                              self.face_name_known_list[self.similar_person_num])
                                with open('Attendance.csv', 'r+', encoding='utf-8-sig') as f:
                                    test2 = self.get_temp()
                                    myDataList = f.readlines()
                                    nameList = []
                                    for line in myDataList:
                                        entry = line.split(',')
                                        nameList.append(entry[0])
                                    if self.current_frame_face_name_list[self.k] not in nameList:
                                            now = time.localtime()
                                            dtString = time.strftime('%Y/%m/%d %H:%M:%S',now)
                                            f.writelines(f'\n{self.current_frame_face_name_list[self.k]}, {test2}, {dtString}')
                                    elif self.current_frame_face_name_list[self.k]  in nameList:
                                            #time.sleep(2)
                                            now = time.localtime()
                                            dtString = time.strftime('%Y/%m/%d %H:%M:%S',now)
                                            f.writelines(f'\n{self.current_frame_face_name_list[self.k]}, {test2}, {dtString}')
                                
                                with open('Attendance.csv', newline='',encoding='utf-8-sig') as csvfile:# 開啟 CSV 檔案 
                                    #rows = csv.DictReader(csvfile) # 讀取 CSV 檔案內容
                                    csv_reader = csv.reader(csvfile)
                                    a=list(csv_reader)[-1][:]
                                    cell = ""
                                    #for self.row2 in rows:# 以迴圈輸出每一列
                                    self.dbname=a[0]
                                    self.tem=a[1] 
                                    self.tim=a[2] 
                                    with conn.cursor() as cursor: # 建立Cursor物件   
                                            for i in range(1,100): 
                                                command = "SELECT student_Temp"+str(i)+" FROM student_imformation.demo where student_Name='"+self.dbname+"';"
                                                cursor.execute(command)# 執行指令
                                                result = cursor.fetchall()# 取得所有資料              
                                                for tem in result:
                                                    if tem[0]==None:
                                                        cell="o"
                                                if cell=="o":
                                                    break
                                                
                                            sql = "UPDATE student_imformation.demo SET student_Temp"+str(i)+"='" + self.tem + "' WHERE student_Name='"+self.dbname+"';"
                                            sql2 = "UPDATE student_imformation.demo SET student_Date"+str(i)+"='" + self.tim + "' WHERE student_Name='"+self.dbname+"';"
                                            cursor.execute(sql)
                                            cursor.execute(sql2)
                                            conn.commit()#儲存變更
                                
                        else:
                                logging.debug("  Face recognition result: 不認識 person")

                        # 7. 生成的視窗添加說明文字 / Add note on cv2 window
                            

                        # cv2.imwrite("debug/debug_" + str(self.frame_cnt) + ".png", img_rd) # Dump current frame image if needed
                        '''with open('Attendance.csv', newline='',encoding='utf-8-sig') as csvfile:# 開啟 CSV 檔案 
                            #rows = csv.DictReader(csvfile) # 讀取 CSV 檔案內容
                            csv_reader = csv.reader(csvfile)
                            a=list(csv_reader)[-1][:]
                            cell = ""
                            #for self.row2 in rows:# 以迴圈輸出每一列
                            self.dbname=a[0]
                            self.tem=a[1] 
                            self.tim=a[2] 
                            with conn.cursor() as cursor: # 建立Cursor物件   
                                    for i in range(1,100): 
                                        command = "SELECT student_Temp"+str(i)+" FROM student_imformation.demo where student_Name='"+self.dbname+"';"
                                        cursor.execute(command)# 執行指令
                                        result = cursor.fetchall()# 取得所有資料              
                                        for tem in result:
                                            if tem[0]==None:
                                                cell="o"
                                        if cell=="o":
                                            break
                                        
                                    sql = "UPDATE student_imformation.demo SET student_Temp"+str(i)+"='" + self.tem + "' WHERE student_Name='"+self.dbname+"';"
                                    sql2 = "UPDATE student_imformation.demo SET student_Date"+str(i)+"='" + self.tim + "' WHERE student_Name='"+self.dbname+"';"
                                    cursor.execute(sql)
                                    cursor.execute(sql2)
                                    conn.commit()#儲存變更'''
                        
                        
                self.draw_note(self.img_rd2)    
                #self.body_show(self.img_rd2)
                # 8. 按下 'q' 鍵退出 / Press 'q' to exit
                if kk == ord('q'):
                    ser.close()
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", self.img_rd2)

                logging.debug("Frame ends\n\n")

    def run(self):
        # cap = cv2.VideoCapture("video.mp4")  # Get video stream from video file
        cap = cv2.VideoCapture(0)              # Get video stream from camera
        cap.set(3,640)
        cap.set(4,480)
        self.process(cap)
        cap.release()
        cv2.destroyAllWindows()


def main():
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()
