# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'final_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
#import face_reco_from_camera_ot as frfco
import features_extraction_to_csv as fetc
import delete
import os
import win32api


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(450, 450)
        MainWindow.setMinimumSize(QtCore.QSize(450, 450))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(20, 20, 180, 180))
        self.pushButton.setMinimumSize(QtCore.QSize(180, 180))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.pushButton.setText("")
        self.pushButton.setAutoRepeat(False)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(240, 20, 180, 180))
        self.pushButton_2.setMinimumSize(QtCore.QSize(180, 180))
        self.pushButton_2.setText("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(20, 220, 180, 180))
        self.pushButton_3.setMinimumSize(QtCore.QSize(180, 180))
        self.pushButton_3.setText("")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(240, 220, 180, 180))
        self.pushButton_4.setMinimumSize(QtCore.QSize(180, 180))
        self.pushButton_4.setText("")
        self.pushButton_4.setObjectName("pushButton_4")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 40, 171, 131))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(260, 50, 141, 121))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(50, 250, 141, 121))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(260, 260, 141, 91))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setWordWrap(True)
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 450, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.pushButton.clicked.connect(self.onButton_1_Click)
        self.pushButton_2.clicked.connect(self.onButton_2_Click)
        self.pushButton_3.clicked.connect(self.onButton_3_Click)
        self.pushButton_4.clicked.connect(self.onButton_4_Click)
        
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "開啟人臉與溫度偵測"))
        self.label_2.setText(_translate("MainWindow", "寫入特徵到csv檔"))
        self.label_3.setText(_translate("MainWindow", "開啟作弊偵測"))
        self.label_4.setText(_translate("MainWindow", "清除學生資訊"))
    
    def onButton_1_Click(self):
        #self.Camstop.setText('hello wolrd')
        #pose.pose_detect()
        #frfco.main()
        #t0 = threading.Thread(target = frfco.main())
        #t0.start()
        #self.view
        #image = QtGui.QPixmap(frfco.main()).scaled(400, 400)
        #self.viewlabel.setPixmap(QPixmap.fromImage(frfco.main()))
        win32api.ShellExecute(0, 'open', '.\\face_reco_from_camera_ot\\face_reco_from_camera_ot.exe', '', '', 1)
    
    def onButton_2_Click(self):
        #self.Camstop.setText('hello wolrd')
        #pose.pose_detect()
        fetc.main()
    
    def onButton_3_Click(self):
        #self.Camstop.setText('hello wolrd')
        #pose.pose_detect()
        #cheating.main()
        #t1 = threading.Thread(target = pose.main())
        #os.system('C:/Users/user/Desktop/final files -exe-test/exe/cheating/cheating.exe')
        #t1.start()'''
        win32api.ShellExecute(0, 'open', '.\\cheating\\cheating.exe', '', '', 1)
    
    
    def onButton_4_Click(self):
        #self.Camstop.setText('hello wolrd')
        #pose.pose_detect()
        delete.main()
    
    
    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

