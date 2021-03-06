# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import multiprocessing
import threading
import pose
import face_reco_from_camera_ot as frfco
import features_extraction_to_csv as fetc
import delete



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(692, 949)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.MainBox = QtWidgets.QVBoxLayout()
        self.MainBox.setContentsMargins(-1, -1, -1, 0)
        self.MainBox.setObjectName("MainBox")
        self.TopBox = QtWidgets.QHBoxLayout()
        self.TopBox.setObjectName("TopBox")
        self.Camopen = QtWidgets.QPushButton(self.centralwidget)
        self.Camopen.setMinimumSize(QtCore.QSize(100, 100))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.Camopen.setFont(font)
        self.Camopen.setObjectName("Camopen")
        self.TopBox.addWidget(self.Camopen)
        self.Camstop = QtWidgets.QPushButton(self.centralwidget)
        self.Camstop.setMinimumSize(QtCore.QSize(100, 100))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.Camstop.setFont(font)
        self.Camstop.setObjectName("Camstop")
        self.TopBox.addWidget(self.Camstop)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.TopBox.addItem(spacerItem)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setMinimumSize(QtCore.QSize(100, 100))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.TopBox.addWidget(self.pushButton)
        self.MainBox.addLayout(self.TopBox)
        self.MidBox = QtWidgets.QHBoxLayout()
        self.MidBox.setObjectName("MidBox")
        self.view = QtWidgets.QScrollArea(self.centralwidget)
        self.view.setMinimumSize(QtCore.QSize(670, 670))
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.view.setWidgetResizable(True)
        self.view.setObjectName("view")
        self.viewform = QtWidgets.QWidget()
        self.viewform.setGeometry(QtCore.QRect(0, 0, 651, 651))
        self.viewform.setObjectName("viewform")
        self.viewlabel = QtWidgets.QLabel(self.viewform)
        self.viewlabel.setGeometry(QtCore.QRect(320, 310, 47, 12))
        self.viewlabel.setObjectName("viewlabel")
        self.view.setWidget(self.viewform)
        self.MidBox.addWidget(self.view)
        self.MainBox.addLayout(self.MidBox)
        self.botBox = QtWidgets.QHBoxLayout()
        self.botBox.setObjectName("botBox")
        self.posebut = QtWidgets.QPushButton(self.centralwidget)
        self.posebut.setMinimumSize(QtCore.QSize(100, 100))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.posebut.setFont(font)
        self.posebut.setObjectName("posebut")
        self.botBox.addWidget(self.posebut)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.botBox.addItem(spacerItem1)
        self.MainBox.addLayout(self.botBox)
        self.verticalLayout_2.addLayout(self.MainBox)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 692, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.Camopen.clicked.connect(self.onButton_1_Click)
        self.Camstop.clicked.connect(self.onButton_2_Click)
        self.pushButton.clicked.connect(self.onButton_3_Click)
        self.posebut.clicked.connect(self.onButton_4_Click)

        
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Webcam_example"))
        self.Camopen.setText(_translate("MainWindow", "???????????????????????????"))
        self.Camstop.setText(_translate("MainWindow", "??????????????????"))
        self.pushButton.setText(_translate("MainWindow", "??????????????????"))
        self.viewlabel.setText(_translate("MainWindow", "TextLabel"))
        self.posebut.setText(_translate("MainWindow", "??????????????????"))
    
    def onButton_1_Click(self):
        #self.Camstop.setText('hello wolrd')
        #pose.pose_detect()
        frfco.main()
        #self.view
        #image = QtGui.QPixmap(frfco.main()).scaled(400, 400)
        #self.viewlabel.setPixmap(QPixmap.fromImage(frfco.main()))
    
    
    
    def onButton_2_Click(self):
        #self.Camstop.setText('hello wolrd')
        #pose.pose_detect()
        fetc.main()
    
    def onButton_3_Click(self):
        #self.Camstop.setText('hello wolrd')
        #pose.pose_detect()
        delete.main()
    
    def onButton_4_Click(self):
        #self.Camstop.setText('hello wolrd')
        #pose.pose_detect()
        pose.main()

    
    
    
    
    

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    sing_thread = threading.Thread(target = Ui_MainWindow.onButton_1_Click)
    song_thread = threading.Thread(target = Ui_MainWindow.onButton_4_Click)
    sing_thread.start()
    song_thread.start()
    sing_thread.join()
    song_thread.join()
    MainWindow.show()
    sys.exit(app.exec_())

