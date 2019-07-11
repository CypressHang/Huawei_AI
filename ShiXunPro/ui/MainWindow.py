# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QBasicTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QLCDNumber

from YuYinTest import TargetTest
from audio import start, stop
from ui.About import Ui_About
from ui.Warning import Ui_Warning


class Ui_MainWindow(QtWidgets.QWidget):
    filepath = ''

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 410)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # frame 识别结果
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 200, 590, 250))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setLineWidth(0)
        self.frame.setObjectName("frame")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(10, 8, 70, 20))
        self.label_2.setObjectName("label_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.frame)
        self.textBrowser.setGeometry(QtCore.QRect(5, 30, 590, 171))
        self.textBrowser.setToolTip("")
        self.textBrowser.setObjectName("textBrowser")
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.textBrowser.setFont(font)
        # frame_2 选择录音或开始识别
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(0, 0, 641, 241))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.frame_2.setVisible(True)
        self.pushButton = QtWidgets.QPushButton(self.frame_2)
        self.pushButton.setGeometry(QtCore.QRect(60, 50, 125, 125))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.pushButton.setFont(font)
        self.pushButton.setToolTip("")
        self.pushButton.setDefault(False)
        self.pushButton.setFlat(False)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setStyleSheet("QPushButton{background-image: url(./src/btn-tianjia.png)}")
        self.pushButton.clicked.connect(self.button_file)
        self.radioButton = QtWidgets.QRadioButton(self.frame_2)
        self.radioButton.setGeometry(QtCore.QRect(321, 90, 89, 16))
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.frame_2)
        self.radioButton_2.setGeometry(QtCore.QRect(430, 90, 89, 16))
        self.radioButton_2.setObjectName("radioButton_2")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setGeometry(QtCore.QRect(284, 92, 54, 12))
        self.label.setObjectName("label")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_2.setGeometry(QtCore.QRect(280, 150, 90, 30))
        self.pushButton_2.setFlat(True)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.enter_record)
        self.pushButton_2.setStyleSheet('''
                                     QPushButton
                                     {text-align : center;
                                     background-color : #58BDFF;
                                     font: bold;
                                     border-color: gray;
                                     border-width: 2px;
                                     border-radius: 10px;
                                     padding: 6px;
                                     height : 14px;
                                     border-style: outset;
                                     font : 14px;}
                                     QPushButton:pressed
                                     {text-align : center;
                                     background-color : light gray;
                                     font: bold;
                                     border-color: gray;
                                     border-width: 2px;
                                     border-radius: 10px;
                                     padding: 6px;
                                     height : 14px;
                                     border-style: outset;
                                     font : 14px;}
                                     ''')
        self.pushButton_3 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_3.setGeometry(QtCore.QRect(430, 150, 90, 30))
        self.pushButton_3.setCheckable(False)
        self.pushButton_3.setFlat(True)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.button_start_recongnition)
        self.pushButton_3.setStyleSheet('''
                             QPushButton
                             {text-align : center;
                             background-color : #58BDFF;
                             font: bold;
                             border-color: gray;
                             border-width: 2px;
                             border-radius: 10px;
                             padding: 6px;
                             height : 14px;
                             border-style: outset;
                             font : 14px;}
                             QPushButton:pressed
                             {text-align : center;
                             background-color : light gray;
                             font: bold;
                             border-color: gray;
                             border-width: 2px;
                             border-radius: 10px;
                             padding: 6px;
                             height : 14px;
                             border-style: outset;
                             font : 14px;}
                             ''')
        # frame_3 录音
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(0, 0, 641, 241))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        # 不显示
        self.frame_3.setVisible(False)
        # 计时器
        self.lcd = QtWidgets.QLCDNumber(self.frame_3)
        self.lcd.setDigitCount(8)
        self.lcd.setMode(QLCDNumber.Dec)
        self.lcd.setSegmentStyle(QLCDNumber.Flat)
        self.lcd.display("0:0:0")
        self.lcd.setGeometry(QtCore.QRect(30, 100, 200, 60))

        # 开始录音按钮
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_4.setGeometry(QtCore.QRect(400, 80, 90, 30))
        self.pushButton_4.setCheckable(False)
        self.pushButton_4.setFlat(True)
        self.pushButton_4.setText("开始录音")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.start_record)
        self.pushButton_4.setStyleSheet('''
                     QPushButton
                     {text-align : center;
                     background-color : #58BDFF;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     QPushButton:pressed
                     {text-align : center;
                     background-color : light gray;
                     font: bold;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 10px;
                     padding: 6px;
                     height : 14px;
                     border-style: outset;
                     font : 14px;}
                     ''')
        self.pushButton_5 = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_5.setGeometry(QtCore.QRect(400, 150, 90, 30))
        self.pushButton_5.setCheckable(False)
        self.pushButton_5.setFlat(True)
        self.pushButton_5.setText("停止录音")
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.return_main)
        self.pushButton_5.setStyleSheet('''
                             QPushButton
                             {text-align : center;
                             background-color : #58BDFF;
                             font: bold;
                             border-color: gray;
                             border-width: 2px;
                             border-radius: 10px;
                             padding: 6px;
                             height : 14px;
                             border-style: outset;
                             font : 14px;}
                             QPushButton:pressed
                             {text-align : center;
                             background-color : light gray;
                             font: bold;
                             border-color: gray;
                             border-width: 2px;
                             border-radius: 10px;
                             padding: 6px;
                             height : 14px;
                             border-style: outset;
                             font : 14px;}
                             ''')

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        MainWindow.setWindowFlags(QtCore.Qt.FramelessWindowHint)  # 无边框，置顶
        # self.setWindowOpacity(0.5)
        # MainWindow.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 透明背景色
        # self.setWindowFlags(QtCore.Qt.Tool)
        # self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # self.setWindowOpacity(0.9)  # 设置窗口透明度

        self.btn_red = QtWidgets.QPushButton(self.frame_2)  # 关闭按钮
        self.btn_yellow = QtWidgets.QPushButton(self.frame_2)  # warning按钮
        self.btn_blue = QtWidgets.QPushButton(self.frame_2)  # about按钮
        self.btn_red.setFixedSize(15, 15)  # 设置关闭按钮的大小
        self.btn_yellow.setFixedSize(15, 15)  # 设置按钮大小
        self.btn_blue.setFixedSize(15, 15)  # 设置最小化按钮大小
        self.btn_red.setGeometry(QtCore.QRect(0, 5, 0, 0))
        self.btn_yellow.setGeometry(QtCore.QRect(15, 5, 0, 0))
        self.btn_blue.setGeometry(QtCore.QRect(30, 5, 0, 0))
        self.btn_red.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.btn_yellow.setStyleSheet(
            '''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.btn_blue.setStyleSheet(
            '''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')

        self.btn_red.clicked.connect(QtCore.QCoreApplication.instance().exit)
        self.btn_yellow.clicked.connect(self.button_Warning)
        self.btn_blue.clicked.connect(self.button_about)
        # self.pushButton_5 = QtWidgets.QPushButton(self.frame_3)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "识别结果:"))
        qfont = QFont("Microsoft YaHei", 11, 75)
        self.label_2.setFont(qfont)
        # self.pushButton.setText(_translate("MainWindow", "PushButton"))
        self.radioButton.setText(_translate("MainWindow", "中文"))
        self.radioButton_2.setText(_translate("MainWindow", "英文"))
        self.label.setText(_translate("MainWindow", "语种:"))
        self.pushButton_2.setText(_translate("MainWindow", "录音"))
        self.pushButton_3.setText(_translate("MainWindow", "开始识别"))
        # self.menuHELP.setTitle(_translate("MainWindow", "Help"))

    def enter_record(self):
        self.frame_2.setVisible(False)
        self.frame_3.setVisible(True)

    def return_main(self):
        self.frame_3.setVisible(False)
        self.frame_2.setVisible(True)
        self.filepath = stop()
        self.timer.stop()

    def start_record(self):
        start()
        # 新建一个QTimer对象
        self.timer = QBasicTimer()
        self.timer.start(1000, self)
        global start_hour, start_minutes, start_sec
        start_hour = time.localtime().tm_hour
        start_minutes = time.localtime().tm_min
        start_sec = time.localtime().tm_sec

    def button_Warning(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_Warning()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()

    def button_about(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_About()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()

    def button_file(self):
        fileName1, filetype = QFileDialog.getOpenFileName(self,
                                                          "选取文件",
                                                          "./",
                                                          "All Files (*);;Text Files (*.wav)")  # 设置文件扩展名过滤,注意用双分号间隔
        print(fileName1, filetype)
        self.filepath = fileName1
        return fileName1

    def button_start_recongnition(self):
        result = TargetTest(self.filepath)
        # result = "今天天器不错"
        # self.textEdit.setText(result)
        self.textBrowser.setText(result)

    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():
            hour = "0"
            minutes = time.localtime().tm_min - start_minutes
            min = "0"
            sec = str(minutes * 60 + time.localtime().tm_sec - start_sec)
            self.lcd.display(hour + ':' + min + ':' + sec)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    # widget=QtWidgets.QWidget()
    window = QtWidgets.QMainWindow()
    window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    window.show()
    # ui.retranslateUi(window)
    sys.exit(app.exec_())
