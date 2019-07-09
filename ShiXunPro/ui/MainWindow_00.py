# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow_00.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from audio import start, stop
from ui.Warning import Ui_Form


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # frame 识别结果
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 240, 641, 211))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setLineWidth(0)
        self.frame.setObjectName("frame")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(10, 8, 54, 12))
        self.label_2.setObjectName("label_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.frame)
        self.textBrowser.setGeometry(QtCore.QRect(10, 20, 621, 171))
        self.textBrowser.setToolTip("")
        self.textBrowser.setObjectName("textBrowser")
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
        self.pushButton_2.setGeometry(QtCore.QRect(280, 150, 75, 23))
        self.pushButton_2.setFlat(True)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_3.setGeometry(QtCore.QRect(430, 150, 75, 23))
        self.pushButton_3.setCheckable(False)
        self.pushButton_3.setFlat(True)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_2.clicked.connect(self.enter_record)
        # frame_3 录音
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(0, 0, 641, 241))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        # 不显示
        self.frame_3.setVisible(False)
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_4.setGeometry(QtCore.QRect(180, 150, 75, 23))
        self.pushButton_4.setCheckable(False)
        self.pushButton_4.setFlat(True)
        self.pushButton_4.setText("开始录音")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.start_record)
        self.pushButton_5 = QtWidgets.QPushButton(self.frame_3)
        self.pushButton_5.setGeometry(QtCore.QRect(280, 150, 75, 23))
        self.pushButton_5.setCheckable(False)
        self.pushButton_5.setFlat(True)
        self.pushButton_5.setText("停止录音")
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.return_main)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # self.setWindowOpacity(0.9)  # 设置窗口透明度
        # self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        # self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        MainWindow.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)  # 无边框，置顶
        # MainWindow.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 透明背景色

        self.btn_red = QtWidgets.QPushButton(self.frame_2)  # 关闭按钮
        self.btn_yellow = QtWidgets.QPushButton(self.frame_2)  # 空白按钮
        self.btn_blue = QtWidgets.QPushButton(self.frame_2)  # 最小化按钮
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
        # self.btn_blue.clicked.connect(self.button_mini)
        self.btn_yellow.clicked.connect(self.button_Warning)
        self.btn_blue.clicked.connect(self.enter_record)
        # self.pushButton_5 = QtWidgets.QPushButton(self.frame_3)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "result"))
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
        stop()

    def start_record(self):
        start()

    def btn_start_recongnition(self):
        return

    def button_Warning(self, event):
        reply = QMessageBox.warning(self,
                                    "消息框标题",
                                    "这是一条警告！",
                                    QMessageBox.Yes | QMessageBox.No)

    def button_information(self, event):
        reply = QMessageBox.question(self,
                                     "消息框标题",
                                     "这是一条问答吗？",
                                     QMessageBox.Yes | QMessageBox.No)

    def button_file(self):
        QFileDialog.getOpenFileName(self,
                                    "选取文件",
                                    "C:/",
                                    "All Files (*);;Text Files (*.txt)")


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    # widget=QtWidgets.QWidget()
    window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    window.show()
    # ui.retranslateUi(window)
    sys.exit(app.exec_())
