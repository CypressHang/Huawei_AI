# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Record.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import ui.audio as ad



class Ui_recoed(object):
    def setupUi(self, recoed):
        recoed.setObjectName("recoed")
        recoed.resize(599, 415)
        self.pushButton = QtWidgets.QPushButton(recoed)
        self.pushButton.setGeometry(QtCore.QRect(210, 180, 75, 23))
        self.pushButton.setObjectName("pushButton")

        self.retranslateUi(recoed)
        QtCore.QMetaObject.connectSlotsByName(recoed)

    def retranslateUi(self, recoed):
        _translate = QtCore.QCoreApplication.translate
        recoed.setWindowTitle(_translate("recoed", "语音识别"))
        self.pushButton.setText(_translate("recoed", "开始录音"))
        # self.pushButton.clicked.connect(self.record)
        self.pushButton.clicked.connect(self.setBrowerPath)
        # self.pushButton.actionEvent(record())

    def record(self):
        audio = ad.Audio()
        audio.get_audio()

    def setBrowerPath(self):
        # download_path = QtWidgets.QFileDialog.getExistingDirectory(self,
        #                                                            "浏览",
        #                                                            "E:\workspace")
        QtWidgets.QFileDialog.getExistingDirectory("E:/test")


if __name__=="__main__":
    import sys
    from PyQt5.QtGui import QIcon
    app=QtWidgets.QApplication(sys.argv)
    widget=QtWidgets.QWidget()
    ui=Ui_recoed()
    ui.setupUi(widget)
    # widget.setWindowIcon(QIcon('web.png'))#增加icon图标，如果没有图片可以没有这句
    widget.show()
    sys.exit(app.exec_())