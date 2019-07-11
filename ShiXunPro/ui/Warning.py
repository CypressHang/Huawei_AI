# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Warning.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Warning(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(411, 267)
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(60, 70, 271, 131))
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(80, 50, 151, 16))
        self.label.setObjectName("label")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Warning"))
        self.groupBox.setTitle(_translate("Dialog", "Warning"))
        self.label.setText(_translate("Dialog", "自毁程序已启动！！！！！"))



if __name__ == '__main__':
    Dialog = QtWidgets.QDialog()
    ui = Ui_Warning()
    ui.setupUi(Dialog)
    Dialog.show()
    Dialog.exec_()
