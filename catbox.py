# Imports for PyQt5 Lib and Functions to be used
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget,QApplication
import sys


# alignment to PyQt Widgets
setStyleQte = """QTextEdit {
    font-family: "Courier"; 
    font-size: 12pt; 
    font-weight: 600; 
    text-align: right;
    background-color: Gainsboro;
}"""

setStyletui = """QLineEdit {
    font-family: "Courier";
    font-weight: 600; 
    text-align: left;
    background-color: Gainsboro;
}"""



class Window(QtWidgets.QWidget):
    def __init__(self):
        '''
        Initilize all the widgets then call the GuiSetup to customize them
        '''
        QtWidgets.QWidget.__init__(self)
        self.v = None
        self.layout = QtWidgets.QVBoxLayout(self)
        self.button2 = QtWidgets.QPushButton('Start New Session')
        self.font = QFont()
        self.font.setPointSize(12)
        self.chatlog = QtWidgets.QTextEdit()
        self.userinput = QtWidgets.QLineEdit()
        self.userinput.returnPressed.connect(self.AddToChatLogUser)
        # self.button2.clicked.connect(self.getBot)
        self.GuiSetup()


    def GuiSetup(self):
        '''
        Styling and Layout.
        '''
        self.chatlog.setStyleSheet(setStyleQte)
        self.userinput.setStyleSheet(setStyletui)
        self.userinput.setFont(self.font)
        self.button2.setFont(self.font)
        self.layout.addWidget(self.button2)
        self.layout.addWidget(self.chatlog)
        self.layout.addWidget(self.userinput)

    def UpdateCycle(self):
        '''
        Retrieves a new bot message and appends to the chat log.
        '''
        bmsg = self.v.getBotMessage()
        self.chatlog.setAlignment(Qt.AlignRight)
        [self.chatlog.append(m) for m in bmsg]
        self.userinput.setFocus()
    def AddToChatLogUser(self):
        '''
        Takes guest's entry and appends to the chatlog
        '''
        umsg = self.userinput.text()
        self.chatlog.setAlignment(Qt.AlignLeft)
        self.chatlog.append(umsg)
        # self.chatlog.setAlignment(Qt.AlignRight)
        self.userinput.setText("")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.setGeometry(10,10,480,480)
    win.show()
    sys.exit(app.exec_())