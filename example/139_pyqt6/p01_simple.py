from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6.QtGui import QIcon
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('PyQt6 Window')
        self.setGeometry(100, 100, 800, 600)
        #self.setWindowIcon(QIcon('icon.png'))
        self.setFixedSize(800, 600)
        self.setStyleSheet('background-color: lightblue;')
        self.setWindowOpacity(0.4 )

app = QApplication(sys.argv)

window = Window()

window.show()

sys.exit(app.exec())