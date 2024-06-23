from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
import sys


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('PyQt6 Window')
        self.setGeometry(100, 100, 800, 600)

app = QApplication(sys.argv)

window = Window()

window.show()

sys.exit(app.exec())