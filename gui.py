import sys
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QCheckBox
from PyQt5.QtGui import QPixmap, QImage
from attack import attack

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Title
        self.title_label = QLabel("Physical Attacks Demo", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        font = self.title_label.font()
        font.setPointSize(20)
        self.title_label.setFont(font)

        # Subtitle
        self.subtitle_label = QLabel("Choose which attacks you would like to do", self)
        self.subtitle_label.setAlignment(Qt.AlignCenter)

        # Upload button
        self.upload_button = QPushButton('Upload Picture', self)
        self.upload_button.clicked.connect(self.open_file_dialog)

        # Attack Checkboxes
        self.checkbox1 = QCheckBox('Black Boxes', self)
        self.checkbox2 = QCheckBox('Rotate', self)
        self.checkbox3 = QCheckBox('Fisheye', self)
        self.checkbox4 = QCheckBox('Dent', self)

        # Attack button
        self.attack_button = QPushButton('Attack!', self)
        self.attack_button.clicked.connect(self.perform_attack)

        # Layout for the left section
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.title_label)
        left_layout.addWidget(self.subtitle_label)
        left_layout.addWidget(self.upload_button)
        left_layout.addWidget(self.checkbox1)
        left_layout.addWidget(self.checkbox2)
        left_layout.addWidget(self.checkbox3)
        left_layout.addWidget(self.checkbox4)
        left_layout.addWidget(self.attack_button)
        left_layout.addStretch(1)

        # Layout for the right section
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Overall layout
        layout = QHBoxLayout()
        layout.addLayout(left_layout)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        self.setWindowTitle('Demo')
        self.setGeometry(100, 100, 600, 400)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            pixmap = QPixmap(fileName)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

    def perform_attack(self):
        attacks = [self.checkbox1.isChecked(), self.checkbox2.isChecked(), self.checkbox3.isChecked(), self.checkbox4.isChecked()]
        # Assuming you have already loaded the image and stored it in self.image_label.pixmap()
        # Convert the pixmap to a numpy array and pass it to the attack function
        pixmap = self.image_label.pixmap()
        image = pixmap.toImage()
        image_np = np.array(image.convertToFormat(QImage.Format_RGB888))
        attacked_image_np = attack(image_np, *attacks)
        attacked_image_pixmap = QPixmap.fromImage(QImage(attacked_image_np.data, attacked_image_np.shape[1], attacked_image_np.shape[0], QImage.Format_RGB888))
        self.image_label.setPixmap(attacked_image_pixmap)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_app = App()
    my_app.show()
    sys.exit(app.exec_())