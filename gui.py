import sys
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QCheckBox
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont
from attack import attack
import torch

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Title
        self.title_label = QLabel("Physical Attacks Demo", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        font = self.title_label.font()
        font.setPointSize(28)
        self.title_label.setFont(font)

        # Subtitle
        self.subtitle_label = QLabel("Choose which attacks you would like to do", self)
        self.subtitle_label.setAlignment(Qt.AlignCenter)

        # Upload button
        self.upload_button = QPushButton('Upload Picture', self)
        self.upload_button.clicked.connect(self.open_file_dialog)

        # Attack Checkboxes
        self.checkbox1 = QCheckBox('White Boxes', self)
        self.checkbox2 = QCheckBox('Rotate', self)
        self.checkbox3 = QCheckBox('Fisheye', self)
        self.checkbox4 = QCheckBox('Dent', self)
        self.checkbox5 = QCheckBox('Random Noise', self)

        # Font Styling
        checkbox_font = self.checkbox1.font()
        checkbox_font.setPointSize(16)
        self.checkbox1.setFont(checkbox_font)
        self.checkbox2.setFont(checkbox_font)
        self.checkbox3.setFont(checkbox_font)
        self.checkbox4.setFont(checkbox_font)
        self.checkbox5.setFont(checkbox_font)

        # Attack button
        self.attack_button = QPushButton('Attack!', self)
        self.attack_button.clicked.connect(self.perform_attack)


        # Test Image button
        self.test_image_button = QPushButton('Test Image', self)
        self.test_image_button.clicked.connect(self.test_image)

        # Button font styling
        button_font = self.upload_button.font()
        button_font.setPointSize(16)
        self.upload_button.setFont(button_font)
        self.attack_button.setFont(button_font)
        self.test_image_button.setFont(button_font)

        # Placeholder labels for top 5 predicted classes
        self.predicted_labels = [QLabel() for _ in range(5)]

        # Layout for the left section
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.title_label)
        left_layout.addWidget(self.subtitle_label)
        left_layout.addWidget(self.upload_button)
        left_layout.addWidget(self.checkbox1)
        left_layout.addWidget(self.checkbox2)
        left_layout.addWidget(self.checkbox3)
        left_layout.addWidget(self.checkbox4)
        left_layout.addWidget(self.checkbox5)
        left_layout.addWidget(self.attack_button)
        left_layout.addWidget(self.test_image_button)
        left_layout.addWidget(QLabel("Top 5 Predicted Classes:"))
        for label in self.predicted_labels:
            left_layout.addWidget(label)
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
        # Extracting values from checkboxes
        attacks = [self.checkbox1.isChecked(), self.checkbox2.isChecked(), self.checkbox3.isChecked(),
                   self.checkbox4.isChecked()]
        noisy = self.checkbox5.isChecked()  # Checkbox for Random Noise
        rotate_imgs = self.checkbox2.isChecked()  # Checkbox for Rotate
        fish_img = self.checkbox3.isChecked()  # Checkbox for Fisheye
        dented = self.checkbox4.isChecked()  # Checkbox for Dent
        add_rects = False  # Assuming add_rects is False by default

        # Convert QPixmap to NumPy array
        pixmap = self.image_label.pixmap()
        image = pixmap.toImage()
        image = image.convertToFormat(QImage.Format_RGB888)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        image_np = np.array(ptr).reshape(height, width, 3)

        # Convert NumPy array to PyTorch tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Call the attack function with dynamically determined values
        attacked_image_tensor = attack(torch.device('cpu'), image_tensor, add_rects, rotate_imgs, fish_img, dented,
                                       noisy)

        # Convert PyTorch tensor back to NumPy array
        attacked_image_np = attacked_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        attacked_image_np = np.clip(attacked_image_np, 0, 255).astype(np.uint8)

        # Convert NumPy array to QPixmap and display
        attacked_image_pixmap = QPixmap.fromImage(
            QImage(attacked_image_np.data, attacked_image_np.shape[1], attacked_image_np.shape[0],
                   QImage.Format_RGB888))
        self.image_label.setPixmap(attacked_image_pixmap)

    def test_image(self):
        # Placeholder function for testing image recognition model
        # Replace this function with actual testing logic
        # Display top 5 predicted classes as placeholders
        font = QFont()
        font.setPointSize(16)
        predicted_classes = ["Road Work Sign - 61%", "Yield Sign - 24%", "Stop Sign - 10%", "Bicycle Sign - 3%", "Dead End Sign - 2%"]
        for label, predicted_class in zip(self.predicted_labels, predicted_classes):
            label.setText(predicted_class)
            label.setFont(font)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_app = App()
    my_app.show()
    sys.exit(app.exec_())
