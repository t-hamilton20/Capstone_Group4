import sys
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QCheckBox
from PyQt5.QtGui import QPixmap, QImage, QFont
from model import CustomNetwork
from torchvision import transforms
from custom_dataset import SignDataset
from preprocessing.data_augmentation import read_class_names
from attack_module.attack import single_attack, attack
import torch
from matplotlib import pyplot as plt
import argparse
import cv2
import torchvision.transforms.functional as TF
from PIL import Image


class App(QWidget):
    def __init__(self):
        super().__init__()

        self.filename = ""

        self.initUI()

    def initUI(self):
        # Title
        self.title_label = QLabel("Physical Attacks Demo", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        font = self.title_label.font()
        font.setPointSize(28)
        self.title_label.setFont(font)

        # Upload button
        self.upload_button = QPushButton('Upload Image', self)
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
        self.original_predicted_labels = [QLabel() for _ in range(5)]

        # Layout for the left section
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.title_label)
        left_layout.addWidget(self.upload_button)
        left_layout.addWidget(self.checkbox1)
        left_layout.addWidget(self.checkbox2)
        left_layout.addWidget(self.checkbox3)
        left_layout.addWidget(self.checkbox4)
        left_layout.addWidget(self.checkbox5)
        left_layout.addWidget(self.attack_button)
        left_layout.addWidget(self.test_image_button)
        left_layout.addWidget(QLabel("<b>No Attacks - Top 5 Predicted Classes:</b>"))
        for label in self.original_predicted_labels:
            label.font().setPointSize(16)
            left_layout.addWidget(label)
        left_layout.addWidget(QLabel("<b>Attacked - Top 5 Predicted Classes:</b>"))
        for label in self.predicted_labels:
            label.font().setPointSize(16)
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
        self.filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if self.filename:
            # Read the image using OpenCV
            cv_image = cv2.imread(self.filename)

            # Resize the image to your desired dimensions
            desired_height = 224
            desired_width = 224
            resized_cv_image = cv2.resize(cv_image, (desired_width, desired_height))

            # Convert the resized OpenCV image (NumPy array) back to a QImage
            height, width, channels = resized_cv_image.shape
            bytesPerLine = channels * width
            qImage = QImage(resized_cv_image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

            # Convert QImage to QPixmap and display it
            pixmap = QPixmap.fromImage(qImage)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

    def perform_attack(self):
        # Extracting values from checkboxes
        attacks = [self.checkbox1.isChecked(), self.checkbox2.isChecked(), self.checkbox3.isChecked(),
                   self.checkbox4.isChecked(), self.checkbox5.isChecked()]
        noisy = self.checkbox5.isChecked()  # Checkbox for Random Noise
        rotate_imgs = self.checkbox2.isChecked()  # Checkbox for Rotate
        fish_img = self.checkbox3.isChecked()  # Checkbox for Fisheye
        dented = self.checkbox4.isChecked()  # Checkbox for Dent
        add_rects = False  # Assuming add_rects is False by default

        # Convert QPixmap to NumPy array
        pixmap = self.image_label.pixmap()
        image = pixmap.toImage()
        image = image.convertToFormat(QImage.Format_RGB888)

        # Get image dimensions
        width = image.width()
        height = image.height()

        # Check if image dimensions are valid
        if width <= 0 or height <= 0:
            print("Invalid image dimensions")
            return

        # Get raw pixel data from the image
        ptr = image.bits()
        ptr.setsize(image.byteCount())

        # Convert raw pixel data to NumPy array
        image_np = np.array(ptr).reshape(height, width, 3)

        # Pass the image to the attack function
        self.attack_image(image_np)

    def test_transform(self):
        transform_list = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5])
        ]
        return transforms.Compose(transform_list)

    def test_image(self):
        args = self.parse_arguments()  # Call parse_arguments within the class

        # Convert the QPixmap to a NumPy array
        pixmap = self.image_label.pixmap()
        image = pixmap.toImage()
        image = image.convertToFormat(QImage.Format_RGB888)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        image_np = np.array(ptr).reshape(height, width, 3)

        # Convert the NumPy array to a PyTorch tensor
        img = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        original_img = Image.open(self.filename)
        transform = self.test_transform()
        tensor_image = transform(original_img)
        attacks = [self.checkbox1.isChecked(), self.checkbox2.isChecked(), self.checkbox3.isChecked(),
                   self.checkbox4.isChecked(), self.checkbox5.isChecked()]
        attacked_image = single_attack("cpu", tensor_image, attacks[0], attacks[1], attacks[2], attacks[3], attacks[4])
        pil_img = TF.to_pil_image(attacked_image)
        original_img_to_test = transform(original_img)
        img_to_test = transform(pil_img)
        print(img_to_test.size())
        pil_img.save("./data/demo/temp.jpg")


        model = CustomNetwork(None, None)
        model.load_state_dict(torch.load(args.s, map_location=torch.device('cpu')))
        model.eval()

        classes = read_class_names('./data/demo/class_names.txt')

        with torch.no_grad():
            outputs = model(original_img_to_test)
            _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)
            top5_predictions_og = [classes[predicted] for predicted in predicted_top5[0]]
        
        with torch.no_grad():
            outputs = model(img_to_test)
            _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)
            top5_predictions = [classes[predicted] for predicted in predicted_top5[0]]

        print(f"Top 5 Predictions: {top5_predictions}")

        print("Setting og")
        for i, label in enumerate(top5_predictions_og):
            self.original_predicted_labels[i].setText(label)
            self.predicted_labels[i].setText(top5_predictions[i])

        print("Setting attacked")
        for i, label in enumerate(top5_predictions):
            self.predicted_labels[i].setText(label)

    def attack_image(self, image_np):
        # Extracting values from checkboxes
        white_boxes_attack = self.checkbox1.isChecked()  # Checkbox for White Boxes
        rotate_imgs = self.checkbox2.isChecked()  # Checkbox for Rotate
        fish_img = self.checkbox3.isChecked()  # Checkbox for Fisheye
        dented = self.checkbox4.isChecked()  # Checkbox for Dent
        noisy = self.checkbox5.isChecked()  # Checkbox for Random Noise

        # Convert NumPy array to PyTorch tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Call the attack function with dynamically determined values
        attacked_image_tensor = attack(torch.device('cpu'), image_tensor, white_boxes_attack, rotate_imgs, fish_img,
                                       dented, noisy)

        # Convert PyTorch tensor back to NumPy array
        attacked_image_np = attacked_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        attacked_image_np = np.clip(attacked_image_np, 0, 255).astype(np.uint8)

        # Convert NumPy array to QPixmap and display
        attacked_image_pixmap = QPixmap.fromImage(
            QImage(attacked_image_np.data, attacked_image_np.shape[1], attacked_image_np.shape[0],
                   QImage.Format_RGB888))
        self.image_label.setPixmap(attacked_image_pixmap)

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Arguments to pass to the test module')
        parser.add_argument('-cuda', type=str, default='cpu', help='device')
        parser.add_argument('-s', type=str, default='./data/models/6_mapillary_vgg_all_classes_brightness.pth',
                            help='weight path')
        parser.add_argument('-i', type=int, default=126, help='image index to test')

        argsUsed = parser.parse_args()
        return argsUsed


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
