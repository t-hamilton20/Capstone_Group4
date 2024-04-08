# Physical Attacks on Self-Driving Systems
By Fred Clark, Keegan Dillon and Tom Hamilton
## Description

This project implements a machine learning or deep learning model for classifying street signs. It includes scripts for training the model for which the Mapillary Traffic Sign dataset was used. Featuring a graphical user interface for easy interaction.

## Installation

To get started with this project, clone this repository to your local machine.

```
git clone https://github.com/t-hamilton20/Capstone_Group4.git
```

Ensure you have Python 3.x installed. Then, install the required dependencies (see requirements)

## Requirements

This project requires the following Python packages:

- torch
- torchvision
- matplotlib
- Pillow
- torchsummary
- numpy
- PyQt5
- opencv-python

You can install all required packages using the following command:

```
pip install torch torchvision matplotlib Pillow torchsummary numpy PyQt5 opencv-python
```
NOTE: use ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118``` for using PyTorch library on CUDA

### Training the Model

To train the model use `train.py`

### Testing the Model

To test the model on the entire dataset use `test_all.py`

To test the model on a single image in the dataset use `test_single_img.py`

### Using the GUI

Launch the graphical user interface for interactive model testing:

```
python gui.py
```
Utilize data/demo folder for use
