# License Plate Detection and Extraction using tflite

This project is a web-based application leveraging computer vision and machine learning for efficient license plate identification. It utilizes TensorFlow Lite for object detection and Tesseract OCR for character extraction.

## Installation

Install the necessary dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Dataset Download

The dataset has been downloaded from Roboflow (license-plate-detector-ogxxg)


## Usage

To use the project, follow these steps:

1. Upload an image using the provided web interface.
2. Click on the detection button to initiate license plate detection.
3. Upon detection, the system will present the license plate, uploaded, and cropped images for user verification.

## Components

### 1. Object Detection Model (TensorFlow Lite)

- A TensorFlow Lite model identifies license plates within uploaded images. It is pretrained using a tflite model and can be used on low-end devices.

### 2. Image Processing

- Images undergo decoding and resizing for efficient object detection upon upload.

### 3. TensorFlow Lite Interpreter

- The TensorFlow Lite interpreter extracts relevant data, including bounding box coordinates and class indices.

### 4. Cropping and OCR

- Detected plates are cropped, and Tesseract OCR extracts alphanumeric characters.

### 5. Temporary Storage

- A temporary directory stores uploaded and processed images, enhancing user interaction.

### 6. User Interface

- The web interface enables easy image upload, initiating plate detection with a single click.

### 7. Results Display

- Upon detection, the system presents the license plate, uploaded, and cropped images for user verification.

### 8. Cleanup Mechanism

- A thread-based cleanup deletes temporary directories after 100 seconds, optimizing resource management.

### 9. Web Technologies

- Developed using Flask, OpenCV, TensorFlow Lite, and Tesseract, the application fuses web development, computer vision, and machine learning.


