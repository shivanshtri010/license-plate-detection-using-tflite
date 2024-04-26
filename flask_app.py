import cv2
from flask import Flask, request, render_template, send_from_directory, url_for
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np
import os
import tempfile
import threading
import shutil
import traceback
import pytesseract

app = Flask(__name__)

# Define the upload folder within the static directory
UPLOADS_FOLDER = '/home/shivansh117/web_app/static/uploads'
app.config['UPLOADS_FOLDER'] = UPLOADS_FOLDER

# Load the object detection model and labels
modelpath = '/home/shivansh117/web_app/detect.tflite'
lblpath = '/home/shivansh117/web_app/labelmap.txt'
min_conf = 0.5
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# Load the labels
with open(lblpath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/uploads/<folder>/<filename>')
def uploaded_file(folder, filename):
    return send_from_directory(os.path.join(app.config['UPLOADS_FOLDER'], folder), filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return "No image uploaded."

    image = request.files['image'].read()
    nparr = np.frombuffer(image, np.uint8)

    try:
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except cv2.error as e:
        return "Error decoding image: " + str(e)

    # Perform object detection on the image
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    float_input = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    # Create a temporary directory in the system's temporary directory
    temp_dir = tempfile.mkdtemp(prefix='uploads_', suffix='', dir=app.config['UPLOADS_FOLDER'])
    print("Temporary Directory Path:", temp_dir)

    # Save the uploaded image to a temporary file
    uploaded_image_path = os.path.join(temp_dir, 'uploaded_image.jpg')
    try:
        cv2.imwrite(uploaded_image_path, image)
        print("Uploaded Image Path:", uploaded_image_path)
    except Exception as e:
        traceback.print_exc()
        return f"Error saving uploaded image: {str(e)}"

    # Initialize license_plate_text
    license_plate_text = ""

    for i in range(len(scores)):
        if scores[i] > min_conf and scores[i] <= 1.0:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            class_index = int(classes[i])

            if class_index < len(labels) and class_index >= 0:
                object_name = labels[class_index]

                cropped_image = image[ymin:ymax, xmin:xmax]

                # Save the cropped image to a temporary file
                cropped_image_path = os.path.join(temp_dir, 'cropped_image.jpg')
                try:
                    cv2.imwrite(cropped_image_path, cropped_image)
                    print("Cropped Image Path:", cropped_image_path)
                except Exception as e:
                    traceback.print_exc()
                    return f"Error saving cropped image: {str(e)}"

                # Use pytesseract for OCR
                license_plate_text = pytesseract.image_to_string(cropped_image, config='--psm 7')

                if license_plate_text:
                    # Delete the temporary directory after 100 seconds
                    threading.Timer(100, lambda: shutil.rmtree(temp_dir)).start()

                    return render_template('index.html', result=license_plate_text, temp_dir_name=os.path.basename(temp_dir))

    # If no license plate detected
    result = "No license plate detected"

    # Delete the temporary directory
    shutil.rmtree(temp_dir)

    return render_template('index.html', result=result, temp_dir_name=os.path.basename(temp_dir))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)
