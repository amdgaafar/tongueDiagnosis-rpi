from flask import Flask, jsonify, request
import numpy as np
import cv2
import base64
import subprocess
app = Flask(__name__)

@app.route('/')
def root():
    return "<h1>Hi</h1>"

@app.route('/td')
def tongueDiagnosis():
    outp = subprocess.check_output(['../main-rpi4', '../input_images/im_1.bmp'])
    return outp

@app.route('/receive_data', methods=['POST', 'GET'])
def receive_data():
    print("======")
    print(request.files)

    all_data = []
    for i in request.files:
        file_path = request.files[i]

        img = file_path.read()
        
        nparr = np.fromstring (img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(filename= 'test.jpg', img=img)

        # Processing on the image
        subprocess.call(['../main-rpi4', 'test.jpg'])
        img = cv2.imread("output_images/im(segmented)-rpi.png")

        _, img_process_encoded = cv2.imencode('.jpg', img)
        pimg_process_encodeed_64 = base64.b64encode(img_process_encoded).decode('utf8')

        save_data = {
            "result_image": pimg_process_encodeed_64,
        }
        all_data.append(save_data)
    return jsonify(all_data)

