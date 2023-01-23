from flask import Flask, jsonify, request

import numpy as np
import cv2
import base64
import subprocess
import tensorflow as tf
import uuid

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
    model = tf.keras.models.load_model('unet_tongue_segmentation.h5')

    all_data = []
    for i in request.files:
        file_path = request.files[i]
        
        # Read input image
        img = file_path.read() # Getting image data
        nparr = np.fromstring (img, np.uint8) # Read from a string and convert to np array
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Convert to a matrix format
        img = cv2.resize(img, (128, 128)) # Resize the image after the conversion
        #--svae data--#
        inputImgId = uuid.uuid4()
        cv2.imwrite(filename= 'input/{inputImgId}.jpg', img=img) # save the image in the disk, for the cpp program


        # Unet Pred
        nparr = np.expand_dims(img, axis=0)
        pred_mask = model.predict(nparr)
        pred_mask_t = (pred_mask > 0.5).astype(np.uint8)
        segUnetImg = cv2.bitwise_and(img, img, mask = pred_mask_t[0])
        #--svae data--#
        unetImgId = uuid.uuid4()
        cv2.imwrite(filename= 'unet/{unetImgId}.jpg', img=img) # record the result

        # Threshold Pred
        subprocess.call(['../main-rpi4', 'test.jpg'])
        thresholdSegImg = cv2.imread("output_images/im(segmented)-rpi.png") # Reading the image again to send the output
        #--svae data--#
        thresholdImgId = uuid.uuid4()
        cv2.imwrite(filename= 'threshold/{thresholdImgId}.jpg', img=img) # record the result


        # Encode image data to base64
        _, unet_process_encoded = cv2.imencode('.jpg', segUnetImg) 
        _, threshold_process_encoded = cv2.imencode('.jpg', thresholdSegImg)

        unet_encodeed_64 = base64.b64encode(unet_process_encoded).decode('utf8')
        threshold_encoded_64 = base64.b64encode(threshold_process_encoded).decode('utf8')

        save_data = {
            "unet_result": unet_encodeed_64,
            "threshold_result": threshold_encoded_64
        }

        all_data.append(save_data)
    return jsonify(all_data)

