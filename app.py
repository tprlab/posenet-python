import os
import threading
import datetime
import time
import logging
import io
import numpy as np
import traceback
import cv2 as cv

if not os.path.isdir("logs"):
    os.makedirs("logs")        

logging.basicConfig(filename="logs/pose.log",level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(threadName)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


from flask import Flask
from flask import send_file, send_from_directory
from flask import jsonify
from flask import request

import pose

app = Flask(__name__)

pose.init()

def get_request_file(request):
  if 'file' not in request.files:
      return None

  file = request.files['file']
  input_file = io.BytesIO()
  file.save(input_file)
  return np.fromstring(input_file.getvalue(), dtype=np.uint8)


def send_pic(pic):
  _, jpg = cv.imencode(".jpg", pic)
  out_file = io.BytesIO()
  out_file.write(jpg)
  out_file.seek(0)
  return send_file(out_file, mimetype="image/jpeg")

def process_request(request, work_proc, out_proc):
  data = get_request_file(request)
  if data is None:
      "file", 400
  
  try:
    data = cv.imdecode(data, cv.IMREAD_UNCHANGED)
    out = work_proc(data)
    return out_proc(out)
  except Exception as e:
    logging.exception("Request failed")
    traceback.print_exc()
    return str(e), 400
    

@app.route('/')
def index():
    return 'Posenet REST Service'


@app.route('/pose/keypoints', methods=['POST'])
def keypoints():
  return process_request(request, pose.get_keypoints, jsonify)

@app.route('/pose/pic', methods=['POST'])
def pose_pic():
  return process_request(request, pose.get_pic, send_pic)



if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80, debug=True, threaded=False, use_reloader=False)

