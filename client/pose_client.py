import requests
import time
import json
import sys
import os
import numpy as np
import cv2 as cv
import io


URL = "http://localhost:8020"


def to_memfile(content):
    memfile = io.BytesIO()
    memfile.write(content)
    memfile.seek(0)
    return memfile


def pose_request(img, path, json = False):
  try:
    _, jpg = cv.imencode('.jpg', img)
    f = to_memfile(jpg)
    params = dict (file = f)
    resp = requests.post(URL + path, files=params, verify=False)
    if resp.status_code == requests.codes.ok:
      ret = resp.json() if json else resp.content
      return 0, ret
    return resp.status_code, resp.content
  except:
    return 503, None


def get_pose_keyponts(img):
  return pose_request(img, "/pose/keypoints", True)

def get_pose_pic(img):
  return pose_request(img, "/pose/pic")


if __name__ == "__main__":
  img = cv.imread(sys.argv[1])
  if img is not None:
    rc, kp = get_pose_keyponts(img)
    if rc == 0:
      print(kp)
    else:
      print ("keypoints error", rc, kp)

    rc, jpg = get_pose_pic(img)
    if rc == 0:
      with open("out.jpg", 'wb') as f:
        f.write(jpg)
    else:
      print ("pic error", rc, jpg)

 

 