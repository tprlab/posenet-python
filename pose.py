import tensorflow as tf
import cv2 as cv
import time
import argparse
import os
import logging

import posenet


tf_sess = None
model_cfg = None
model_outputs = None


def init(model = 50):
  global tf_sess, model_cfg, model_outputs

  t = time.time()

  tf_sess = tf.Session()
  model_cfg, model_outputs = posenet.load_model(model, tf_sess)

  t = time.time() - t
  logging.debug("Model loaded in {:.4f} secs".format(t))


def process_img(img):
  global tf_sess, model_cfg, model_outputs

  t = time.time()

  output_stride = model_cfg['output_stride']

  input_image, _, output_scale = posenet.process_input(
            img, output_stride=output_stride)

  heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = tf_sess.run(
      model_outputs,
      feed_dict={'image:0': input_image}
  )

  pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
      heatmaps_result.squeeze(axis=0),
      offsets_result.squeeze(axis=0),
      displacement_fwd_result.squeeze(axis=0),
      displacement_bwd_result.squeeze(axis=0),
      output_stride=output_stride,
      max_pose_detections=10,
      min_pose_score=0.25)

  t = time.time() - t
  logging.debug("Pose estimated in {:.4f} secs".format(t))


  keypoint_coords *= output_scale
  return pose_scores, keypoint_scores, keypoint_coords


def get_pic(img):
  pose_scores, keypoint_scores, keypoint_coords = process_img(img)

  return posenet.draw_skel_and_kp(
      img, pose_scores, keypoint_scores, keypoint_coords,
      min_pose_score=0.25, min_part_score=0.25)


def get_keypoints(img):
  pose_scores, keypoint_scores, keypoint_coords = process_img(img)

  ret = []
  for pi in range(len(pose_scores)):
    if pose_scores[pi] == 0.:
      break
    p = {}
    p["id"] = pi
    kp = []
    p["kp"] = kp
    ret.append(p)
    
    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
      kp.append({"c" : ki, "s" : round(s,2), "p" : [int(x) for x in c]})
      #print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
  return ret

if __name__ == "__main__":
  init()
  img = cv.imread("in/image.jpg")
  kp = get_pose_keypoints(img)
  print (kp)
  ppic = get_pose_pic(img)
  cv.imwrite("posenet.jpg", ppic)
