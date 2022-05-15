#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 23:28:19 2022

@author: a975193
"""

import cv2
import time
import numpy as np
import tensorflow.compat.v1 as tf

input_folder = "dataset/source_videos/"
input_video = "sample_video_18.mp4"
model_path = "saved_model/saved_model"

colors = [(255,0,0), (229, 52, 235), (235, 85, 52),
          (14, 115, 51), (14, 115, 204)]

path = input_folder+input_video

cap = cv2.VideoCapture(path)
cv2.namedWindow("display", cv2.WINDOW_NORMAL)

video = cv2.VideoWriter(input_video, cv2.VideoWriter_fourcc(*'MP4V'), 30, (512,512))

def process_keypoint(kp, kp_s, h, w, img):
    for i, kp_data in enumerate(kp):
        cv2.circle(img,(int(kp_data[1]*w), int(kp_data[0]*h)),5,colors[i],-1)
    return img

with tf.Session(graph = tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['serve'], model_path)
    graph = tf.get_default_graph()
    input_tensor = graph.get_tensor_by_name("serving_default_input_tensor:0")
    det_score = graph.get_tensor_by_name("StatefulPartitionedCall:6")
    det_class = graph.get_tensor_by_name("StatefulPartitionedCall:2")
    det_boxes = graph.get_tensor_by_name("StatefulPartitionedCall:0")
    det_numbs = graph.get_tensor_by_name("StatefulPartitionedCall:7")
    det_keypoint = graph.get_tensor_by_name("StatefulPartitionedCall:4")
    det_keypoint_score = graph.get_tensor_by_name("StatefulPartitionedCall:3")
    print("Model Loaded")
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame,(512,512),interpolation = cv2.INTER_AREA)
            height, width, _ = frame.shape
            image_exp_dims = np.expand_dims(frame, axis=0)
            start_time = time.time()
            score,classes,boxes,nums_det, \
            keypoint,keypoint_score = sess.run([det_score, det_class, det_boxes, 
                                                det_numbs,det_keypoint,det_keypoint_score], 
                                                feed_dict={input_tensor:image_exp_dims})
            for i in range(int(nums_det[0])):
                if(score[0][i]*100 > 50): 
                    per_box = boxes[0][i]
                    y1 = int(per_box[0]*height)
                    x1 = int(per_box[1]*width)
                    y2 = int(per_box[2]*height)
                    x2 = int(per_box[3]*width)
        
                    p1 = (x1,y1)
                    p2 = (x2,y2)
                    cv2.rectangle(frame, p1, p2, (0,255,0), 3)
            frame = process_keypoint(keypoint[0][0], keypoint_score[0], height, width, frame)
            cv2.imshow("display",frame)
            video.write(frame)
            print("Time: ", time.time() - start_time)
            cv2.waitKey(1)
        else:
            print("break")
            break
cap.release()
video.release()
cv2.destroyAllWindows()