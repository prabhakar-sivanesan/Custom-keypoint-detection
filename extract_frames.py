#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 01:49:44 2022

@author: prabhakar
"""

import cv2
import glob

input_folder = "dataset/source_videos"
output_folder = "dataset/images"
cv2.namedWindow("display", cv2.WINDOW_NORMAL)

def main():
    count = 1
    for video_path in glob.glob(input_folder+"/*.mp4"):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if ret:
                file_path = output_folder+"/image_"+str(count)+".jpg"
                cv2.imwrite(file_path, frame)
                cv2.imshow("display", frame)
                count+=1
                cv2.waitKey(1)
            else:
                break
        cap.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()
