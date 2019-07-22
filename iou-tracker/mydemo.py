#!/usr/bin/env python

# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2017 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
# ---------------------------------------------------------

from time import time
import argparse
import glob
import os

from iou_tracker import track_iou
from util import load_viva, save_to_csv


def main(args):
    for model_result in os.listdir(args.detection_dir):
        detection_path = os.path.join(args.detection_dir, model_result)
        outdir = os.path.join(args.output_dir, model_result)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        dpaths = glob.glob(detection_path+'/*.txt')
        total_time = 0
        total_frames = 0
        for dpath in dpaths:
            detections = load_viva(dpath)
            name = os.path.basename(dpath)
            print("Processing %s."%(name[:-4]))

            start = time()
            tracks = track_iou(detections, args.sigma_l, args.sigma_h, args.sigma_iou, args.t_min, start_idx=0)
            end = time()
            total_time += end - start

            total_frames += len(detections)

            save_to_csv(os.path.join(outdir, name), tracks)

        print("Model: %s Total Tracking took: %.3f for %d frames or %.1f FPS"%(model_result, total_time,total_frames,total_frames/total_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="IOU Tracker demo script")
    parser.add_argument('-d', '--detection_dir', type=str, default='/new_home/mnt/10011_data8/Diana/hand-detection-10011/result/VIVA-tracking-test-result/model_',#required=True,
                        help="full path to CSV file containing the detections")
    parser.add_argument('-o', '--output_dir', type=str, default='track_results/model_',#required=True,
                        help="output directory to store the tracking results (MOT challenge devkit compatible format)")
    parser.add_argument('-sl', '--sigma_l', type=float, default=0,
                        help="low detection threshold")
    parser.add_argument('-sh', '--sigma_h', type=float, default=0.5,
                        help="high detection threshold")
    parser.add_argument('-si', '--sigma_iou', type=float, default=0.5,
                        help="intersection-over-union threshold")
    parser.add_argument('-tm', '--t_min', type=float, default=2,
                        help="minimum track length")

    args = parser.parse_args()

    main(args)
