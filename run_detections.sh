#!/bin/bash

python3 src/detection_process_v2.py \
    --model=model_data/triplet.pb \
    --dir_mot=./MOT16/train \
    --dir_out=./MOT16_test_results/mars_model/triplet_model

python3 src/detection_process_v2.py \
    --model=model_data/triplet.pb \
    --dir_mot=./MOT16/test \
    --dir_out=./MOT16_test_results/mars_model/triplet_model
