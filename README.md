# Real Time Person Tracking using DeepSort and Yolo_v4

For object tracking, this work uses SIMPLE ONLINE AND REALTIME TRACKING WITH A DEEP ASSOCIATION METRIC (Deep Sort) | Paper Link: [arxiv](https://arxiv.org/pdf/1703.07402.pdf)

For real-time object detection, open-source Yolo code by AlexeyAB is used ([link](https://github.com/AlexeyAB/darknet))

## Dataset
The dataset used for this project is Multi Object Tracking(MOT16). It can be downloaded from the [link](https://motchallenge.net/data/MOT16/)<br>
We expect to download dataset and put it in the root directory of this project. The directory structure will look like this (for one subset):
```
MOT16
  ├── test
  | ├── MOT16-06
  | | ├── det
  | | ├── img1
  | | ├── seqinfo.ini
  ├── train

```

## Prerequisite
Python = 3.6+
Tensorflow==1.15.0

## Setup

1. Clone the repo to your local machine. <br>
`git clone https://github.com/Computer-Vision-IIITH-2021/project-blindpc`
2. Move to the project directory.<br>
`cd project-blindpc`
3. Download dataset and set the directory structure as mentioned above.<br>

Note: For setting darknet, we have provided a seperate Google Colab notebook([darknet_demo.ipynb](darknet_demo.ipynb)). We expect the user to run it on a GPU provided in Google Colab



## Running Tracker on MOT Dataset
To run the tracker, use the following command. It will generate a text file with bounding boxes and tracing id for each detection. <br>
```
python app.py --sequence_path=./MOT16/test/MOT16-06 --detection_path=./detections/MOT16-06.npy \ 
              --output_path=./MOT16_test_results/MOT16-06.txt --min_conf=0.3 --nn_budget=100
```
The detection file can be directly downloaded from the [link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/madhav_agarwal_research_iiit_ac_in/EqBOq4hZWlpHsCDFLihz44kBYaNcv4TDwc4rZZUrpTz2NA?e=dGnS6J) or they can be generated using:<br>
```
python src/detection_process_v2.py --model=model_data/mars-small128.pb --dir_mot=./MOT16/test \
                                --detection_path=./detections/MOT16-06.npy --dir_out=./detections
```          

We have also tested different detection models like triplet and magnet model (apart from default cosine metric learning model). The models are present in [model_data](./model_data) directory. 

## Generating Videos from MOT Benchmark

The benchmark file having bounding boxes and tracking id generated above, can be use to create a visual output.<br>
The video can be generated using:
```
python result_to_video.py --mot_dir=./MOT16/test/MOT16-06/ \
                          --result_file=./MOT16_test_results/MOT16-06.txt \
                          --output_path=./videos/
```

## Real Time Object Tracking

DeepSort can be integrated with a multi-object detector to perform real-time tracking. We have used Yolo implemented in Darknet.<br>

One can run an end-to-end code using our demo file [darknet_demo.ipynb](darknet_demo.ipynb) on Google Colab <br>

OR<br>
  1. Setup Yolo on the local machine by following instructions from AlexeyAB [github repo](https://github.com/AlexeyAB/darknet)
  2. Download Yolov4 weights from the above repo and put them in the darknet root directory.
  3. Clone the entire content of this repo into the darknet folder. (Make sure to rename 'src' folder to 'src_code' of this repo, to avoid name clash.)
  4. Run the command: <br>
     `python yolo_with_deepsort.py`
 
  It will run the darknet on the `yolo_person.mp4` in videos folder and generate `Deep_sort_output.mp4` as output.

Disclaimer: <br>
This project was done as a part of the course CSE578: Computer Vision, Spring 2021, IIIT-Hyderabad.
