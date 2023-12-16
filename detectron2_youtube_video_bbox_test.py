#!/usr/bin/env python
# coding: utf-8

# # Detectron2 Beginner's Tutorial
# 
# <img src="https://dl.fbaipublicfiles.com/detectron2/Detectron2-Logo-Horz.png" width="500">
# 
# Welcome to detectron2! In this tutorial, we will go through some basics usage of detectron2, including the following:
# * Run inference on images or videos, with an existing detectron2 model
# * Train a detectron2 model on a new dataset
# 
# You can make a copy of this tutorial to play with it yourself.
# 

# # Install detectron2

# In[8]:


# install dependencies
get_ipython().system('pip install -U torch torchvision cython')
get_ipython().system("pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")
import torch, torchvision
torch.__version__


# In[9]:


#!git clone https://github.com/facebookresearch/detectron2 detectron2_repo
get_ipython().system('pip install -e detectron2_repo')
get_ipython().system('pip install opencv-python')


# In[3]:


# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import cv2
import random

# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from matplotlib import pyplot as plt


# # Run a pre-trained detectron2 model

# Then, we create a detectron2 config and a detectron2 `DefaultPredictor` to run inference on this image.

# In[4]:


# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import cv2
import random

# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from matplotlib import pyplot as plt

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
predictor = DefaultPredictor(cfg)


# In[2]:


image_cats_dogs = cv2.imread("./cats_and_dogs.jpg")
RGB_img = cv2.cvtColor(image_cats_dogs, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
plt.imshow(RGB_img)
plt.show()

outputs2 = predictor(RGB_img)

v = Visualizer(image_cats_dogs[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs2["instances"].to("cpu"))
plt.figure(figsize=(15, 10))
plt.imshow(v.get_image()[:, :, ::-1])
plt.show()


image_new_york = cv2.imread("./new_york.jpg")
image_new_york_outputs = predictor(image_new_york)
image_new_york_outputs["instances"].pred_classes
image_new_york_outputs["instances"].pred_boxes
image_new_york_rgb = cv2.cvtColor(image_new_york, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15, 10))  # Adjust the figure size as needed
plt.imshow(image_new_york_rgb)
plt.show()

v = Visualizer(image_new_york_rgb[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(image_new_york_outputs["instances"].to("cpu"))
plt.figure(figsize=(15, 10))
plt.imshow(v.get_image()[:, :, ::-1])
plt.show()


# In[12]:


get_ipython().system('pip install pytube')

from pytube import YouTube

def download_youtube_video(youtube_url, output_path='./'):
    yt = YouTube(youtube_url)
    video_stream = yt.streams.filter(file_extension='mp4').first()
    video_stream.download(output_path)
    return output_path


# In[5]:


import time


def inference_on_video(video_path):
    # Load pre-trained Detectron 2 model

    

    # Open the video
    video_capture = cv2.VideoCapture(video_path)

    while True:        
        # Read a frame from the video
        ret, frame = video_capture.read()

        # Perform inference on the frame
        outputs = predictor(frame)

        
        # Visualize the results
 
        outputs2 = predictor(frame)

        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        v = v.draw_instance_predictions(outputs2["instances"].to("cpu"))
        plt.figure(figsize=(15, 10))
        plt.imshow(v.get_image()[:, :, ::-1])
        plt.show()

   

        time.sleep(1)
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close the OpenCV window
    video_capture.release()
    cv2.destroyAllWindows()


# In[6]:


youtube_url = 'https://www.youtube.com/shorts/SCglass2C5I'
video_path = download_youtube_video(youtube_url)


# In[12]:


import cv2 
inference_on_video('./EPIC DRIVING SKILLS CAUGHT ON DASH CAMERA.mp4')


# In[34]:


import cv2
from detectron2.utils.visualizer import Visualizer

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
predictor = DefaultPredictor(cfg)

def inference_on_video(video_path, output_video_path='output_video.mp4'):
    # Open the video
    video_capture = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fps = int(video_capture.get(5))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
        # Get class names from COCO metadata
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

    j = 1
    while j < 48:  # Adjust the number of frames for testing
        # Read a frame from the video
        ret, frame = video_capture.read()
        if not ret:
            break

        # Perform inference on the frame
        outputs = predictor(frame)

        # Get instances and their corresponding colors
        instances = outputs["instances"]
        instance_boxes = instances.pred_boxes.tensor.cpu().numpy()
        instance_scores = instances.scores.cpu().numpy()
        instance_classes = instances.pred_classes.cpu().numpy()
        instance_masks = instances.pred_masks.cpu().numpy()

        # Draw instance predictions on the frame
        for i in range(len(instance_scores)):
            box = instance_boxes[i]
            class_name = class_names[instance_classes[i]]

            score = instance_scores[i]

            # Convert to integers
            box = box.astype(int)

            # Draw bounding box on the frame
            frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                        # Add text annotation
            label = f"Class: {class_name}, Score: {score:.2f}"

            frame = cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame with drawn bounding boxes to the output video
        out.write(frame)
        print(f"processing frame: {j}")
        j += 1

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    print("finished!")


# In[35]:


inference_on_video('./input_video_from_youtube.mp4')


# In[23]:


from IPython.display import HTML

# Replace 'path/to/your/video.mp4' with the actual path to your video file
video_path = './output_video.mp4'

# Display the video in the notebook using HTML
HTML(f'<video width="640" height="480" controls><source src="{video_path}" type="video/mp4"></video>')

