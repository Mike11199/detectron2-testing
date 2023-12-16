# install dependencies
get_ipython().system('pip install -U torch torchvision cython')
get_ipython().system("pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'")
import torch, torchvision
torch.__version__

#!git clone https://github.com/facebookresearch/detectron2 detectron2_repo
get_ipython().system('pip install -e detectron2_repo')
get_ipython().system('pip install opencv-python')


get_ipython().system('pip install pytube')
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

from pytube import YouTube

def download_youtube_video(youtube_url, output_path='./'):
    yt = YouTube(youtube_url)
    video_stream = yt.streams.filter(file_extension='mp4').first()
    video_stream.download(output_path)
    return output_path



youtube_url = 'https://www.youtube.com/shorts/SCglass2C5I'
video_path = download_youtube_video(youtube_url)




import cv2
from detectron2.utils.visualizer import Visualizer
import random




cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
predictor = DefaultPredictor(cfg)

def inference_on_video(video_path, output_video_path='output_video.mp4'):
    
    video_capture = cv2.VideoCapture(video_path)
        
    def get_random_color():
        return tuple(random.randint(0, 255) for _ in range(3))

    # Get video properties
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fps = int(video_capture.get(5))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Get class names from COCO metadata
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

    # Get class names from COCO metadata
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes

    # Store colors for each class
    class_colors = {class_name: get_random_color() for class_name in class_names}
    confidence_threshold = 0.55  
    



    counter = 1
    while True:  # Adjust the number of frames for testing
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


        
        for i in range(len(instance_scores)):
            box = instance_boxes[i]
            class_name = class_names[instance_classes[i]]
            score = instance_scores[i]
            mask = instance_masks[i]  # Get the mask for this instance

            # Check if the confidence is above the threshold
            print(score)
            print(class_name)
            print(box)
            if score >= confidence_threshold:
                # Convert to integers
                box = box.astype(int)

                # Get or generate a color for the class
                if class_name not in class_colors:
                    class_colors[class_name] = get_random_color()

                color = class_colors[class_name]

                # Draw bounding box on the frame
                frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

                # Add text annotation
                label = f"Class: {class_name}, Score: {score*100:.1f}%"
                frame = cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

                # Draw segmentation mask
                alpha = 0.2
                for x in range(frame.shape[0]):
                    for y in range(frame.shape[1]):
                        # Check if any element in the segmentation mask is True
                        if mask[x, y]:
                            # If it's True and confidence is above the threshold, set the pixel to the mask color with transparency
                            frame[x, y] = (
                                alpha * color[0] + (1 - alpha) * frame[x, y][0],
                                alpha * color[1] + (1 - alpha) * frame[x, y][1],
                                alpha * color[2] + (1 - alpha) * frame[x, y][2],
                            )
                                

        # Write the frame with drawn bounding boxes to the output video
        out.write(frame)
        print(f"processing frame: {counter}")
        counter += 1
        # if counter == 32:
        #     break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()
    print("finished!")




inference_on_video('./input_video_from_youtube.mp4')





