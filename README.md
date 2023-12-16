# detectron2-video-testing-pre-trained

Boundary Box Video Test:
https://www.youtube.com/watch?v=h1MykeoDTn0

Segmentation Mask Test
https://youtube.com/shorts/reoHAhDNXyA

- Takes about 15-30 minutes to process the video in a Jupyter Lab Notebook, using a ml.t3.medium AWS SageMaker instance.
- This is due to the video containg over 500 frames that need to be processed.
- Was able to draw the segmentation masks by looping manually over each pixel in the image with cv2 and manually blending the alpha values of a class color and the initial rgb color with a blend if the mask array was true for that particular pixel.  Using the built in detectron2 functions for boundary boxes and masks did not work when processing with output.write with cv2 for video.

![image](https://github.com/Mike11199/GIFs/blob/main/detectron2videotest.gif )
![image](https://github.com/Mike11199/detectron2-testing/assets/91037796/7e00a78f-7abd-4aaf-81e3-fbe156ebe61f)
![image](https://github.com/Mike11199/detectron2-testing/assets/91037796/161231d9-37c8-4c6f-898b-92cb706b616b)
![image](https://github.com/Mike11199/detectron2-testing/assets/91037796/c6d8fb2f-3735-466f-8465-c7a67b4ee187)



# detectron2-image-testing-pre-trained

- Machine learning on my personal AWS SageMaker account.

![image](https://github.com/Mike11199/detectron2-testing/assets/91037796/b425471d-3d85-4987-b4a7-76e22f21ca5a)
![image](https://github.com/Mike11199/detectron2-testing/assets/91037796/80a9e1f6-5ff1-4432-b775-d0ce8c43ceea)
![image](https://github.com/Mike11199/detectron2-testing/assets/91037796/fca90430-a2d3-49f3-b863-e2c7468469ca)
![image](https://github.com/Mike11199/detectron2-testing/assets/91037796/063e2f97-349e-4950-950c-7eb7634daf90)
![image](https://github.com/Mike11199/detectron2-testing/assets/91037796/147fc8ec-d3df-447b-b056-cd203b076789)

