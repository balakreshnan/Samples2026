# Yolov26 Segmenation in Azure Machine Learning Compute Instance

## Prerequisites

- Azure Subscription
- Azure Machine Learning Workspace
- Quota for Standard_D13_v2 cores

## Introduction

- This notebook demonstrates how to use the yolov26 model for image segmentation using Azure Machine Learning.
- Download from Hugging face and run the inferencing in a notebook.
- Using image and text prompts to generate segmentation masks.
- Vision based object detection and segmentation.

## Create Azure Machine Learning Workspace

- Create a brand new Azure Machine Learning Workspace in Azure Portal.
- Create a new resource group or use an existing one.
- Now for storage account provide these RBAC roles to the AML workspace managed identity.
  - Storage Blob Data Contributor
  - Storage File Privileged Data Contributor
- Create a new Azure Machine Learning Compute Instance in the AML workspace.
- Choose the appropriate size for the compute instance based on your requirements.
- SKU i am using is Standard_D13_v2

## Code Example

### Setup Compute Instance

- First create a new compute instance in Azure Machine Learning workspace.
- Now go to terminal and run the following commands to create a sam3env python virtual environment with required dependencies.

```bash
conda create --name yolo26env -y python=3.12
conda activate yolo26env
conda install pip -y
conda install ipykernel -y
python -m ipykernel install --user --name yolo26env --display-name "yolo26env"

pip install ultralytics
pip install openai-clip

pip install azureml-inference-server-http
pip install azure-monitor-opentelemetry-exporter
pip install pip install azure-ai-ml
pip install pip install azure-identity
pip install matplotlib
```

### Notebook Code

- make sure install openai-clip package to avoid errors while loading the model.

```
%pip install openai-clip
```

- now lets download the model

```
!mkdir -p ./models
!cd ./models
!curl -L -o ./models/yoloe-26n-seg.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg.pt
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/yolo26-seg-1.jpg 'yolo11n Managed Online Endpoint Test Result')

- To avoid insufficient memory issues we will use yolov26seg model which is optimized for segmentation tasks.
- in each cell let's run command to create ultralytics home directory.

```
!mkdir -p /home/azureuser/.ultralytics
!export ULTRALYTICS_HOME=/home/azureuser/.ultralytics
```

- Now download the mobile clip

```
!cd /home/azureuser/.ultralytics
!curl -L -o mobileclip2_b.ts https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/yolo26-seg-2.jpg 'yolo11n Managed Online Endpoint Test Result')

- now set the home directory path for ultralytics

```
import os
os.environ["ULTRALYTICS_HOME"] = "/home/azureuser/.ultralytics"
```

- Check for token

```
import clip
print(clip)
print(hasattr(clip, "tokenize"))
```

- now write the code to load the model

```

import os
os.environ["ULTRALYTICS_HOME"] = "/home/azureuser/.ultralytics"

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

segmodel = YOLO("./models/yoloe-26n-seg.pt")

names = ["person", "bus"]
segmodel.set_classes(names, segmodel.get_text_pe(names))

results = segmodel.predict(
    "https://ultralytics.com/images/bus.jpg",
    conf=0.1
)

r = results[0]
img = r.plot()

plt.figure(figsize=(12,8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
```

- Run the cell and wait for output

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/yolo26-seg-3.jpg 'yolo11n Managed Online Endpoint Test Result')

- You should see the segmentation masks and bounding boxes for the detected objects in the image.
- You can change the image URL and the class names to test with different images and objects.
- open ai clip, mobileclip2_b.ts and yolov26seg model are required for this to work.
- Make sure the curl URL is current versions are used from ultralytics official github repo.