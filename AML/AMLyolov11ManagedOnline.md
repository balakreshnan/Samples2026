# Deploy Yolo 11n in Azure Machine Learning Managed Online Endpoint

## Introduction

- Ability to deploy Yolo 11n model in Azure Machine Learning Managed Online Endpoint.
- Download Yolo 11n model from Hugging Face Model Hub.
- Create a scoring script to load the model and perform inference.
- Test and validate the model in compute instance
- Register the model in Azure Machine Learning workspace.
- Deploy the model to a managed online endpoint.
- Test the deployed endpoint with sample input data.
- Create Environment with required dependencies.

## Create Azure Machine Learning Workspace

- Create a brand new Azure Machine Learning Workspace in Azure Portal.
- Create a new resource group or use an existing one.
- Now for storage account provide these RBAC roles to the AML workspace managed identity.
  - Storage Blob Data Contributor
  - Storage File Privileged Data Contributor
- Create a new Azure Machine Learning Compute Instance in the AML workspace.
- Choose the appropriate size for the compute instance based on your requirements.
- SKU i am using is Standard_D32a_v4

## Code Example

### Setup Compute Instance

- First create a new compute instance in Azure Machine Learning workspace.
- Now go to terminal and run the following commands to create a yolo11env python virtual environment with required dependencies.

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create --name yolo11env -y python=3.12
conda activate yolo11env
conda install pip -y
conda install ipykernel -y
python -m ipykernel install --user --name yolo11env --display-name "yolo11env"

pip install ultralytics
pip install azureml-inference-server-http
pip install azure-monitor-opentelemetry-exporter
pip install pip install azure-ai-ml
pip install pip install azure-identity
```

- First is to accept the terms of service for Anaconda packages.
- Next create a new conda environment named yolo11env with python 3.12
- Activate the environment and install pip and ipykernel.
- Install ultralytics package to use yolo 11n model.
- Install azureml-inference-server-http package to create scoring script for Azure ML managed online endpoint.
- Install azure-monitor-opentelemetry-exporter package for monitoring.
- Then install azure-ai-ml and azure-identity packages to register and deploy model.
- Next create new folders one called models and another called src.
- Upload the yolo11n.pt model file from Hugging Face Model Hub to models folder.
- Create the score.py in src folder.
- here is the file structure

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/yolo11nonline-10.jpg 'yolo11n Managed Online Endpoint Test Result')

- Next create a new python script named score.py with the following code to load the model and perform inference.

```python
import os
import json
import requests
from ultralytics import YOLO
from PIL import Image
import io

def init():
    """
    Initialize the YOLO model once when the endpoint starts.
    """
    global model
    try:
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "yolo11n.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = YOLO(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def run(raw_data):
    """
    Run inference on an image URL.
    Expects raw_data as JSON string: {"image_url": "<URL>"}
    Returns JSON dict with YOLO results or error message.
    """
    try:
        # Parse input
        data = json.loads(raw_data)
        if "image_url" not in data:
            return {"error": "Missing 'image_url' in request"}
        image_url = data["image_url"]

        # Download image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))

        # Run YOLO inference
        results = model(img)
        result = results[0]
        serialized_result = result.to_json()

        return serialized_result

    except requests.exceptions.RequestException as re:
        return {"error": f"Failed to download image: {str(re)}"}
    except Exception as e:
        return {"error": str(e)}
```

- The init() function loads the yolo11n model from the model directory when the endpoint starts.
- The run() function takes input as a JSON string containing an image URL, downloads the image
- performs inference using the yolo11n model, and returns the results as a JSON dict.

## Register and Deploy Model

- We are using AML SDK v2 to register and deploy the model.
- Import necessary packages like os, json, requests, ultralytics YOLO, PIL Image, and io.

```
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import (
ManagedOnlineEndpoint,
ManagedOnlineDeployment,
Model,
Environment,
CodeConfiguration,
)
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
```

- Authenticate MLClient using DefaultAzureCredential.

```python
subscription_id = "xxxxxxxxxxxxxxxxxxxxxxxx"
resource_group = "rgname"
workspace = "workspace_name"

ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
```

- Now register the model

```
model_name = 'yolo11n-object'
model_local_path = "models"
model = ml_client.models.create_or_update(
        Model(name=model_name, path=model_local_path, type=AssetTypes.CUSTOM_MODEL, description="Yolov11n Model")
)
```

### Managed Online Endpoint and Deployment

- Create Managed Online Endpoint
- Create Endpoint with unique name

```
endpoint_name = "yolo11n-object-20251209-1"
```

- Create the endpoint

```python
import datetime

endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="An online endpoint to do object detection using yolov11n",
    auth_mode="key",
    tags={"env": "dev"},
)
```

- Create or update the endpoint in AML workspace

```python
poller = ml_client.begin_create_or_update(endpoint)
result = poller.result()   # waits until completion

print("Status:", poller.status())
print("Done:", result)
```

- let's create the score.py file in the src folder if not already created.

```
%%writefile src/score.py
import os
import json
import requests
from ultralytics import YOLO
from PIL import Image
import io

def init():
    """
    Initialize the YOLO model once when the endpoint starts.
    """
    global model
    try:
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "yolo11n.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = YOLO(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def run(raw_data):
    """
    Run inference on an image URL.
    Expects raw_data as JSON string: {"image_url": "<URL>"}
    Returns JSON dict with YOLO results or error message.
    """
    try:
        # Parse input
        data = json.loads(raw_data)
        if "image_url" not in data:
            return {"error": "Missing 'image_url' in request"}
        image_url = data["image_url"]

        # Download image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))

        # Run YOLO inference
        results = model(img)
        result = results[0]
        serialized_result = result.to_json()

        return serialized_result

    except requests.exceptions.RequestException as re:
        return {"error": f"Failed to download image: {str(re)}"}
    except Exception as e:
        return {"error": str(e)}
```

- Now create the environment with required dependencies
- Create Environment from environment.yml

```python
%%writefile environment.yml
name: yolo11env
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - opencv
  - pip:
    - azureml-inference-server-http
    - inference-schema[numpy-support]
    - joblib
    - Pillow
    - ultralytics
    - azure-monitor-opentelemetry-exporter
```

- Save the above file the root folder.
- Now create the environment in AML workspace

```python
yolo_env = Environment(
    name="yolo11-env",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    conda_file="environment.yml"  # <-- now pointing to a real file
)

ml_client.environments.create_or_update(yolo_env)
```

- Now create the blue deployment for the endpoint

```python
from azure.ai.ml.entities import ManagedOnlineDeployment

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    # model=model,
    model=Model(path="./models/yolo11n.pt"),
    environment=yolo_env,
    code_path="./src",
    scoring_script="score.py",
    instance_type="Standard_D16a_v4",
    instance_count=1,
    app_insights_enabled=True,
)
```

- Create or update the deployment

```python
poller = ml_client.online_deployments.begin_create_or_update(blue_deployment)

# This will block until deployment is finished (Succeeded or Failed)
deployment_result = poller.result()

print("Deployment finished!")
print("Deployment name:", deployment_result.name)
print("Provisioning state:", deployment_result.provisioning_state)
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/yolo11nonline-1.jpg 'yolo11n Managed Online Endpoint Test Result')

- When completed successfully you will see the below image.
  
![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/yolo11nonline-2.jpg 'yolo11n Managed Online Endpoint Test Result')

- Set the traffic to blue deployment

```python
endpoint.traffic = {"blue": 100}
```

- Update the endpoint with traffic settings

```python
# assume ml_client and endpoint object are defined
poller = ml_client.begin_create_or_update(endpoint)

# Wait until the endpoint is fully created
endpoint_result = poller.result()  # blocks here until completion

print("Endpoint creation finished!")
print("Provisioning state:", endpoint_result.provisioning_state)
print("Endpoint name:", endpoint_result.name)
```

- showing progress

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/yolo11nonline-3.jpg 'yolo11n Managed Online Endpoint Test Result')

- Once completed

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/yolo11nonline-4.jpg 'yolo11n Managed Online Endpoint Test Result')

- Managed online endpoint is created and traffic is set to blue deployment.

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/yolo11nonline-5.jpg 'yolo11n Managed Online Endpoint Test Result')

- Get logs

```
ml_client.online_deployments.get_logs(
    name="blue",
    endpoint_name=endpoint_name,
    lines=200
)
```

- Test the endpoint with sample input data

```python
import json

payload = {"image_url": "https://ultralytics.com/images/bus.jpg"}

# Write payload to a temp file
with open("input.json", "w") as f:
    json.dump(payload, f)

# Invoke endpoint using request_file
response = ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_name,
    deployment_name="blue",
    request_file="input.json",   # must be a file path
)

print(response)
```

- ouptut will be like below

```
"[{\"name\":\"bus\",\"class\":5,\"confidence\":0.94015,\"box\":{\"x1\":3.83272,\"y1\":229.36424,\"x2\":796.19458,\"y2\":728.41229}},{\"name\":\"person\",\"class\":0,\"confidence\":0.88822,\"box\":{\"x1\":671.01721,\"y1\":394.83307,\"x2\":809.80975,\"y2\":878.71246}},{\"name\":\"person\",\"class\":0,\"confidence\":0.87825,\"box\":{\"x1\":47.40473,\"y1\":399.56512,\"x2\":239.30066,\"y2\":904.19501}},{\"name\":\"person\",\"class\":0,\"confidence\":0.85577,\"box\":{\"x1\":223.05899,\"y1\":408.68857,\"x2\":344.46762,\"y2\":860.43573}},{\"name\":\"person\",\"class\":0,\"confidence\":0.62192,\"box\":{\"x1\":0.02171,\"y1\":556.06854,\"x2\":68.88546,\"y2\":872.35919}}]"
```

- Now inference the image and show the predicted boxes on the image

```python
import json
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display, Image as IPythonImage

# ─── Call endpoint ──────────────────────────────────────────────────
payload = {"image_url": "https://ultralytics.com/images/bus.jpg"}

with open("input.json", "w") as f:
    json.dump(payload, f)

response = ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_name,
    deployment_name="blue",
    request_file="input.json"
)

# ─── Parse response (handle common string-wrapped JSON) ─────────────
try:
    if isinstance(response, str):
        cleaned = response.strip('"').replace('\\"', '"')
        detections = json.loads(cleaned)
    else:
        detections = json.loads(response)
except Exception as e:
    print("Failed to parse:", e)
    print("Response:", repr(response))
    detections = []

# ─── Download image ─────────────────────────────────────────────────
img_url = "https://ultralytics.com/images/bus.jpg"
img_resp = requests.get(img_url)
img = Image.open(BytesIO(img_resp.content))
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ─── Draw detections ────────────────────────────────────────────────
class_colors = {0: (0, 255, 0), 5: (0, 165, 255)}  # person: green, bus: orange

for det in detections:
    try:
        box = det["box"]
        x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
        
        label = det["name"]
        conf = det["confidence"]
        cls = det["class"]
        
        color = class_colors.get(cls, (255, 0, 255))
        
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
        
        text = f"{label} {conf:.1%}"
        cv2.rectangle(img_cv, (x1, y1-35), (x1+len(text)*13, y1), color, -1)
        cv2.putText(img_cv, text, (x1+4, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
                    
    except Exception as e:
        print(f"Bad detection: {det} → {e}")

# ─── Save & Display (notebook friendly) ─────────────────────────────
cv2.imwrite("bus_detected_azureml.jpg", img_cv)
print("Done! Image saved as bus_detected_azureml.jpg")

# Show using matplotlib (recommended)
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(14, 10))
plt.imshow(img_rgb)
plt.axis('off')
plt.title("YOLO11n Detection Result")
plt.show()

# Alternative: pure IPython display
# display(IPythonImage(filename="bus_detected_azureml.jpg"))
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/yolo11nonline-6.jpg 'yolo11n Managed Online Endpoint Test Result')

- Now display the captured image with detected boxes

```
from IPython.display import Image, display

# After drawing
cv2.imwrite("bus_detected_azureml.jpg", img_cv)
print("Done! Image saved as: bus_detected_azureml.jpg")

# Show inline in notebook
display(Image(filename="bus_detected_azureml.jpg"))
```

- output will be like below

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/yolo11nonline-7.jpg 'yolo11n Managed Online Endpoint Test Result')

- Let's clean up the resources by deleting the endpoint

```python
# Step 1: Get current endpoint
endpoint = ml_client.online_endpoints.get(endpoint_name)

# Step 2: Set traffic for "blue" to 0% (remove all traffic)
# If you have multiple deployments, keep others or set one to 100%
traffic = {"blue": 0}   # or {} if you want to remove all traffic

# Alternative: if you have another deployment (e.g. "green"), you can do:
# traffic = {"green": 100}

updated_endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    traffic=traffic
)

print("Updating traffic allocation to 0% for blue...")
ml_client.online_endpoints.begin_create_or_update(updated_endpoint).result()

print("Traffic updated successfully. Now deleting deployment...")

# Step 3: Delete the deployment (now safe)
ml_client.online_deployments.begin_delete(
    name="blue",
    endpoint_name=endpoint_name
).result()

print("Deployment 'blue' deleted successfully!")
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/yolo11nonline-8.jpg 'yolo11n Managed Online Endpoint Test Result')

- Once completed

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/yolo11nonline-9.jpg 'yolo11n Managed Online Endpoint Test Result')

- Done! You have successfully deployed Yolo 11n model in Azure Machine Learning Managed Online Endpoint.