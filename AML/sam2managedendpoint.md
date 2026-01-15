# Deploy Meta Sam 3 in Azure Machine Learning Managed Online Endpoint

## Introduction

- Ability to deploy Meta Sam 3 model in Azure Machine Learning Managed Online Endpoint.
- Download Meta Sam 3 model from Hugging Face Model Hub.
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
- Now go to terminal and run the following commands to create a sam3env python virtual environment with required dependencies.

```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create --name sam3env -y python=3.12
conda activate sam3env
conda install pip -y
conda install ipykernel -y
python -m ipykernel install --user --name sam3env --display-name "sam3env"
pip install git+https://github.com/facebookresearch/sam3.git
pip install azureml-inference-server-http
pip install azure-monitor-opentelemetry-exporter
pip install pip install azure-ai-ml
pip install pip install azure-identity
pip install matplotlib
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
- Next create a new python script named score.py with the following code to load the model and perform inference.


```
%%writefile src/score.py
import os
import json
import torch
from PIL import Image
import io
import requests
import numpy as np

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Global variables (loaded once during init)
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def init():
    """
    Initialize SAM3 model once when the endpoint starts.
    This function runs automatically when the container starts.
    """
    global model, processor
    import os
    from huggingface_hub import login

    # ── Add these 3-4 lines ────────────────────────────────
    # hf_token = os.environ.get("HF_TOKEN")
    hf_token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    if hf_token:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face")
    else:
        print("Warning: HF_TOKEN environment variable not found!")
    # ─────────────────────────────────────────────────────────

    try:
        # SAM3 doesn't use AZUREML_MODEL_DIR by default,
        # but we can still print environment for debugging
        print(f"Initializing SAM3 on device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        # Load SAM3 image model (downloads weights automatically on first run)
        print("Loading SAM3 image model...")
        model = build_sam3_image_model()
        model = model.to(device)
        model.eval()

        # Create processor
        processor = Sam3Processor(model)

        print("SAM3 model and processor initialized successfully")

    except Exception as e:
        error_msg = f"Failed to initialize SAM3: {str(e)}"
        print(error_msg)
        raise RuntimeError(error_msg)

def to_serializable(obj):
    """Convert torch/numpy types to JSON-friendly Python types"""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj

def run(raw_data):
    """
    Perform segmentation with text prompt.
    
    Input format (JSON string):
    {
        "image_url": "https://...",
        "prompt": "person"          // required text prompt
    }

    Output format:
    {
        "masks": [...],             // list of mask arrays (as list of lists)
        "boxes": [...],             // list of [x1,y1,x2,y2]
        "scores": [...],            // list of confidence scores
        "num_objects": N,
        "prompt": "person",
        "image_size": [width, height]
    }
    """
    try:
        # Parse input
        data = json.loads(raw_data)
        
        if "image_url" not in data:
            return {"error": "Missing 'image_url' in request"}
        if "prompt" not in data or not data["prompt"].strip():
            return {"error": "Missing or empty 'prompt' in request"}

        image_url = data["image_url"]
        text_prompt = data["prompt"].strip()

        # Download image
        print(f"Downloading image from: {image_url}")
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()

        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        print(f"Image loaded: {img.size}")

        # Prepare image
        print("Preparing image for SAM3...")
        inference_state = processor.set_image(img)

        # Run text prompt inference
        print(f"Running SAM3 inference with prompt: '{text_prompt}'")
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt
        )

        # Convert the whole dictionary safely
        safe_result = to_serializable({
            "masks": [m.cpu().numpy().squeeze().astype(bool).tolist() for m in output["masks"]],
            "boxes": [b.cpu().numpy().astype(float).tolist() for b in output["boxes"]],
            "scores": output["scores"].cpu().numpy().astype(float).tolist(),
            "num_objects": len(output["masks"]),
            "prompt": text_prompt,
            "image_size": [int(img.width), int(img.height)],
            "device": device
        })

        return json.dumps(safe_result)

    except requests.exceptions.RequestException as re:
        return {"error": f"Failed to download image: {str(re)}"}
    except Exception as e:
        error_msg = f"Inference error: {str(e)}"
        print(error_msg)
        return {"error": error_msg}
```

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
model_name = 'models--facebook--sam3'
model_local_path = "models"
model = ml_client.models.create_or_update(
        Model(name=model_name, path=model_local_path, type=AssetTypes.CUSTOM_MODEL, description="Yolov11n Model")
)
```

### Managed Online Endpoint and Deployment

- Create Managed Online Endpoint
- Create Endpoint with unique name

```
endpoint_name = "sam3ep"
```

- Create the endpoint

```
import datetime

endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="An online endpoint to do object detection using yolov11n",
    auth_mode="key",
    tags={"env": "dev"},
)
```

- Create or update the endpoint

```poller = ml_client.begin_create_or_update(endpoint)
result = poller.result()   # waits until completion

print("Status:", poller.status())
print("Done:", result)
```

- Next create Environment with required dependencies.

```
%%writefile environment.yml
name: sam3env
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
    - torch==2.7
    - torchvision 
    - torchaudio
    - matplotlib
    - pycocotools
    - decord
    - einops
    - scikit-image
    - scikit-learn
    - pandas
    - git+https://github.com/facebookresearch/sam3.git
```

- Create Environment from the environment.yml file

```
sam3_env = Environment(
    name="sam3-env",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    conda_file="environment.yml"  # <-- now pointing to a real file
)

ml_client.environments.create_or_update(sam3_env)
```

- Now create Managed Online Deployment

```
from azure.ai.ml.entities import ManagedOnlineDeployment

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    # model=model,
    model=Model(path="./models/models--facebook--sam3"),
    environment=sam3_env,
    code_path="./src",
    scoring_script="score.py",
    instance_type="Standard_NC6s_v3",
    instance_count=1,
    app_insights_enabled=True,
)
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/sam3mgendpoint-4.jpg 'yolo11n Managed Online Endpoint Test Result')

- Create or update the deployment

```
poller = ml_client.online_deployments.begin_create_or_update(blue_deployment)

# This will block until deployment is finished (Succeeded or Failed)
deployment_result = poller.result()

print("Deployment finished!")
print("Deployment name:", deployment_result.name)
print("Provisioning state:", deployment_result.provisioning_state)
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/sam3mgendpoint-1.jpg 'yolo11n Managed Online Endpoint Test Result')

- Once deployment is finished, we need to set the deployment as the default for the endpoint.

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/sam3mgendpoint-2.jpg 'yolo11n Managed Online Endpoint Test Result')

- Setup the deployment traffic

```
endpoint.traffic = {"blue": 100}
```

- Update the endpoint to apply the traffic settings

```
# assume ml_client and endpoint object are defined
poller = ml_client.begin_create_or_update(endpoint)

# Wait until the endpoint is fully created
endpoint_result = poller.result()  # blocks here until completion

print("Endpoint creation finished!")
print("Provisioning state:", endpoint_result.provisioning_state)
print("Endpoint name:", endpoint_result.name)
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/sam3mgendpoint-5.jpg 'yolo11n Managed Online Endpoint Test Result')

- Now load the endpoint and get the keys to test the endpoint.

```
# ────────────────────────────────────────────────────────────────
# Option A - Get the Endpoint object itself
# ────────────────────────────────────────────────────────────────
try:
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    print("Endpoint found!")
    print(f"  Name:        {endpoint.name}")
    print(f"  Description: {endpoint.description}")
    print(f"  Auth mode:   {endpoint.auth_mode}")           # key / aml_token
    print(f"  Scoring URI: {endpoint.scoring_uri}")
    print(f"  Location:    {endpoint.location}")
except Exception as e:
    print(f"Endpoint '{endpoint_name}' not found or error: {e}")
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/sam3mgendpoint-6.jpg 'yolo11n Managed Online Endpoint Test Result')

- Here is the code to test the endpoint with sample input data.

```
# Option 1 - Simple & clean (most popular)
import json

payload = {"image_url": "https://ultralytics.com/images/bus.jpg",
            "prompt": "person"
        }

response = ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_name,
    deployment_name="blue",
    request_file="input.json",
)

print("Raw response length:", len(response), "characters")
print("First 200 characters of response:")
print(response[:200])
print("-" * 80)

try:
    result = json.loads(response)
    print("Successfully parsed JSON!")
    
    # Now safely access fields
    print("Inference successful!")
    print(f"Number of detected objects: {result.get('num_objects', 0)}")
    print(f"Prompt used:               {result.get('prompt', 'N/A')}")
    print(f"Image size:                {result.get('image_size', ['?', '?'])}")
    print(f"Device:                    {result.get('device', 'unknown')}")
    
    if 'scores' in result:
        scores = result['scores']
        print(f"Scores ({len(scores)}): {', '.join(f'{s:.3f}' for s in scores[:8])} ...")

    if 'masks' in result:
        print(f"Number of masks: {len(result['masks'])}")
        if result['masks']:
            h, w = len(result['masks'][0]), len(result['masks'][0][0])
            print(f"Mask resolution: {h} × {w} pixels")

except json.JSONDecodeError as e:
    print("JSON parsing FAILED!")
    print("Error position:", e.pos)
    print("Problematic snippet around error:")
    start = max(0, e.pos - 50)
    end = min(len(response), e.pos + 50)
    print(response[start:end])
    print("\nRaw response is NOT valid JSON → check score.py return value!")

except Exception as e:
    print("Unexpected error during processing:", str(e))
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/sam3mgendpoint-7.jpg 'yolo11n Managed Online Endpoint Test Result')

- Clean output after testing the endpoint

```
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

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/sam3mgendpoint-8.jpg 'yolo11n Managed Online Endpoint Test Result')

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/sam3mgendpoint-9.jpg 'yolo11n Managed Online Endpoint Test Result')

- Once completed delete the endpoint

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/sam3mgendpoint-10.jpg 'yolo11n Managed Online Endpoint Test Result')

- Here is the complete code to test the endpoint with sample input data.
- This is for development and testing purpose only.