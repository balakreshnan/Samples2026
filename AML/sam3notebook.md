# Inferencing SAM3 Model in a Notebook using Azure Machine Learning

## Introduction

- This notebook demonstrates how to use the SAM3 model for image segmentation using Azure Machine Learning.
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
- SKU i am using is Standard_NC6s_v3

## Code Example

### Setup Compute Instance

- First create a new compute instance in Azure Machine Learning workspace.
- Now go to terminal and run the following commands to create a sam3env python virtual environment with required dependencies.

```bash
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

### Notebook Code

- Now lets import the required libraries and setup the sam model for inferencing.

```python
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ─── Import SAM3 components ─────────────────────────────────────────────
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ─── Device setup ───────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

- Also use cuda enabled compute instance for faster inferencing.
- Lets log into hugging face to download the model.

```python
from huggingface_hub import notebook_login

notebook_login()   # ← old version, still supported
```

- Let's load the sam3 model and processor.

```python
# ─── 1. Load the SAM3 image model ───────────────────────────────────────
print("Loading SAM3 image model...")
model = build_sam3_image_model()           # automatically loads latest checkpoint if available
model = model.to(device=device)
model.eval()

# ─── 2. Create processor ────────────────────────────────────────────────
processor = Sam3Processor(model)
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/sam3-notebook-1.jpg 'yolo11n Managed Online Endpoint Test Result')

- now lets load the image

```
# Option B: From URL (most convenient for testing)
from io import BytesIO
import requests

url = "https://ultralytics.com/images/bus.jpg"
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")

print(f"Image loaded: {image.size}")

# ─── 4. Prepare image for inference ─────────────────────────────────────
print("Preparing image...")
inference_state = processor.set_image(image)
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/sam3-notebook-2.jpg 'yolo11n Managed Online Endpoint Test Result')

- now lets create the mask using text prompt
- Then send to model and inference
- Finally visualize the results
- Output the masks and bounding boxes

```
# ─── 5. Run inference with text prompt ──────────────────────────────────
text_prompt = "person"          # ← change this to whatever you want: "bus", "person", "wheel", etc.

print(f"Running inference with prompt: '{text_prompt}'")
output = processor.set_text_prompt(
    state=inference_state,
    prompt=text_prompt
)

# ─── 6. Get results ─────────────────────────────────────────────────────
masks = output["masks"]      # list of binary masks (torch.Tensor)
boxes = output["boxes"]      # bounding boxes [N,4]
scores = output["scores"]    # confidence scores [N]

print(f"Found {len(masks)} objects with prompt '{text_prompt}'")

# ─── 7. Visualization ───────────────────────────────────────────────────
plt.figure(figsize=(12, 8))
plt.imshow(image)

# Draw each detected mask and box
for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
    # Convert mask to numpy
    mask_np = mask.cpu().numpy().squeeze()
    
    # Show mask contour
    plt.contour(mask_np, colors='red', alpha=0.7, linewidths=2)
    
    # Draw bounding box
    x1, y1, x2, y2 = box.cpu().numpy()
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                        fill=False, edgecolor='cyan', linewidth=2,
                        label=f"Score: {score:.3f}")
    plt.gca().add_patch(rect)
    
    # Label
    plt.text(x1, y1-10, f"{text_prompt} {score:.2f}",
             color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.6))

plt.title(f"SAM3 Text Prompt: '{text_prompt}'")
plt.axis('off')
plt.legend()
plt.tight_layout()
plt.show()

# Optional: save result
plt.savefig("sam3_result.jpg", dpi=300, bbox_inches='tight')
print("Result saved as sam3_result.jpg")
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/sam3-notebook-3.jpg 'yolo11n Managed Online Endpoint Test Result')

- Done! You have successfully run inferencing using SAM3 model in an Azure Machine Learning Notebook.