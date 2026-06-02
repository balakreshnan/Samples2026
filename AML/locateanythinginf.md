# NVIDIA Locate Anything Inference in Azure Machine Learning

## Introduction

- This document provides an overview of how to use NVIDIA's Locate Anything model for inference within the Azure Machine Learning environment.
- Model size is 3B parameters.
- I am using it from https://huggingface.co/nvidia/LocateAnything-3B
- i created a new environment
- Need cuda 12.2 and above to function properly.
- If the operatiing system has older version please update
- Create a new python virtual environment
- i am using python 3.12. 

```
Note: numpy 1.26.4 is required. anything lower version will error out in installation
```

```
conda create -n locany3b python=3.10 -y
conda activate locany3b
pip install ipykernel
python -m ipykernel install --user --name locany3b --display-name "Python (locany3b)"

pip install opencv-python-headless==4.11.0.86 transformers==4.57.1 numpy==1.26.4 Pillow==11.1.0 peft torchvision decord==0.6.0 lmdb==1.7.5
```

```
nvidia-smi
sudo apt update
# For most modern GPUs (RTX 30/40/50 series, A100, H100, L40 etc.)
sudo apt install nvidia-driver-550   # or 560 / 570 / 580
sudo reboot
nvidia-smi
```

- now go to environment created

```
conda activate locany3b

# Best option for most users (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

- to clean up huggingface cache

```
ls -lh ~/.cache/huggingface/hub

rm -rf ~/.cache/huggingface/hub
```

## Code to Inference

- idea here is to use locateanthing - small vision language model to inference with image
- Downloaded a sample image from google, public image
- Create the python virtual environment and activate it
- Install the required dependencies in the virtual environment
- here is the class to use the model and parse
- Model inference code is available: https://huggingface.co/nvidia/LocateAnything-3B
- I am only trying to adapt it for Azure machine learning notebook using compute instance.
- using SKU as Standard_NC24ads_A100_v4
- has 24 vCPUs and 160 GB RAM and 2 A100 GPUs.

```
import re
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class LocateAnythingWorker:
    """Stateful worker that loads the model once and serves perception queries."""

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",   # "fast" (MTP) | "slow" (NTP/AR) | "hybrid"
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> dict:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )

        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    # ---- Convenience methods for each task ----

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        """Object detection / document layout analysis."""
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — single instance."""
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — multiple instances."""
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_text(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Text grounding."""
        prompt = f"Please locate the text referred as {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_text(self, image: Image.Image, **kwargs) -> dict:
        """Scene text detection."""
        prompt = "Detect all the text in box format."
        return self.predict(image, prompt, **kwargs)

    def ground_gui(self, image: Image.Image, phrase: str, output_type: str = "box", **kwargs) -> dict:
        """GUI grounding (box or point)."""
        if output_type == "point":
            prompt = f"Point to: {phrase}."
        else:
            prompt = f"Locate the region that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Pointing."""
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    # ---- Utility: parse model output ----

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate bounding boxes.

        Coordinates in model output are normalized integers in [0, 1000].
        """
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({
                "x1": x1 / 1000 * image_width,
                "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width,
                "y2": y2 / 1000 * image_height,
            })
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate points."""
        points = []
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({
                "x": x / 1000 * image_width,
                "y": y / 1000 * image_height,
            })
        return points
```

- Now we are going to load the model and image

```
worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
# img = Image.open("example.jpg").convert("RGB")
img = Image.open("people1.jpg").convert("RGB")
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/locanything-3.jpg 'Agent task fine tuning')

- here is the actual input image

`![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/locanything-1.jpg 'Agent task fine tuning')

- Now parse the output and display the results

```
import re
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

img = Image.open("people1.jpg").convert("RGB")

w, h = img.size

result = worker.detect(
    img,
    ["person", "car", "bicycle", "bus", "vehicle", "building", "billboard", "board", "window"]
)

answer = result["answer"]

fig, ax = plt.subplots(figsize=(16,10))
ax.imshow(img)

current_label = None

tokens = re.findall(
    r'<ref>(.*?)</ref>|<box>(.*?)</box>',
    answer
)

for ref_text, box_text in tokens:

    if ref_text:
        current_label = ref_text

    elif box_text:

        coords = re.findall(r'<(\d+)>', box_text)

        if len(coords) != 4:
            continue

        x1, y1, x2, y2 = map(int, coords)

        # scale from 0-1000 to image size
        x1 = x1 * w / 1000
        x2 = x2 * w / 1000
        y1 = y1 * h / 1000
        y2 = y2 * h / 1000

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )

        ax.add_patch(rect)

        ax.text(
            x1,
            max(5, y1-5),
            current_label,
            color='white',
            bbox=dict(facecolor='red', alpha=0.7)
        )

plt.axis("off")
plt.show()
```

- here is the output

`![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/locanything-2.jpg 'Agent task fine tuning')

## Conclusion

The results showcase the model's strong inferencing and visual grounding capabilities. Despite the complexity of the scene and the large number of objects present, the model is able to accurately detect, localize, and reason about relevant elements within the image. Its performance in crowded environments is particularly noteworthy, demonstrating the growing maturity of multimodal AI systems for real-world image understanding. This remains an experimental initiative focused on exploring the capabilities and future potential of the technology.