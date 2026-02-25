# Physical AI Cosmos Reason2 2B World Model inference in Azure Machine Learning

## Introduction

- Idea here is to demonstrate how to run inference using the Physical AI Cosmos Reason2 2B World Model in Azure Machine Learning.
- Here is the model information - https://huggingface.co/nvidia/Cosmos-Reason2-2B
- We will be using the Hugging Face Transformers library to load the model and run inference on a sample input.
- Create a new python environment in Azure Machine Learning and install the necessary libraries, including transformers and torch.
- This is for physical AI cosmos reason2 2B model.

## Steps to run inference

- First create compute instance with GPU in Azure Machine Learning.
- SKU used - Standard_NC24ads_A100_v4 (24 cores, 220 GB RAM, 64 GB disk)
- Idea here is to simulate edge device inference using smaller GPU instance, but you can choose a larger instance if needed for faster inference.
- Now lets create conda environment and install necessary libraries.

````
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create --name cosmos3env -y python=3.12
conda activate cosmos3env
conda install pip -y
conda install ipykernel -y
python -m ipykernel install --user --name cosmos3env --display-name "cosmos3env"

pip install transformers torch accelerate bitsandbytes pillow torchvision av
```

- Now create a new Jupyter notebook in Azure Machine Learning and run the following code to load the model and run inference.
- Make sure hugging face api key ia available as environment variable in the notebook.
- Lets log into Hugging Face.

```
hf_token = """""
import os
from huggingface_hub import login

login(token=hf_token)
```

- Now time to write the inference code.
- make sure there is a sample mp4 file available in the same directory as the notebook for testing.
- import necessary libraries and load the model.

````
import transformers
import torch
```

- Now Download the model

```
model_name = "nvidia/Cosmos-Reason2-2B"
model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
)
processor: transformers.Qwen3VLProcessor = (
    transformers.AutoProcessor.from_pretrained(model_name)
)
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/cosmos-reason2-2B-1.jpg 'yolo11n Managed Online Endpoint Test Result')

- now prepare the message with the video input and run inference.

```
video_messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}],
    },
    {"role": "user", "content": [
            {
                "type": "video", 
                "video": "sample.mp4",
                "fps": 4,
            },
            {"type": "text", "text": (
                    "Is it safe to turn right? Answer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag."
                )
            },
        ]
    },
]
```

- Now run the inference and print the output.

```
inputs = processor.apply_chat_template(
    video_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    fps=4,
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=4096)
generated_ids_trimmed = [
    out_ids[len(in_ids) :]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
```

- now print the output text.

```
print(output_text)
```

- output

```
["<think>\nOkay, let's see. The user is asking if it's safe to turn right based on the video provided.\n\nFirst, I need to recall the details from the video. The scene is a quiet suburban neighborhood with a single-story house. The car is stopped at the intersection, facing a crosswalk. There are parked cars on both sides of the street. The driver is in a parked position, looking around, and there's no one else moving. The camera shows a wide view with power lines and trees.\n\nNow, the question is about turning right. The key factors here would be checking for oncoming traffic, pedestrians, and any obstacles. The video shows the car is stopped, so maybe it's waiting for a safe gap. Since there are parked cars on both sides, turning right might require checking for any vehicles that could be coming from the opposite direction. Also, there's a crosswalk, which suggests that pedestrians might be crossing, so the driver needs to be cautious. The absence of moving vehicles in the video implies that right now, it's safe, but the driver must ensure that no cars are approaching from the left (assuming the crosswalk is perpendicular) and that pedestrians aren't in the way. However, since the video doesn't show any traffic, maybe it's safe. But the presence of a crosswalk means that even if there's no immediate traffic, the driver should proceed with caution. So the answer would depend on whether there's a safe path to turn right without conflicting traffic or pedestrians. Since the video doesn't show any oncoming traffic or pedestrians, and the car is stopped, it might be safe to turn right now. But the driver should still check for any potential hazards before proceeding.\n</think>\n\nBased on the video, it is **not safe to turn right immediately**. The car is stopped at the intersection facing a crosswalk, and while there are no visible moving vehicles or pedestrians in the frame, the presence of a crosswalk and parked cars on both sides of the street suggests potential hazards. Turning right could risk colliding with oncoming traffic (if approaching from the left) or pedestrians crossing the crosswalk. The driver must first ensure the intersection is clear before proceeding."]
```

- You can see the output both thinking and also the output
- here is the actual output from the model.

```
Based on the video, it is **not safe to turn right immediately**. The car is stopped at the intersection facing a crosswalk, and while there are no visible moving vehicles or pedestrians in the frame, the presence of a crosswalk and parked cars on both sides of the street suggests potential hazards. Turning right could risk colliding with oncoming traffic (if approaching from the left) or pedestrians crossing the crosswalk. The driver must first ensure the intersection is clear before proceeding.
```

- done

## Conclusion

- This demonstrates how to run inference using the Physical AI Cosmos Reason2 2B World Model in Azure Machine Learning. By following these steps, you can load the model, prepare your input data, and obtain insights based on the video content. This model can be used for various applications, such as analyzing traffic scenarios, providing safety recommendations, and more.
- The key takeaway is that with the right setup and environment, you can leverage powerful AI models
to gain insights from complex data inputs, such as videos, and make informed decisions based on those insights.
- Next to build actual use cases around this model, such as in autonomous driving, traffic safety analysis, and other scenarios where understanding the environment through video is crucial.