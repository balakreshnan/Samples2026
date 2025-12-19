# Azure Machine Learning - Inferencing Fara 7B model

## Prerequisites

- Azure subscription with Azure Machine Learning workspace.
- Fara 7B model files and tokenizer files.
- Python Environment with necessary libraries installed (e.g., transformers, torch, azureml-sdk).

```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create --name gpuenv -y python=3.12
conda activate gpuenv 
conda install pip -y
conda install ipykernel -y
python -m ipykernel install --user --name gpuenv --display-name "gpuenv"


pip install transformers torch accelerate bitsandbytes pillow torchvision
```

## Steps to deploy and infer Fara 7B model

- Now write the code

```
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import requests
```

- Load the model and processor

```
# Load the model and processor
model_id = "microsoft/Fara-7B"
```

- Load the model

```
# For lower VRAM: Use 4-bit quantization (requires bitsandbytes)
# device_map = "auto"
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,  # Or torch.float16
    device_map="auto",           # Automatically uses GPU(s)
    trust_remote_code=True,
    # quantization_config=quantization_config,  # Uncomment for 4-bit
)
```

- Now submit a text prompt

```
# Example 1: Text-only inference
messages_text_only = [
    {"role": "user", "content": "Hello! Tell me about yourself."}
]

prompt_text = processor.apply_chat_template(messages_text_only, add_generation_prompt=True)
inputs_text = processor(text=prompt_text, return_tensors="pt").to(model.device)

generated_ids_text = model.generate(
    **inputs_text,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

response_text = processor.batch_decode(generated_ids_text, skip_special_tokens=True)[0]
print("Text-only response:")
print(response_text)
```

- output will be similar to below

```
Text-only response:
system
You are a helpful assistant.
user
Hello! Tell me about yourself.
assistant
Hi there, I'm Fara, a model developed by Microsoft Research AI Frontiers to assist users with web browsing tasks. Fara is fine-tuned from the Qwen2.5-VL model and can help you find information online, summarize pages, and more.
<tool_call>
{"name": "Assistant", "arguments": {"action": "terminate", "status": "success"}}
</tool_call>
```

- now with image and text

```
from PIL import Image
import requests
from IPython.display import display  # To display the image in the notebook

# # New working image: Flight booking search form screenshot
# image_url = "https://developers.amadeus.com/PAS-EAS/api/v1/cms-gateway/sites/default/files/inline-images/Boostrap_flight_booking_form_search_results.png"

# image = Image.open(requests.get(image_url, stream=True).raw)

# display(image)  # Displays in notebook for your reference
# Load and display the image
image_url = "https://www.travelpayouts.com/blog/wp-content/uploads/2017/05/tp-blog-1864x980-1-1.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
display(image)
```

- now prompt with image and text

```
# Prepare the multimodal prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},  # This tells the processor to include the image
            {"type": "text", "text": "Describe this screenshot in detail and suggest the next steps to book a flight from New York to London next month."}
        ]
    }
]

# Apply chat template and process inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

# Generate response
generated_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

# Decode and print the model's response
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Fara-7B Response:")
print(response)
```

- output will be similar to below

```
Fara-7B Response:
system
You are a helpful assistant.
user
Describe this screenshot in detail and suggest the next steps to book a flight from New York to London next month.
assistant
The screenshot is of an online flight-search interface with a bright, playful illustration in the background showing someone holding a toy airplane surrounded by money symbols. In the foreground is a white “Search Widget” panel containing:

• Two text fields labeled “From where?” (for origin) and “To where?” (for destination)  
• A “Depart” date picker field with a calendar icon  
• A “Return” date picker field (currently empty)  
• A “Passenger” dropdown selector (default set to one adult)  
• A large green “Search Flights” button at the bottom

Next steps to book a round-trip from New York to London for next month:

1. Fill in “From where?” with your departure airport (e.g., JFK or EWR).  
2. Enter “To where?” as your arrival (e.g., LHR or HEA).  
3. Open the “Depart” calendar and choose the date for your outbound flight (the first available date in the next month).  
4. If you’re not yet sure about the return, leave the “Return” field blank or pick the earliest convenient date.  
5. Set the passenger count to two adults (or adjust as needed).  
6. Click the green “Search Flights” button to view available options.  
7. Browse the results, apply any filters (price, airline, times), and select your preferred flights.  
8. Proceed to seat selection and payment to complete the booking.
<tool_call>
{"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart", "arguments": {"name": "Mozart",
```

- Done