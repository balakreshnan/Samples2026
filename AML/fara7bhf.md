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
The image is a clean, modern “Search Widget” for booking flights. Key elements include:

• Background: A bright purple-blue gradient with abstract shapes, floating green money-bills and coins, and a stylized illustration of a person in a yellow jacket holding a toy airplane.  
• Main Card: White, rounded-corner box centered on the page.  
• Title: “Search Widget” in bold, dark text at the top.  
• Input Fields:  
  – “From where?” (text field)  
  – “To where?” (text field)  
  – Depart (calendar icon)  
  – Return (calendar icon)  
  – Passenger (dropdown arrow)  
• Action Button: A wide, lime-green button labeled “Search Flights.”  

Next Steps to Book a Flight from New York to London Next Month

1. Enter your origin in the “From where?” field—type “New York” and choose the correct airport (likely JFK or EWR) by selecting from the auto-suggest list.
2. Type “London” into the “To where?” field and pick your destination (Heathrow, Gatwick or Stansted) from the suggestions.
3. Click the Depart calendar button to open the date picker.  
4. Use the calendar to select the outbound date for next month (e.g., June 15).  
5. Repeat the calendar step for the return date—choose the date you want to fly back, typically one week or two weeks after the outbound.
6. Open the Passenger dropdown and set it to “2 adults, 0 children, 0 infants.”
7. Hit the green “Search Flights” button to load available flight options.
8. Review the results—filter by airline, price, duration, stopover length or departure/arrival times.
9. Select a suitable outbound and return flight, review any baggage fees or change/cancellation policies.
10. Proceed to passenger details and payment to complete the booking.

If you’d like me to walk through specific airport codes or suggested airlines, just let me know!
<tool_call>
{"name": "Morerio", "arguments": {"action": "terminate", "status": "success"}}
</tool_call>
```

- one more example with different image

```
from PIL import Image
import requests
from IPython.display import display  # To display the image in the notebook

# # New working image: Flight booking search form screenshot
# image_url = "https://developers.amadeus.com/PAS-EAS/api/v1/cms-gateway/sites/default/files/inline-images/Boostrap_flight_booking_form_search_results.png"

# image = Image.open(requests.get(image_url, stream=True).raw)

# display(image)  # Displays in notebook for your reference
# Load and display the image
image_url = "https://skift.com/wp-content/uploads/2017/11/Screen-Shot-2017-11-17-at-2.37.43-PM.png"
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

- here is the output you will get

```
Fara-7B Response:
system
You are a helpful assistant.
user
Describe this screenshot in detail and suggest the next steps to book a flight from New York to London next month.
assistant
This is a screenshot of Google Flights in its default “round-trip” view, with the following elements:

1. Top bar  
   • Google Flights logo (top left)  
   • Trip type selector set to “Round trip”  
   • Passenger count: “1 passenger”  
   • Cabin class: “Economy”  

2. Main search form (centered below the top bar)  
   • Departure city: “New York City”  
   • Destination field: “Where to?”  
   • Date pickers showing “Sun, Dec 3” (departure) and “Thu, Dec 7” (return)  

3. Recent destination and tracking widget  
   • A note that your recent destination was London  
   • A “Tracked prices” card showing “New York City to Milan – Cheapest flight – Round trip” at 
1,748), with a “VIEW ALL” link to see more  

4. Popular destinations list  
   • Toronto (Nonstop, 1 h 30 m, 
214)  
   • Philadelphia (Nonstop, 1 h 4 m, $473)  
   • London (not shown in the visible excerpt but implied by the destination field)  

5. Embedded map on the right showing a zoomed-in view of New York and surrounding areas  

Next steps for booking a round-trip from New York to London next month:  
• Click the “Where to?” box to enter “London” as your destination.  
• Adjust your travel dates if needed—Google Flights will prompt you to pick departure and return days.  
• Leave the trip type set to Round trip, keep 1 passenger in Economy, and hit Search.  
• Once results load, you can sort by price or duration, select the nonstop option if available, and click through to book on your preferred airline’s site.
<tool_call>
{"name": "computer", "arguments": {"action": "terminate", "status": "success"}}
</tool_call>
```

- Done