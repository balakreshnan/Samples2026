# How to fine tune a model for Agentic AI Task planning

## Introduction

- A experiment to fine tune a model for Agentic AI Task planning
- Agentic AI task planning seems to be the most challenging aspect of Agentic AI development.
- For a given business process, how can we get the deterministic tasks to process without skipping any steps?
- Above question is what i am trying to answer in this experiment.
- This is only experimental.
- Using Qwen 2.5 instruct model

## Pre-requisites

- Azure Subscription
- Azure machine learning
- Azure Storage Blob
- Python Environment
- Create a virtual environment
- Create a sample data set for training and testing

## Creating Data set

- I created a python script to create sample data set for training and testing.
- Save those data into JSONL files
- IN this repo there is folder called ftdata, where the train and test files are available

## Fine Tuning Steps

- Log into Azure machine learning
- I am using a compute with GPU - Standard_NC24ads_A100_v4
- Create a python virtual environment
- Install all these python libraries

```
%pip install -U "transformers>=4.40" datasets accelerate peft bitsandbytes trl huggingface_hub
%pip install evaluate
%pip install rouge-score nltk absl-py
%pip install sacrebleu
%pip install tqdm
```

- Setup huggingface key and login into huggingface

```
hf_token = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
import os
from huggingface_hub import login

login(token=hf_token)
```

- Make sure upload the train.jsonl, test.jsonl into the same folder as this notebook.
- Now let's setup the data set information to train the model.

```
import os
import json
from dataclasses import dataclass

import torch
from datasets import load_dataset
from huggingface_hub import login

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
# , DataCollatorForCompletionOnlyLM


# -----------------------
# CONFIG YOU SHOULD EDIT
# -----------------------
MODEL_NAME = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
TRAIN_FILE = os.getenv("TRAIN_FILE", "train.jsonl")
TEST_FILE  = os.getenv("TEST_FILE",  "test.jsonl")
```

- Setup configuration parameters for training and dataset

```
# Set this to your target HF repo, e.g. "your-user/qwen-workflow-planner-lora"
HUB_MODEL_ID = os.getenv("HUB_MODEL_ID", "Balab2021/qwen-workflow-planner-qwen2p5-lora")

# If you want the repo private by default (requires paid/private permissions depending on account)
HUB_PRIVATE = os.getenv("HUB_PRIVATE", "false").lower() == "true"

# Training knobs
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs-qwen-workflow")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "2048"))
EPOCHS = float(os.getenv("EPOCHS", "3"))
LR = float(os.getenv("LR", "2e-4"))
BATCH = int(os.getenv("BATCH", "1"))
GRAD_ACC = int(os.getenv("GRAD_ACC", "8"))
LOG_STEPS = int(os.getenv("LOG_STEPS", "10"))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "100"))
EVAL_STEPS = int(os.getenv("EVAL_STEPS", "100"))

# QLoRA (4-bit) toggle
USE_4BIT = os.getenv("USE_4BIT", "true").lower() == "true"


def convert_conversations_to_messages(example):
    """
    Convert your dataset format:
      {"id": "...", "conversations": [{"from":"system","value":"..."}, ...]}
    into TRL-friendly format:
      {"messages": [{"role":"system","content":"..."}, ...]}
    """
    conv = example["conversations"]
    messages = [{"role": m["from"], "content": m["value"]} for m in conv]
    return {"messages": messages}
```

- Now lets' login into huggingface
- Load the local dataset we are using

```
hf_token = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
if hf_token:
    login(token=hf_token)  # uses env var token
else:
    print("WARNING: HF_TOKEN is not set. You can still train, but push_to_hub will fail.")

# -----------------------
# LOAD DATA
# -----------------------
train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
test_ds  = load_dataset("json", data_files=TEST_FILE,  split="train")
```

- Convert the schema for qwen model

```
# Convert schema
train_ds = train_ds.map(convert_conversations_to_messages, remove_columns=train_ds.column_names)
test_ds  = test_ds.map(convert_conversations_to_messages,  remove_columns=test_ds.column_names)
```

- Let's now tokensize the dataset

```
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Ensure pad token exists (common for decoder-only LMs)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

- Load model now with Lora config

```
quant_config = None
if USE_4BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto",
    quantization_config=quant_config,
)
```

- Lora Config

```
# -----------------------
# LoRA CONFIG
# -----------------------
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # target_modules left default; you can set explicit modules if you want
)
```

- Setup Training configuration

```
# -----------------------
# TRAINING CONFIG
# -----------------------
# TRL's SFTTrainer supports conversational datasets (messages) and applies chat template. 【1-f061c7】
sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    #max_seq_length=MAX_SEQ_LEN,
    logging_steps=LOG_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy="steps",
    save_strategy="steps",
    report_to="none",
    bf16=torch.cuda.is_available(),     # if supported on your GPU
    fp16=not torch.cuda.is_available(), # fallback
    push_to_hub=True,                   # enable hub upload
    hub_model_id=HUB_MODEL_ID,
    hub_private_repo=HUB_PRIVATE,
    disable_tqdm=True,   # ✅ FIX HER
)
```

- Setup the trainer

```
trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
    peft_config=peft_config
)
```

- Now let's disable tqdm so that training won't error

```
import os
from transformers.utils import logging

os.environ["DISABLE_TQDM"] = "true"
logging.disable_progress_bar()
disable_tqdm=True
os.environ["DISABLE_TQDM"] = "true"
```

- Now Start the training

```
# -----------------------
# TRAIN + EVAL
# -----------------------
trainer.train()
```

- IN progress

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/agentft-1.jpg 'Agent task fine tuning')

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/agentft-2.jpg 'Agent task fine tuning')

- Completed

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/agentft-3.jpg 'Agent task fine tuning')

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/agentft-4.jpg 'Agent task fine tuning')

- Now display the metrics
- Evaulate the model

```
# Optional final eval
metrics = trainer.evaluate()
print("Final eval metrics:", metrics)
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/agentft-5.jpg 'Agent task fine tuning')

- Now let's push the fine tune model to Huggingface hub

```
# -----------------------
# SAVE + PUSH
# -----------------------
# This will push the trained artifacts to HUB_MODEL_ID because push_to_hub=True
# TRL reference scripts describe pushing to Hub when push_to_hub is enabled. 【7-03ac67】
trainer.push_to_hub()

# Also push tokenizer explicitly (sometimes helpful for consumers)
tokenizer.push_to_hub(HUB_MODEL_ID)

print(f"✅ Done. Model pushed to: {HUB_MODEL_ID}")
```

- in progress

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/agentft-6.jpg 'Agent task fine tuning')

- After completion

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/agentft-8.jpg 'Agent task fine tuning')

- Now lets create a inferencing code to load the new model and inference

```
question = "Design an execution plan for: Plan the workflow for: Create and deploy a new Azure Web Application using agents."
OUTPUT_DIR = "./outputs-qwen-workflow"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_MODEL_PATH = "./outputs-qwen-workflow"

# -----------------------------
# LOAD TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# -----------------------------
# LOAD BASE MODEL
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
    device_map="auto"
)

# -----------------------------
# LOAD YOUR FINETUNED WEIGHTS
# -----------------------------
model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
model.eval()

# -----------------------------
# INFERENCE FUNCTION
# -----------------------------
def generate_workflow(user_prompt):
    messages = [
        {
            "role": "system",
            "content": "You are an expert cloud solution architect and workflow planner."
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # Apply Qwen chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate output
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.2,
        top_p=0.9,
        do_sample=True
    )

    # Decode output
    full_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only assistant response
    prompt_length = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
    response = full_output[prompt_length:]

    return response.strip()


# -----------------------------
# TEST IT
# -----------------------------
result = generate_workflow(
    "Plan the workflow for creating a CI/CD pipeline for a cloud application."
)

print("✅ GENERATED OUTPUT:\n")
print(result)
```

- run the above cell with questions
- wait for the response

```
GENERATED OUTPUT:

1. Authenticate with Azure
   - Run 'az login' or use service principal

2. Create Resource Group
   - az group create --name rg-prod --location eastus

3. Create App Service Plan
   - az appservice plan create --name asp-47 --resource-group rg-prod --sku S1

4. Create Web App
   - az webapp create --name app689 --resource-group rg-prod --plan asp-47

5. Configure Deployment
   - Setup deployment credentials or GitHub Actions
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/agentft-7.jpg 'Agent task fine tuning')

- Above output provides us the steps to take based on the question asked.
- Need to validate this with mulitple questions.
- same question second run

```
GENERATED OUTPUT:

1. Authenticate with Azure
   - Run 'az login' or use service principal

2. Create Resource Group
   - az group create --name rg-prod --location eastus

3. Create App Service Plan
   - az appservice plan create --name asp-50 --resource-group rg-prod --sku S1

4. Create Web App
   - az webapp create --name app678 --resource-group rg-prod --plan asp-50

5. Configure Deployment
   - Setup deployment credentials or GitHub Actions
```

- Here is the third run

```
GENERATED OUTPUT:

1. Authenticate with Azure
   - Run 'az login' or use service principal

2. Create Resource Group
   - az group create --name rg-prod --location eastus

3. Create App Service Plan
   - az appservice plan create --name asp-50 --resource-group rg-prod --sku S1

4. Create Web App
   - az webapp create --name app789 --resource-group rg-prod --plan asp-50

5. Configure Deployment
   - Setup deployment credentials or GitHub Actions
```

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/agentft-9.jpg 'Agent task fine tuning')

- More evals and testing.

## Gradio app to download the huggingface model and inference

- Idea here is to create chat UI for the model.
- Here is the python code
- Create a new project in visual studio code
- Create a virtual environment
- Install the required dependencies

```
```

- Here is the inferencing local python code.
- This is executing the model locally.

```
import os
from typing import List, Tuple

import gradio as gr
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_ID = "Balab2021/qwen-workflow-planner-qwen2p5-lora"
HF_TOKEN_KEYS = os.getenv("HF_TOKEN_KEYS")


def get_hf_token() -> str:
	if not HF_TOKEN_KEYS:
		raise RuntimeError(
			"Missing HF_TOKEN_KEYS environment variable. "
			"Set it to one or more token env var names (comma-separated), "
			"for example: HF_TOKEN_KEYS=HF_TOKEN"
		)

	raw_value = HF_TOKEN_KEYS.strip().strip("\"'")

	# Allow HF_TOKEN_KEYS to hold a direct Hugging Face token.
	if raw_value.startswith("hf_"):
		return raw_value

	keys = [key.strip() for key in raw_value.split(",") if key.strip()]

	if not keys:
		raise RuntimeError(
			"HF_TOKEN_KEYS is empty. "
			"Set it to one or more token env var names, for example: HF_TOKEN"
		)

	for key in keys:
		token = os.getenv(key)
		if token:
			return token.strip().strip("\"'")
	raise RuntimeError(
		"Missing Hugging Face token. None of the env vars listed in "
		f"HF_TOKEN_KEYS contain a token value. Checked keys: {', '.join(keys)}"
	)


def build_messages(history: List[Tuple[str, str]], user_message: str):
	messages = []
	for user_text, assistant_text in history:
		if user_text:
			messages.append({"role": "user", "content": user_text})
		if assistant_text:
			messages.append({"role": "assistant", "content": assistant_text})
	messages.append({"role": "user", "content": user_message})
	return messages


def create_app():
	load_dotenv()
	token = get_hf_token()

	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_ID,
		token=token,
		torch_dtype="auto",
		device_map="auto",
	)

	def chat_fn(
		message: str,
		history: List[Tuple[str, str]],
		temperature: float,
		max_new_tokens: int,
	) -> str:
		messages = build_messages(history, message)
		prompt = tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
		)

		inputs = tokenizer(prompt, return_tensors="pt")
		inputs = {k: v.to(model.device) for k, v in inputs.items()}

		with torch.no_grad():
			output_ids = model.generate(
				**inputs,
				max_new_tokens=max_new_tokens,
				temperature=temperature,
				do_sample=temperature > 0,
				pad_token_id=tokenizer.eos_token_id,
			)

		generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
		response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
		return response

	demo = gr.ChatInterface(
		fn=chat_fn,
		additional_inputs=[
			gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="Temperature"),
			gr.Slider(32, 2048, value=512, step=32, label="Max New Tokens"),
		],
		title="Qwen Workflow Planner Chat",
		description=f"Model: {MODEL_ID}",
	)
	return demo


if __name__ == "__main__":
	app = create_app()
	server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
	server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
	app.launch(server_name=server_name, server_port=server_port, inbrowser=True)
```

- Run the app
  - Open a terminal or command prompt
  - Navigate to the project directory
  - Activate the virtual environment
  - Run the Python script: `python app.py`

- Now test the app using http://localhost:7680/
- Provide the same question from Azure machine learning notebook inferencing.
- Compare the results with the Azure machine learning notebook inferencing.

![info](https://github.com/balakreshnan/Samples2026/blob/main/AML/images/agentft-10.jpg 'Agent task fine tuning')

- What surprised me is the quality of the responses generated by the locally running model, which were quite similar to those from the Azure machine learning notebook inferencing.
- wanted to make sure the locally running model was performing as expected.