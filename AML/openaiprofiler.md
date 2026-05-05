# OpenAI Profiler inference in Azure Machine learning compute instance

## Prerequisites

- Azure Subscription
- Azure Machine Learning workspace
- Compute instance in Azure Machine Learning
- Huggingface Account and Key
- Python Environment
- Need GPU compute: Standard_NC24ads_A100_v4

## Code Steps

- Create a python environment 3.13
- Install required packages: `pip install -U git+https://github.com/huggingface/transformers.git`
- Set up Huggingface API key in environment variables: `export HUGGINGFACE_API_KEY=<your_huggingface_api_key>`
- Verify that the API key is set correctly: `echo $HUGGINGFACE_API_KEY`

- if you want to authenticate to huggingface in notebook then

```
from huggingface_hub import login

# Replace "YOUR_HF_TOKEN" with your actual Hugging Face token.
# For security, consider storing your token as a Colab secret.
# You can access Colab secrets via: from google.colab import userdata; userdata.get('HF_TOKEN')
login(token="HF_TOKEN")
```

- Set the pipeline with model to use

```
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("token-classification", model="openai/privacy-filter")
```

- download the model

```
# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("openai/privacy-filter")
model = AutoModelForTokenClassification.from_pretrained("openai/privacy-filter")
```

- Now test the model with input

```
sample_text = "My email is john.doe@example.com and my phone number is 123-456-7890 and social security is 123-45-6789."
results = pipe(sample_text)
print(results)
```

- Let's parse the output and see the masked text

```
final_entities = []
temp_entity = {}

# Process results to group B, I, E tags into full entities
for i, item in enumerate(results):
    # Split entity tag (B, I, E) from entity type (e.g., private_email)
    entity_tag, entity_type = item['entity'].split('-', 1) if '-' in item['entity'] else ('S', item['entity'])
    start = item['start']
    end = item['end']
    word = sample_text[start:end] # Extract the actual text segment

    if entity_tag == 'B' or entity_tag == 'S': # Beginning of a new entity or a single token entity
        if temp_entity: # If there was a previous entity being built, add it to final_entities
            final_entities.append(temp_entity)
        # Start a new temporary entity
        temp_entity = {
            'type': entity_type,
            'start': start,
            'end': end,
            'text': word
        }
    elif entity_tag == 'I': # Inside an existing entity
        if temp_entity and temp_entity['type'] == entity_type: # Continue the current entity
            temp_entity['end'] = end
            temp_entity['text'] += word
        else: # Malformed sequence (I- tag without matching B- or different type), treat as new B-
            if temp_entity:
                final_entities.append(temp_entity)
            temp_entity = {
                'type': entity_type,
                'start': start,
                'end': end,
                'text': word
            }
    elif entity_tag == 'E': # End of an entity
        if temp_entity and temp_entity['type'] == entity_type: # End the current entity
            temp_entity['end'] = end
            temp_entity['text'] += word
            final_entities.append(temp_entity)
            temp_entity = {} # Reset for the next entity
        else: # Malformed sequence (E- tag without matching B- or I-), treat as standalone
            if temp_entity:
                final_entities.append(temp_entity)
            final_entities.append({
                'type': entity_type,
                'start': start,
                'end': end,
                'text': word
            })
            temp_entity = {}

if temp_entity: # Add any remaining entity that was being built after the loop finishes
    final_entities.append(temp_entity)

print("Detected Entities:")
for entity in final_entities:
    # Format entity type for better readability (e.g., 'private_email' -> 'Private Email')
    print(f"- Type: {entity['type'].replace('_', ' ').title()}, Text: '{entity['text']}', Span: ({entity['start']}, {entity['end']})")

print("\nOriginal Text:")
print(sample_text)

# Create a masked version of the text by replacing entities with placeholders
masked_parts = []
last_end = 0
for entity in final_entities:
    # Add the text segment before the current entity
    masked_parts.append(sample_text[last_end:entity['start']])
    # Add the placeholder for the detected entity
    masked_parts.append(f"[{entity['type'].upper()}]")
    last_end = entity['end']

# Add any remaining text after the last entity
masked_parts.append(sample_text[last_end:])

print("\nText with Entities Masked:")
print("".join(masked_parts))
```

- Here is the output

```
Detected Entities:
- Type: Private Email, Text: ' john.doe@example.com and', Span: (11, 36)
- Type: Private Phone, Text: '123-456-7890', Span: (56, 68)
- Type: Account Number, Text: '123-45-6789', Span: (92, 103)

Original Text:
My email is john.doe@example.com and my phone number is 123-456-7890 and social security is 123-45-6789.

Text with Entities Masked:
My email is[PRIVATE_EMAIL] my phone number is [PRIVATE_PHONE] and social security is [ACCOUNT_NUMBER].
```

- So you can see the Masked text.
- Done