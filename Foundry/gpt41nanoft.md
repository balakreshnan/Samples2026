# Azure Open AI GPT 4.1 Nano Fine Tuning - GUI

## Introduction

- Fine tuning GPT 4.1 Nano models using Azure Open AI Service.
- This guide provides step-by-step instructions to prepare your data, create a fine-tuning job, and deploy the fine-tuned model.
- Prepare the data set for the task you want to fine-tune the model on.
- This document is for agentic ai tasks.
- Dataset should be in jsonl format.
- My samples has about 350+ rows.
- Have a separate validation set.
- Data set in messages format, like role, content pairs.
- No Code required, all steps are through GUI.

## Prerequisites

- Azure Subscription
- Microsoft Foundry Account
- Cosmodb Account
- Azure Storage Blob Storage Account
- Application insights
- Azure Monitor
- AI Gateway
- Azure AI Search Service

## Steps to Fine Tune GPT 4.1 Nano

- Prepare your dataset in JSONL format with the required structure.
- Have the jsonl file ready in your local system.
- Go to https://ai.azure.com
- Navigate to the "Fine-tuning" section.
- Click Fine tune a model.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-1.jpg 'fine tuning model')

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-2.jpg 'fine tuning model')

- For customization select Supervised Fine-tuning.
- Training Type either global or Developer.
- Select the model as GPT-4.1-Nano.
- Select the training data source, upload the jsonl file.
- Optionally, upload a validation dataset.
- Provide a suffix for the fine-tuned model name.
- Leave the additional hyperparameters as default.
- Click on Submit to start the fine-tuning job.
- Wait for to complete the fine-tuning process.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-7.jpg 'fine tuning model')

- Check the status of the fine-tuning job in the Fine-tuning section.
- Check the metrics and logs for the fine-tuning job.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-8.jpg 'fine tuning model')

- Once the fine-tuning is complete, deploy the fine-tuned model.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-9.jpg 'fine tuning model')

- Check the checkpoints

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-10.jpg 'fine tuning model')

- Complete the deployment steps by selecting the appropriate options.
- Now select the model and deploy it to an endpoint.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-11.jpg 'fine tuning model')

- Status when creating

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-12.jpg 'fine tuning model')

- Once completed, you can use the fine-tuned model via the endpoint.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-13.jpg 'fine tuning model')

### Evaluation

- Create a ground truth dataset to evaluate the fine-tuned model.
- Go to Evaluation section in Foundry.
- Create a new evaluation.
- Select the fine-tuned model as the model to evaluate.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-14.jpg 'fine tuning model')

- Next upload the ground truth dataset.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-15.jpg 'fine tuning model')

- next Map the fields from the ground truth dataset to the evaluation fields.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-16.jpg 'fine tuning model')

- Summary of the evaluation. and submit.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-17.jpg 'fine tuning model')

- here is the status of the evaluation.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-18.jpg 'fine tuning model')

- Once completed, view the evaluation results.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-19.jpg 'fine tuning model')

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-20.jpg 'fine tuning model')

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/gpt41nano-ft-21.jpg 'fine tuning model')

## Conclusion

- You have successfully fine-tuned a GPT 4.1 Nano model using Azure Open AI Service.
- You can now use the fine-tuned model for your specific tasks.