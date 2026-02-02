# Microsoft Foundry with Agent Framework - Single Environment CI/CD with Batch Execution

## Introduction

- This guide demonstrates how to set up a CI/CD pipeline for Microsoft Foundry using GitHub Actions for a single environment with batch execution.
- We are adding realtime evaluation and batch execution capabilities using agent framework with foundry sdk.
- We also show how Red Team testing can be integrated into the CI/CD pipeline to ensure the security and robustness of the deployed agents.
- This setup allows for automated deployment, testing, and validation of agents in a Foundry environment.
- The CI/CD pipeline will help streamline the development and deployment process, ensuring that agents are consistently tested and evaluated before being deployed to production.
- We can repeat the same with multiple environments by creating separate workflows for each environment.
- Deployment to multiple environments can be gated as per your requirements.
- This example is to show the flow and doesn't have any production ready code.

## Prerequisites

- Azure Subscription with Foundry deployed.
- We are using Standard Agent Setup for Enterprise best practices.
- GitHub repository with your Foundry deployment code.
- Create a service principal in Azure with the necessary permissions to deploy to Foundry.
- Make Sure to create a service principal with Contributor role on the Foundry resource group.
- Create service principal and assign role using Azure CLI:
  
  ```bash
  az ad sp create-for-rbac --name "<service-principal-name>" --role contributor --scopes /subscriptions/<subscription-id>/resourceGroups/<resource-group-name>
  ```

- Reset the secret for 15 days
- Code and Architecture available at: https://github.com/balakreshnan/msagentframework
  
```
az ad sp credential reset --name "<service-principal-name>" --end-date "2024-12-31T23:59:59Z"
```

- Now go to Entra ID in portal and search for App registrations, find your service principal and note down the following details:
  - Application (client) ID
  - Directory (tenant) ID
  - Client Secret (you will need to create one if you haven't already)
  - On the overview page click create service principal.

- Now assign Azure AI Developer, Azure AI User role to the service principal on the Foundry, foundry project resource group.
- Now assign Storage blob data contributor, Storage File Privileged contributor role to the service principal on the Foundry storage account.
- Above is needed to run the evals and red team agents.

## Setting up GitHub Secrets

- Now go to Github Repository -> Settings -> Secrets and variables -> Actions -> New Environment secret.
- Create Dev environment
- Now add these following secrets to the repository:
  - `AZURE_CLIENT_ID`: Your service principal's Application (client) ID.
  - `AZURE_TENANT_ID`: Your service principal's Directory (tenant) ID.
  - `AZURE_CLIENT_SECRET`: Your service principal's Client Secret.
  - `AZURE_SUBSCRIPTION_ID`: Your Azure Subscription ID.
  - `AZURE_AI_PROJECT`: The resource group where Foundry is deployed.
  - `AZURE_AI_PROJECT_ENDPOINT`: The name of your Foundry instance.
  - `AZURE_OPENAI_KEY`: Your Azure OpenAI API key.
  - `AZURE_OPENAI_ENDPOINT` : Your Azure OpenAI endpoint URL.
  - `AZURE_AI_MODEL_DEPLOYMENT_NAME` : The deployment name of your Azure OpenAI model.
  - `AZURE_OPENAI_DEPLOYMENT` : The deployment name of your Azure OpenAI model.
  - `AZURE_AI_SEARCH_INDEX_NAME` : The name of your Azure Cognitive Search index.
  - `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`: The deployment name for chat models.
  - `AZURE_OPENAI_API_VERSION`: The API version for Azure OpenAI.
  - `AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME`: The deployment name for response generation models.
  - `AZURE_RESOURCE_GROUP`: The resource group where Foundry is deployed.
  - `AGENT_NAME`: The name of the agent to run evaluations and scans.
- Now add credentials
  - AZURE_CREDENTIALS

```
{
  "clientId": "xxxxxxx",
  "clientSecret": "xxxxxxx",
  "subscriptionId": "sub id",
  "tenantId": "tenant id"
}
```

## GitHub Actions Workflow

- now we have the github setup with necessary secrets.
- Make sure your repository has the following files:
  - exagent.py : Script to execute the agent in realtime.
  - agenteval.py : Script to run evaluations on the agent in realtime.
  - batchevalagent.py : Script to run batch evaluations on the agent.
  - redteam.py : Script to perform red team testing on the agent.
  - requirements.txt : List of Python dependencies required for the scripts.
  - Also we need datarfp.jsonl file for agent batch evaluation input data.
  - Make sure the scripts are designed to accept command-line arguments for resource group, project, and agent name.
- Agent was created using code as well once you have the agent created you can use the same agent name in the workflow.
- Code below only uses an existing agent for evaluation and batch execution.
- Lets create a workflow file in .github/workflows/agent-consumption-single-env.yml

```yaml
name: Agent Consumption - Single Environment

on:
  workflow_dispatch:

jobs:
  consume-agent:
    name: Consume Existing Agent (Dev)
    runs-on: ubuntu-latest
    environment: dev

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Azure login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}

      - name: Run agent execution test
        env:
          AZURE_AI_PROJECT: ${{ secrets.AZURE_AI_PROJECT }}
          AZURE_AI_PROJECT_ENDPOINT: ${{ secrets.AZURE_AI_PROJECT_ENDPOINT }}
          AZURE_OPENAI_KEY: ${{ secrets.AZURE_OPENAI_KEY }}
          AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
          AZURE_AI_MODEL_DEPLOYMENT_NAME: ${{ secrets.AZURE_AI_MODEL_DEPLOYMENT_NAME }}
          AZURE_OPENAI_DEPLOYMENT: ${{ secrets.AZURE_OPENAI_DEPLOYMENT }}
          AZURE_AI_SEARCH_INDEX_NAME: ${{ secrets.AZURE_AI_SEARCH_INDEX_NAME }}
          AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: ${{ secrets.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME }}
          AZURE_OPENAI_API_VERSION: ${{ secrets.AZURE_OPENAI_API_VERSION }}
          AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME: ${{ secrets.AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME }}
        run: |
          python exagent.py \
            --resource-group "${{ secrets.AZURE_RESOURCE_GROUP }}" \
            --project "${{ secrets.AZURE_AI_PROJECT }}" \
            --agent-name "${{ secrets.AGENT_NAME }}"

      - name: Run evaluation
        env:
          AZURE_AI_PROJECT: ${{ secrets.AZURE_AI_PROJECT }}
          AZURE_AI_PROJECT_ENDPOINT: ${{ secrets.AZURE_AI_PROJECT_ENDPOINT }}
          AZURE_OPENAI_KEY: ${{ secrets.AZURE_OPENAI_KEY }}
          AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
          AZURE_AI_MODEL_DEPLOYMENT_NAME: ${{ secrets.AZURE_AI_MODEL_DEPLOYMENT_NAME }}
          AZURE_OPENAI_DEPLOYMENT: ${{ secrets.AZURE_OPENAI_DEPLOYMENT }}
          AZURE_AI_SEARCH_INDEX_NAME: ${{ secrets.AZURE_AI_SEARCH_INDEX_NAME }}
          AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: ${{ secrets.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME }}
          AZURE_OPENAI_API_VERSION: ${{ secrets.AZURE_OPENAI_API_VERSION }}
          AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME: ${{ secrets.AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME }}
        run: |
          python agenteval.py \
            --resource-group "${{ secrets.AZURE_RESOURCE_GROUP }}" \
            --project "${{ secrets.AZURE_AI_PROJECT }}" \
            --agent-name "${{ secrets.AGENT_NAME }}"
      
      - name: Run Batch evaluation
        env:
          AZURE_AI_PROJECT: ${{ secrets.AZURE_AI_PROJECT }}
          AZURE_AI_PROJECT_ENDPOINT: ${{ secrets.AZURE_AI_PROJECT_ENDPOINT }}
          AZURE_OPENAI_KEY: ${{ secrets.AZURE_OPENAI_KEY }}
          AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
          AZURE_AI_MODEL_DEPLOYMENT_NAME: ${{ secrets.AZURE_AI_MODEL_DEPLOYMENT_NAME }}
          AZURE_OPENAI_DEPLOYMENT: ${{ secrets.AZURE_OPENAI_DEPLOYMENT }}
          AZURE_AI_SEARCH_INDEX_NAME: ${{ secrets.AZURE_AI_SEARCH_INDEX_NAME }}
          AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: ${{ secrets.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME }}
          AZURE_OPENAI_API_VERSION: ${{ secrets.AZURE_OPENAI_API_VERSION }}
          AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME: ${{ secrets.AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME }}
        run: |
          python batchevalagent.py \
            --resource-group "${{ secrets.AZURE_RESOURCE_GROUP }}" \
            --project "${{ secrets.AZURE_AI_PROJECT }}" \
            --agent-name "${{ secrets.AGENT_NAME }}"

      - name: Run red team tests (optional)
        env:
          AZURE_AI_PROJECT: ${{ secrets.AZURE_AI_PROJECT }}
          AZURE_AI_PROJECT_ENDPOINT: ${{ secrets.AZURE_AI_PROJECT_ENDPOINT }}
          AZURE_OPENAI_KEY: ${{ secrets.AZURE_OPENAI_KEY }}
          AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
          AZURE_AI_MODEL_DEPLOYMENT_NAME: ${{ secrets.AZURE_AI_MODEL_DEPLOYMENT_NAME }}
          AZURE_OPENAI_DEPLOYMENT: ${{ secrets.AZURE_OPENAI_DEPLOYMENT }}
          AZURE_AI_SEARCH_INDEX_NAME: ${{ secrets.AZURE_AI_SEARCH_INDEX_NAME }}
          AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: ${{ secrets.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME }}
          AZURE_OPENAI_API_VERSION: ${{ secrets.AZURE_OPENAI_API_VERSION }}
          AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME: ${{ secrets.AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME }}
        run: |
          python redteam.py \
            --resource-group "${{ secrets.AZURE_RESOURCE_GROUP }}" \
            --project "${{ secrets.AZURE_AI_PROJECT }}" \
            --agent-name "${{ secrets.AGENT_NAME }}"
```

- Based on the code above there are 4 main steps after setting up the environment and installing dependencies:
  - Run agent execution test using exagent.py script.
  - Run evaluation using agenteval.py script.
  - Run batch evaluation using batchevalagent.py script.
  - Run red team tests using redteam.py script.
- Save the file and commit to the repository.
- Manully trigger the workflow from the Actions tab in your GitHub repository.
- Wait for it to complete and check the logs for any errors.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundrycicdwithbatch-1.jpg 'fine tuning model')

- while executing the batch evals you can see the progress in the portal as well.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundrycicdwithbatch-2.jpg 'fine tuning model')

- Once the workflow completes successfully, your agent has been evaluated and tested in the Foundry environment.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundrycicdwithbatch-3.jpg 'fine tuning model')

- Now lets look at the evaluation results in the portal.
- Go to Foundry portal -> Your Project -> Evaluations

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundrycicdwithbatch-4.jpg 'fine tuning model')

- Select the realtime evaluation to see the details.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundrycicdwithbatch-5.jpg 'fine tuning model')

- now select the batch evaluation to see the details.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundrycicdwithbatch-6.jpg 'fine tuning model')

- now go to Red Team Scans to see the details.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundrycicdwithbatch-7.jpg 'fine tuning model')

- now lets look at the observability of agent execution.
- Go to Foundry portal -> Agents -> Your Agent -> Traces

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundrycicdwithbatch-8.jpg 'fine tuning model')

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundrycicdwithbatch-9.jpg 'fine tuning model')

## Conclusion

- In this guide, we set up a CI/CD pipeline using GitHub Actions to deploy and evaluate agents in a Microsoft Foundry environment.
- We integrated realtime evaluation, batch execution, and red team testing into the pipeline to ensure the robustness and security of the deployed agents.
- This setup helps streamline the development and deployment process, ensuring that agents are consistently tested and validated before being deployed to production.
- You can further enhance this pipeline by adding more stages, such as automated testing, notifications, and deployment to multiple environments based on your requirements.