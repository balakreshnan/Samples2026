# Github CI/CD for Foundry for Single environment

## Overview

- This document outlines the steps to set up a Continuous Integration/Continuous Deployment (CI/CD) pipeline for Foundry using GitHub Actions for a single environment.
- Pipeline will use existing agent, run agent evaluation and Red Team scans.
- Pipeline will deploy to a single Foundry environment.
- Showing development environment as example.
- Assumes you have a working knowledge of GitHub Actions and Foundry deployment processes.
- This guide assumes you have already set up your Foundry environment and have the necessary permissions to deploy.

## Prerequisites

- Azure Subscription with Foundry deployed.
- GitHub repository with your Foundry deployment code.
- Create a service principal in Azure with the necessary permissions to deploy to Foundry.
- Make Sure to create a service principal with Contributor role on the Foundry resource group.
- Create service principal and assign role using Azure CLI:
  
  ```bash
  az ad sp create-for-rbac --name "<service-principal-name>" --role contributor --scopes /subscriptions/<subscription-id>/resourceGroups/<resource-group-name>
  ```

- REset the secret for 15 days
  
```
az ad sp credential reset --name "<service-principal-name>" --end-date "2024-12-31T23:59:59Z"
```

- Now go to Entra ID in portal and search for App registrations, find your service principal and note down the following details:
  - Application (client) ID
  - Directory (tenant) ID
  - Client Secret (you will need to create one if you haven't already)
  - On the overview page click create service principal.

- Now assign Azure AI Developer role to the service principal on the Foundry, foundry project resource group.
- Now assign Storage blob data contributor, Storate File Privileged contributor role to the service principal on the Foundry storage account.
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

- Now create a folder in the root of your repository named `.github/workflows`.
- create a new file agent-consumption-single-env.yml
- Add the following code to the file:

```

name: Agent Consumption - Single Environment

on:
  workflow_dispatch:
  push:
    branches:
      - main

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

- Commit and push the changes to your GitHub repository.
- To manually trigger the workflow, go to the "Actions" tab in your GitHub repository, select the "Agent Consumption - Single Environment" workflow, and click on the "Run workflow" button.
- Monitor the workflow run to ensure that the agent execution, evaluation, and red team scans complete successfully.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundrycicdgithub-1.jpg 'fine tuning model')

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundrycicdgithub-2.jpg 'fine tuning model')

## Conclusion

- You have successfully set up a CI/CD pipeline for Foundry using GitHub Actions for a single environment.
- Only for single environment deployments.
- You can now extend this pipeline to include additional steps or environments as needed.