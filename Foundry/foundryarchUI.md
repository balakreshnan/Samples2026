# Microsoft Foundry Architecture - Agentic AI Applications

## Overview

- Build an entire agentic ai application architecture using Microsoft Foundry.
- Ability to bring in knowledge from multiple data sources.
- Even bring fabric and azure databricks sources.
- Also bring unstructured data from blob storage and using Azure AI search to create knowledge base.
- Create agentic AI applications using Foundry IQ.
- Ability to manage multiple models using AI gateway.
- Tracing using Application insights and Azure Monitor.
- Ability to create secure identity to agents to access data sources.
- Connect to remote and local MCP using tools.
- Use A2A agents to connect to external agents.
- Agent evaluation to test and improve the agents.
- Redteam agents to test the agents for vulnerabilities.
- Also ability to publish the agents in Agents 365.
- Once in Agents 365, can use the agents in Teams and other MS products.
- Ability to use the agents in Copilot UI and Copilot studio to build agentic AI solutions in copilot studio.
- Build enterprise grade agentic AI solutions using Microsoft Foundry.

## Architecture Diagram

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundryarch-1.jpg 'fine tuning model')

## Architecture Details

- Idea here is to use Foundry as the main platform to build agentic AI applications.
- Able to push documents into Azure Blob storage and use Azure AI search to create knowledge base from unstructured data. Which can handle upserts and deletes also.
- Foundry IQ has options to connect to Fabric IQ and Azure Databricks to bring in structured data as knowledge base.
- Foundry has Foundry IQ to build agentic AI solutions using low code/no code approach.
- Foundry also has Pro code option to build complex agentic AI solutions.
- Foundry has ability to connect to multiple data sources including Fabric, Azure Databricks, Blob storage etc.
- Using Blob storage connector to Azure AI Search to create knowledge base from unstructured data.
- Foundry has AI gateway to manage multiple model deployments and token limits.
- Abiity to use model router to route the requests to different models based on the requirements.
- Also ability to deploy the agents as MCP or A2A agents.
- So we can integrate with other applications and services.
- Tracing is available through Application insights and Azure Monitor.
- Tracing is very good to debug agents execution.
- Ability to create identity to agents to access other data sources securely.
- Build agentic ai application using workflows in foundry and agents.
- Publish the agents in Agents 365.
- Also publish to Teams, outlook and Copilots UI.
- This shows build enterprise grade agentic AI solutions using Microsoft Foundry and consuming in multiple MS products.
- Bottom section is for the IT governance and security.
- We can use Azure DevOps and Github Actions for CI/CD.
- Use Azure Policy and Blueprints for governance.
- Use Azure Security center and Sentinel for security and threat management.
- Use Azure Key vault to manage secrets and keys.
- Use Azure AD for identity and access management.
- Use Microsoft Defender for Endpoint for endpoint security.
- Use Microsoft Purview for data governance and compliance.
- Use Microsoft Information Protection for data classification and protection.