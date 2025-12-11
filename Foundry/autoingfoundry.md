# Microsoft Foundry Architecture with Automatic Knowledge Ingestion

## Introduction

- Build a agentic AI solutions using Microsoft Foundry with automatic knowledge ingestion.
- Context driven agentic RAG solutions.
- Creating the context knowledge base from multiple data sources.
- Create a pipeline to automate the knowledge ingestion process.
- Ability to insert, update and delete knowledge base content.
- Enabling through Foundry IQ.

## Prerequisites

- Azure Subscription
- Microsoft Foundry Account
- Cosmodb Account
- Azure Storage Blob Storage Account
- Application insights
- Azure Monitor
- AI Gateway
- Azure AI Search Service

## Architecture Overview

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/autoingestarch1.jpg 'fine tuning model')

### Architecture Details

- Ability to upload docs into Azure Storage Blob Storage.
- Using Blob storage connector to Azure AI Search to create the knowledge base.
- Above connectors has ability to index, update and delete the content in the knowledge base.
- Then use the AI search inside Foundry IQ to create the context for the agentic AI solution.
- Foundry also has AI gateway to manage token limits within model deployments.
- Foundry has ability to deploy multiple models and manage them through AI gateway.
- Model catalog to manage multiple models.
- we can use GUI to create the Agentic AI solutions inside Foundry.
- Pro Code option is also available.
- Tracing is available through Application insights and Azure Monitor.
- Tracing is very good to debug agents execution.
- Ability to create identity to agents to access other data sources securely.
- Options to connect to remote and local MCP using tools.
- Ability to use A2A agents to connect to external agents.
- Agent evaluation to test and improve the agents.
- Redteam agents to test the agents for vulnerabilities.