# Microsoft Foundry Agent using Model Router

## Introduction

- To show case how to use Microsoft Foundry Agent with Model Router
- Model router is the Model from model catalog which has many models and can route requests to the appropriate model based on the input.
- I am using the latest deployment of the model router as of writing the tutorial.
- We are also showing how to integrate with real world example like RFP for construction.
- In this example i am using sample documents from old RFP available in public domain.
- I am using Foundry IQ is to use in the model router enabled agent.

## Prerequisites

- Azure subscription
- Microsoft foundry resource
- Storage and cosmos db for foundry
- Azure AI developer permission to access the foundry.
- Azure AI Search

## Demo Video 

<a href="https://youtu.be/uoP_dipSebM" target="_blank">Open video in new window</a>

### Optimizing AI Performance and Cost: A Deep Dive into Microsoft Foundry Agents and Model Routers

In the rapidly evolving world of artificial intelligence, developers often face a difficult trade-off: do you use a powerful, expensive model for every query to ensure quality, or a smaller, cheaper model to save on costs? A recent technical demonstration explores a sophisticated solution to this dilemma using Microsoft Foundry Agents and a Model Router

### The Core Concept: Intelligent Model Selection

The demonstration highlights a Foundry Agent that isn't tied to a single static model. Instead, it is connected to a Model Router—a service designed to act as an intermediary that "picks and chooses" different models based on the complexity of the user's prompt. The ultimate goal is to optimize both cost and performance automatically

For example, when a user asks a complex, detailed question, the router might select a high-performance model like the latest GPT-4o (referred to in the demo as GPDI). Conversely, for simpler queries—such as basic questions about fiber optics—the router identifies that a heavy-duty model is unnecessary and switches to GPT-4o-mini to reduce token usage and cost

### Advanced Retrieval with Foundry IQ

To ensure the agent provides accurate, grounded responses, the system utilizes Foundry IQ, which the presenter describes as agentic or context-aware RAG (Retrieval-Augmented Generation).

#### The workflow follows a precise path:

1. Search Indexing: Data is indexed using Azure AI Search. In this demo, the source material consists of open-source datasets and RFP (Request for Proposal) documents.
2. Contextual Pull: When a question is asked, Foundry IQ pulls the top five most relevant pieces of content to provide the most accurate results.
3. Source Verification: The system identifies the specific PDF files it is drawing information from, ensuring the user can validate that the response is coming from the right location.

#### A Transparent User Interface

The demo is presented through a Streamlit application designed for both interaction and technical validation. The UI is split into two sections:

Left Side: Displays the standard conversation history between the user and the agent.
Right Side: Provides deep technical insights, including which model was selected, token usage, the parameters used (like temperature and frequency penalty), and the response duration.

This transparency allows developers to test and validate that the Model Router is making sensible choices in real-time.

#### Scaling for the Real World

While the demo currently stores conversation history in runtime for simplicity, the presenter notes that a production-level application would move this to long-term storage. Suggested back-end solutions include Cosmos DB or Redis cache for session history, or a Postgres server to maintain detailed usage metadata.

By combining Foundry IQ’s precise retrieval capabilities with the Model Router’s cost-saving logic, developers can create AI applications that are both highly intelligent and economically sustainable.

## Conclusion

- Demo video demonstrates how agent was created, then a streamlit UI created to show how the end user will interact with the agent.
- Show case the model router switching model based on complexity of the question by itself without using any custom logic.