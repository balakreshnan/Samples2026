# Microsoft Foundry - Building Microsoft IQ using Work IQ, Fabric IQ, Foundry IQ

## Introduction

- Idea here is to show case Microsoft IQ as a foundry agent
- Using WorkIQ, Fabric IQ and Foundry IQ
- Work IQ for teams, email, calendar, and tasks and other copilot logs
- Fabric IQ for Order management data for retail scenario
- Foundry IQ to show case construction RFP data

## Video Demo

<a href="https://youtu.be/Y3j3aePsfrU" target="_blank">Open video in new window</a>

<iframe width="560" height="315" src="https://www.youtube.com/embed/Y3j3aePsfrU" frameborder="0" allowfullscreen></iframe>

### Building Unified Intelligence: A Guide to Microsoft IQ with Foundry Agents

In the evolving landscape of AI, the ability to synthesize information from disparate sources—like your emails, corporate databases, and private document repositories—is a game-changer. A recent demonstration highlights a powerful implementation of this concept: Microsoft IQ built with Foundry Agents. This integration allows users to create a single, intelligent agent capable of navigating diverse data environments to provide comprehensive answers

### Understanding the Three Pillars of Microsoft IQ

The core of this system lies in its ability to connect to three distinct "IQs," each representing a different data domain:

#### Work IQ

This layer encompasses your day-to-day productivity data, including emails, Teams interactions, and calendar events.

#### Fabric IQ

This functions as a data agent for structured information. In the demonstration, this included retail sales data exposed through an ontology, allowing the agent to query specific business metrics.

#### Foundry IQ

Powered by agentic RAG (Retrieval-Augmented Generation), this IQ handles unstructured data. It uses AI search and embeddings to process and retrieve information from private document collections, such as public RFPs

### How to Build the Agent

Building this unified agent within the Foundry environment involves a few critical steps

#### Model Selection & Prompting

The demo utilized the 5.4 model. Significant focus is placed on crafting meta-prompts that instruct the agent to prioritize information from the connected tools rather than general web knowledge

#### Integrating IQs

Within the Foundry interface, developers can select the specific IQs they wish to include. Access to these is managed at the enterprise level, so administrative permission may be required

#### Ontology and Embeddings

For Fabric and Foundry IQs to work, data must be properly prepared. This involves creating an ontology for structured data in Fabric and generating embeddings for documents in Foundry's knowledge base

### The IQ Agent in Action

The true power of this setup is seen in how the agent intelligently routes queries to the correct tool

#### Testing Work IQ

When asked about "top five recent meetings," the agent identifies Work IQ as the appropriate tool and requests the necessary permissions to access calendar data

#### Testing Fabric IQ

When queried about product sales in a retail space, the agent automatically connects to the retail ontology in Fabric to pull the exact figures

#### Testing Foundry IQ

For complex questions regarding specific documents (like a Department of Transportation RFP), the agent utilizes its knowledge base to extract details from private PDF files, ensuring the response is grounded in proprietary data rather than external sources

### Traceability and Integration

One of the standout features of building with Foundry is the built-in traceability. Developers can view detailed logs to see exactly how the agent processed a request and which tools were utilized. Additionally, the platform addresses AI quality and safety aspects by default

For those looking to move beyond the Foundry UI, the system provides the necessary code snippets to consume the agent within other applications, making it a flexible solution for enterprise-scale AI deployment

## Conclusion

The Microsoft IQ built with Foundry Agents represents a significant step toward truly contextual AI. By bridging the gap between communication tools, structured business data, and private document repositories, organizations can empower their teams with an agent that doesn't just chat, but actually understands the full breadth of their enterprise data