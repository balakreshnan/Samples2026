# Microsoft m365 Copilot UI + Mircosoft Foundry Agents + Foundry IQ

## Introduction

- Create a RFP application using Microsoft m365 Copilot UI, Microsoft Foundry Agents, and Foundry IQ.
- We are going to use existing Azure Search index which has some open source RFP documents to create a RFP application.
- RFP documents are chunked and stored in Azure Search index, and we will use Microsoft Foundry Agents to retrieve relevant chunks based on user queries.
- Then create a Foundry IQ knowledge base in Microsoft Foundry, by leveraging existing Azure AI Search service which has RFP documents indexed, and use it to answer user queries in natural language.
- Next we create a Agent in Microsoft Foundry that will use the Foundry IQ knowledge base to answer user queries and provide relevant information from the RFP documents.
- Create the prompt for the agent to retrieve relevant information from the RFP documents based on user queries and provide a comprehensive response.
- Next we publish the agent to Microsoft Teams. Copilot which gets published to Microsoft Agent 365 for approval. Once approved, the agent will be available in Microsoft Teams for users to interact with and get answers to their RFP related queries.
- Once approved, we can see it in Copilot UI. 

## Steps

- First lets create the foundry IQ knowledge base in Microsoft Foundry, by leveraging existing Azure AI Search service which has RFP documents indexed, and use it to answer user queries in natural language.
- Make sure you have the Azure Search index set up with the RFP documents and that it is accessible from Microsoft Foundry.
- In Microsoft Foundry, navigate to the Knowledge Base section and create a new knowledge base.
- Select the option to connect to an existing Azure Search index and provide the necessary details to connect to your Azure Search service.
- Once connected, you can configure the knowledge base to use the RFP documents indexed in Azure Search. This will allow the knowledge base to retrieve relevant information from the RFP documents based on user queries.
- After setting up the knowledge base, you can test it by entering sample queries related to RFPs and verifying that it retrieves the correct information from the Azure Search index.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-1.jpg 'fine tuning model')

- Next we are going to create an agent in Microsoft Foundry that will use the Foundry IQ knowledge base to answer user queries and provide relevant information from the RFP documents.
- In Microsoft Foundry, navigate to the Agents section and create a new agent.
- Configure the agent to use the Foundry IQ knowledge base that you created in the previous step. This will allow the agent to access the information from the RFP documents indexed in Azure Search.
- Next, you will need to create a prompt for the agent to retrieve relevant information from the RFP documents based on user queries and provide a comprehensive response. The prompt should be designed to guide the agent in understanding the user's query and retrieving the most relevant information from the knowledge base.
- Once you have created the prompt, you can test the agent by entering sample queries related to RFPs and verifying that it retrieves the correct information from the Foundry IQ knowledge base.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-2.jpg 'fine tuning model')

- Above image shows the prompt we created for the agent to retrieve relevant information from the RFP documents based on user queries and provide a comprehensive response. The prompt is designed to guide the agent in understanding the user's query and retrieving the most relevant information from the knowledge base.
- It also show the MCP selected for foundry IQ, quality metrics and other resposibile AI metrics calculated for the agent.

- now we are going to publish the agent to Microsoft Teams. Copilot which gets published to Microsoft Agent 365 for approval. Once approved, the agent will be available in Microsoft Teams for users to interact with and get answers to their RFP related queries.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-3.jpg 'fine tuning model')

- Select the options to publish the agent to Microsoft Teams and submit it for approval. The approval process may take some time, so be patient while waiting for the agent to be reviewed and approved by Microsoft.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-4.jpg 'fine tuning model')

- Now you will see the details to prepare the agent for submission to Microsoft Agent 365 for approval. This includes providing necessary information about the agent, such as its name, description, and any relevant documentation or resources that may be required for the approval process.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-5.jpg 'fine tuning model')

- now configure the organization scope or individual team scope for the agent to consume the agent in Microsoft Teams. You can select the appropriate scope based on your requirements and submit the agent for approval.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-6.jpg 'fine tuning model')

- Then click submit

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-7.jpg 'fine tuning model')

- Once the agent is submitted for publish, below screen is the confirmation

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-8.jpg 'fine tuning model')

- Now go to M365 Administration center and navigate to the Teams apps section. Here you can see the status of your agent submission for approval. It may take some time for Microsoft to review and approve the agent, so be patient while waiting for the approval process to complete.
- https://admin.cloud.microsoft/?#/agents/all/requested
- In the All agents section, go to Requested tab to see the status of your agent submission for approval. Once the agent is approved, it will be available in Microsoft Teams for users to interact with and get answers to their RFP related queries.
- you should see the agent submitted for approval

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-9.jpg 'fine tuning model')

- Click on the agent and click publish on the top next to the name of the agent. This will publish the agent to Microsoft Teams and make it available for users to interact with and get answers to their RFP related queries.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-10.jpg 'fine tuning model')

- When you click publish, there will a set of screen to walk throug.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-11.jpg 'fine tuning model')

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-12.jpg 'fine tuning model')

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-13.jpg 'fine tuning model')

- confirm changes

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-14.jpg 'fine tuning model')

- Status

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-15.jpg 'fine tuning model')

- Once published, you should see the confirmation screen as below

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-16.jpg 'fine tuning model')

- Now go to Microsoft Copilot https://m365.cloud.microsoft/
- on the left navigation, there should be a agent called constructionRFP agent

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-17.jpg 'fine tuning model')

- Select the agent

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-18.jpg 'fine tuning model')

- Ask few questions related to RFP and see the responses from the agent based on the information retrieved from the Azure Search index through the Foundry IQ knowledge base.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-19.jpg 'fine tuning model')

- validate the sources from where the agent is retrieving the information to answer the user queries. The agent should be able to provide relevant information from the RFP documents indexed in Azure Search through the Foundry IQ knowledge base.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/copilotui-20.jpg 'fine tuning model')

## Conclusion

- In this walkthrough, we have successfully created a RFP application using Microsoft m365 Copilot UI, Microsoft Foundry Agents, and Foundry IQ. We leveraged an existing Azure Search index with RFP documents to create a knowledge base in Microsoft Foundry, and then created an agent that uses this knowledge base to answer user queries in natural language. Finally, we published the agent to Microsoft Teams and verified that it is able to provide relevant information from the RFP documents based on user queries. This demonstrates the power of integrating Microsoft Foundry with Azure Search and Microsoft m365 Copilot UI to create intelligent applications that can provide valuable insights and information to users.
- Azure AI Search - Custom documents -> Indexed documents -> Microsoft Foundry IQ -> Knowledge base -> Microsoft Foundry Agent -> Microsoft Copilot UI