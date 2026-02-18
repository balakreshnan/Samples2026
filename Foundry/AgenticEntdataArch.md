# Agentic AI Enterprise Data Architecture - Microsoft Fabric IQ + Ontology + Microsoft Foundry

## Overview

- Build the next generation enterprise data architecture for agentic AI using Microsoft Fabric IQ, Ontology, and Microsoft Foundry.
- Leverage Microsoft Fabric IQ for data integration, transformation, and analytics.
- Utilize Ontology for semantic data modeling and knowledge representation.
- Implement Microsoft Foundry for data governance, security, and operationalization.
- Create a unified data architecture that supports the needs of agentic AI applications, enabling seamless data flow, advanced analytics, and robust governance.
- Make use of analytical data in data lakes, data warehouses, and data marts to support various AI workloads and use cases.
- Create the data virualization layer to enable seamless access to data across the enterprise, regardless of its physical location or format.
- Implement data governance policies and practices to ensure data quality, security, and compliance across the enterprise.
- Enable self-service analytics and data discovery for business users, empowering them to derive insights and make data-driven decisions.
- Build the AI era analytics using natual language based interactions, enabling users to ask questions and receive insights in a conversational manner.

## Reference Architecture

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/agenticairefarch-1.jpg 'fine tuning model')

## Flow as Implementated

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/fabriciqfoundryflow.jpg 'fine tuning model')

- idea here is to bring data from various sources into fabric or use mirroring, shortcut to connect to data sources into Lake house.
- Then build a ontology layer on top of the data to create a semantic model that can be used for AI applications. We can build the ontology by connecting to various different data sources and creating a unified view of the data.
- Build a data agent on top of ontology layer to enable natural language interactions with the data. This will allow users to ask questions and receive insights in a conversational manner.
- We can use the Fabric UI to ask for insights using data agents.
- Also we can customize and extend the data agents using Microsoft Foundry to create more advanced and sophisticated AI applications.
- Data agents can be published to Microsoft A365 for enterprise visibility and consumption. This will allow users across the organization to access and use the data agents for their own needs.
- From there it can be pushed to Copilot and other Microsoft services or tools for wider consumption and use.

## Pre-requisites

- Azure Subscription
- MIcrosoft Fabric Capacity
- Microsoft Foundry
- Agent 365 Access - part of M365 admin center
- Data sources to connect to Microsoft Fabric IQ and build the ontology layer.
- Sample data set to test the data agents and AI applications.
- Permissions to access and manage the data sources, Microsoft Fabric IQ, Microsoft Foundry, and Agent 365.
- Make sure you have fabric capacity and foundry resources provisioned and set up in your Azure subscription.
- Ensure you have the necessary permissions and access to the data sources you plan to connect to Microsoft Fabric IQ and build the ontology layer.

## Steps to Implement

- Go to Fabric UI
- Create a lak house and then upload each files as a different table in the lake house.
- i am using a simple dataset of sales data for this example, but you can use any dataset that is relevant to your use case.
- Once the data is uploaded, we can start building the ontology layer on top of the data. This will involve creating a semantic model that represents the relationships between the different data entities and attributes.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-1.jpg 'fine tuning model')

- Now go to Workspace - Select the workspace
- Click New Item -> Select Ontology
- Give a name to the ontology and click Create

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-2.jpg 'fine tuning model')

- will take few minutes to configure the ontology screen.
- Make sure the proper data lake house is selected in the data source settings.
- Now Select the tables that you want to include and build the ontology layer on top of it. You can select multiple tables and create relationships between them to build a unified view of the data.
- In my example select FactSales and then create relationships with DimCustomer, DimProduct, DimDate and DimStore tables based on the common keys between them.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-5.jpg 'fine tuning model')

- Now we are done creating the ontology layer on top of the data. We can now build a data agent on top of the ontology layer to enable natural language interactions with the data.
- Next go to Workspace -> New item -> Select Data Agent
- Give a name and then on the left screen, select add data source and select the ontology that we just created as the data source for the data agent.
- Click Add

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-6.jpg 'fine tuning model')

- now you should see the Chat UI

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-7.jpg 'fine tuning model')

- Now time to ask a question from the data

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-8.jpg 'fine tuning model')

- Now click Publish button
- Then go to Settings and Click Publish and copy the workspace URL and then click on Test Agent to test the data agent that we just created.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-9.jpg 'fine tuning model')

- Now that the data agent on top of ontology layer is working, we can now customize and extend the data agent using Microsoft Foundry to create more advanced and sophisticated AI applications.
- Go to Microsoft Foundry and create a new project for the data agent that we just created.
- In the Foundry UI.
- Create a new Agent
- Select the model as gpt 5.2

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-10.jpg 'fine tuning model')

- now select the Tools and select Fabric Data Agent that we just created as the tool for the agent.
- you need workspace id and asset it for data agent which you can get from the URL of the data agent that we published in the previous steps.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-11.jpg 'fine tuning model')

- now ask question to the agent in the testing console and you should see the response from the data agent that we created in Microsoft Fabric IQ.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-12.jpg 'fine tuning model')

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-13.jpg 'fine tuning model')

- So now we can access fabric data agent using ontology layer in Microsoft Foundry.
- Now to publish the fabric data agent to Agent 365 go to Fabric, select the data agent and click Publiish button.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-14.jpg 'fine tuning model')

- to publish to Agent 365 to foundry, select the agent created and click on publish.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-15.jpg 'fine tuning model')

- in the next screen fill the information and prepare the agent

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-16.jpg 'fine tuning model')

- Once prepartion is done, Select organization and click on publish
- Now go to Agent 365 and you should see the data agent that we created in the list of agents available for use.
- Here is the Foundry agent

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-17.jpg 'fine tuning model')

- here is the fabric agent

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/ontology-18.jpg 'fine tuning model')

## Conclusion

- we have successfully built the next generation enterprise data architecture for agentic AI using Microsoft Fabric IQ, Ontology, and Microsoft Foundry.
- we have created a unified data architecture that supports the needs of agentic AI applications, enabling seamless data flow, advanced analytics, and robust governance.
- we have leveraged Microsoft Fabric IQ for data integration, transformation, and analytics, utilized Ontology for semantic data modeling and knowledge representation, and implemented Microsoft Foundry for data governance, security, and operationalization.
- we have created a data virtualization layer to enable seamless access to data across the enterprise, regardless of its physical location or format, and implemented data governance policies and practices to ensure data quality, security, and compliance across the enterprise.
- we have enabled self-service analytics and data discovery for business users, empowering them to derive insights and make data-driven decisions, and built the AI era analytics using natural language based interactions, enabling users to ask questions and receive insights in a conversational manner.