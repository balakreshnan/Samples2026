# Creating Microsoft Fabric Data agents using Databricks Unity Catalog

## Introduction

- Build a Microsoft Fabric Data Agent using Databricks Unity Catalog as the data source.
- Ability to consume the Delta lake data in the Unity Catalog and use it as a data source for the agent.
- Create semantic model if needed.
- Using Azure Databricks Mirroring in Microsoft Fabric which is as of in preview stage, when this article was written.
- Using the latest version of Microsoft Fabric and Azure Databricks as of writing this article.
- To create a ontology we need to create a lake house with shortcut to mirrored dataset
- From the lake house we can create ontology layer and then data agents.

## Prerequisites

- Azure subscription
- Azure Databricks workspace with Unity Catalog enabled
- Microsoft Fabric tenant
- Create the catalog
- Make sure some data set is available
- for my experiement i copied the sample dataset for tpch and stored as delta lake in the databricks unity catalog.

## Steps to make sure permission is set in Azure databricks unity catalog

- Log into Azure Databricks workspace.
- Go to Catalog
- Now enable metastore for external data access, this will allow Microsoft Fabric to access the data in unity catalog.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-7.jpg 'fine tuning model')

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-8.jpg 'fine tuning model')

- Select Catalog -> click settings (gear) -> Select metastore -> External data access  Preview -> enabled.
- Go to SQL Editor -> run the below command to give permission to Microsoft Fabric tenant id to access the data in unity catalog.

```
GRANT EXTERNAL USE SCHEMA
ON SCHEMA adbtest26_xxxxxx.default 
TO `admin@xxxxxxx.onmicrosoft.com`;

GRANT USE CATALOG ON CATALOG adbtest26_xxxx TO `admin@xxxxxxx.onmicrosoft.com`;
GRANT USE SCHEMA ON SCHEMA adbtest26_xxxxx.default TO `admin@xxxxxxx.onmicrosoft.com`;
GRANT SELECT ON SCHEMA adbtest26_xxxxx.default TO `admin@xxxxxxx.onmicrosoft.com`;

SHOW GRANTS ON SCHEMA adbtest26_xxxx.default;
```

- the above is must in Azure databricks unity catalog to allow Microsoft Fabric to access in Microsoft fabric

## Now step is fabric

- Go to Fabric
- Select the workspace -> Azure Databricks Mirroring -> create new link
  
![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-1.jpg 'fine tuning model')

- Copy the databricks workspace URL and paste in the link creation page and click next

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-2.jpg 'fine tuning model')

- Select the tables to add

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-3.jpg 'fine tuning model')

- Make sure auto sync is enabled and click create

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-4.jpg 'fine tuning model')

- Give a name mirror shortcut and create

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-5.jpg 'fine tuning model')

- Now check the mirror link and query the data

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-6.jpg 'fine tuning model')

## Creating Data Agent

- Go to workspace and add items -> data agent
- Give a name and click next
- Then add the data source, select the mirror shortcut created in the previous step and click next

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-10.jpg 'fine tuning model')

- Now select the table to use

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-11.jpg 'fine tuning model')

- Now ask questions and test

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-12.jpg 'fine tuning model')


## Create Ontology

- First we need to create a lake house with the mirror shortcut as source
- items -> lake house -> give a name -> next
- select the mirror shortcut created in the previous step as source and click next

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-13.jpg 'fine tuning model')

- Select the schema

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-14.jpg 'fine tuning model')

- Map the tables to Lake house and click create

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-15.jpg 'fine tuning model')

- Now check the lake shouse should be created for the ontology layer

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-16.jpg 'fine tuning model')

- now lets create ontology layer, items -> ontology -> give a name -> next
- select the lake house created in the previous step as source and click next

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-17.jpg 'fine tuning model')

- Bind the tables to ontology layer and click create

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-18.jpg 'fine tuning model')

- after binding

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-19.jpg 'fine tuning model')

- now create a data agent using the ontology layer as source and test the questions
- Items -> data agent -> give a name -> next

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-20.jpg 'fine tuning model')

- ask questions and test
- when processing

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-21.jpg 'fine tuning model')

![info](https://github.com/balakreshnan/Samples2026/blob/main/Fabric/images/adbunitylink-22.jpg 'fine tuning model')

- For semantic model please create one above the lake house and then create ontology layer using the semantic model and then create data agent using the ontology layer as source and test the questions. This will help to improve the question answering capabilities of the agent.
- Done