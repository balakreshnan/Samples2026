# Azure Machine Learning Input Output Logging

## Introduction

- Logging inputs and outputs of experiments in Azure Machine Learning.
- Using built-in logging capabilities to track data flow.
- Helps in debugging and auditing machine learning workflows.
- Ability to troubleshoot issues by examining logged inputs and outputs.

## Prerequisites

- Azure Subscription
- Azure Machine Learning Workspace
- Compute Instance
- Compute cluster for training jobs
- Refering code from - https://github.com/Azure/azureml-examples/blob/main/cli/monitoring/azureml-e2e-model-monitoring/notebooks/model-monitoring-e2e.ipynb
- Input and output of Managed online endpoints
- Here is the score script - https://github.com/Azure/azureml-examples/blob/main/cli/monitoring/azureml-e2e-model-monitoring/code/score.py
- Also show how to view the input and output using code based on the endpoint and deployment names.

## Code Example

- Log into Azure ML workspace
- Create a compute instance
- Download all the files and folders from azureml-e2e-model-monitoring
- Entire repo might be overkill but if you want to use it's also okay.
- Create a notebook
- We are going to run through training, deployment and testing of the endpoint.
- There are few changes to wait for deployment to be succeeded.
- Also ability to view the input and outpt

```
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Connect to the project workspace
ml_client = MLClient.from_config(credential=DefaultAzureCredential())
```

- here is the compute cluster creation

```
from azure.ai.ml.entities import AmlCompute

cluster_basic = AmlCompute(
    name="cpu-cluster",
    type="amlcompute",
    size="STANDARD_F2S_V2",  # you can replace it with other supported VM SKUs
    location=ml_client.workspaces.get(ml_client.workspace_name).location,
    min_instances=0,
    max_instances=1,
    idle_time_before_scale_down=360,
)

ml_client.begin_create_or_update(cluster_basic).result()
```

- Now work on the data set

```
import pandas as pd
import datetime

# Read the default_of_credit_card_clients dataset into a pandas data frame
data_path = "https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv"
df = pd.read_csv(data_path, header=1, index_col=0).rename(
    columns={"default payment next month": "DEFAULT_NEXT_MONTH"}
)

# Split the data into production_data_df and reference_data_df
# Use the iloc method to select the first 80% and the last 20% of the rows
reference_data_df = df.iloc[: int(0.8 * len(df))].copy()
production_data_df = df.iloc[int(0.8 * len(df)) :].copy()

# Add a timestamp column in ISO8601 format
timestamp = datetime.datetime.now() - datetime.timedelta(days=45)
reference_data_df["TIMESTAMP"] = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
production_data_df["TIMESTAMP"] = [
    timestamp + datetime.timedelta(minutes=i * 10)
    for i in range(len(production_data_df))
]
production_data_df["TIMESTAMP"] = production_data_df["TIMESTAMP"].apply(
    lambda x: x.strftime("%Y-%m-%dT%H:%M:%S")
)
```

- split the data set into train and test

```
import os


def write_df(df, local_path, file_name):
    # Create directory if it does not exist
    os.makedirs(local_path, exist_ok=True)

    # Write data
    df.to_csv(f"{local_path}/{file_name}", index=False)


# Write data to local directory
reference_data_dir_local_path = "../data/reference"
production_data_dir_local_path = "../data/production"

write_df(reference_data_df, reference_data_dir_local_path, "01.csv"),
write_df(production_data_df, production_data_dir_local_path, "01.csv")
```

- Now create the ML table for reference and production data

```
import mltable
from mltable import MLTableHeaders, MLTableFileEncoding

from azureml.fsspec import AzureMachineLearningFileSystem
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes


def upload_data_and_create_data_asset(
    local_path, remote_path, datastore_uri, data_name, data_version
):
    # Write MLTable file
    tbl = mltable.from_delimited_files(
        paths=[{"pattern": f"{datastore_uri}{remote_path}*.csv"}],
        delimiter=",",
        header="all_files_same_headers",
        infer_column_types=True,
        include_path_column=False,
        encoding="utf8",
    )

    tbl.save(local_path)

    # Instantiate file system
    fs = AzureMachineLearningFileSystem(datastore_uri)

    # Upload data
    fs.upload(
        lpath=local_path,
        rpath=remote_path,
        recursive=False,
        **{"overwrite": "MERGE_WITH_OVERWRITE"},
    )

    # Define the Data asset object
    data = Data(
        path=f"{datastore_uri}{remote_path}",
        type=AssetTypes.MLTABLE,
        name=data_name,
        version=data_version,
    )

    # Create the data asset in the workspace
    ml_client.data.create_or_update(data)

    return data


# Datastore uri for data
datastore_uri = "azureml://subscriptions/{}/resourcegroups/{}/workspaces/{}/datastores/workspaceblobstore/paths/".format(
    ml_client.subscription_id, ml_client.resource_group_name, ml_client.workspace_name
)

# Define paths
reference_data_dir_remote_path = "data/credit-default/reference/"
production_data_dir_remote_path = "data/credit-default/production/"

# Define data asset names
reference_data_asset_name = "credit-default-reference"
production_data_asset_name = "credit-default-production"

# Write data to remote directory and create data asset
reference_data = upload_data_and_create_data_asset(
    reference_data_dir_local_path,
    reference_data_dir_remote_path,
    datastore_uri,
    reference_data_asset_name,
    "1",
)
production_data = upload_data_and_create_data_asset(
    production_data_dir_local_path,
    production_data_dir_remote_path,
    datastore_uri,
    production_data_asset_name,
    "1",
)
```

- Now train the model

```
from azure.ai.ml import load_job

# Define training pipeline directory
training_pipeline_path = "../configurations/training_pipeline.yaml"

# Trigger training
training_pipeline_definition = load_job(source=training_pipeline_path)
training_pipeline_job = ml_client.jobs.create_or_update(training_pipeline_definition)

ml_client.jobs.stream(training_pipeline_job.name)
```

- Now provide a new endpoint and deployment name

```
from azure.ai.ml import load_online_endpoint

# Define endpoint directory
endpoint_path = "../endpoints/endpoint.yaml"

# Trigger endpoint creation
endpoint_definition = load_online_endpoint(source=endpoint_path)
endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint_definition)
```

- wait for endpoint to be created

```
# Check endpoint status
endpoint = ml_client.online_endpoints.get(name=endpoint_definition.name)
print(
    f'Endpoint "{endpoint.name}" with provisioning state "{endpoint.provisioning_state}" is retrieved'
)
```

- Now create deployment

```from azure.ai.ml import load_online_deployment

# Define deployment directory
deployment_path = "../endpoints/deployment.yaml"

# Trigger deployment creation
deployment_definition = load_online_deployment(source=deployment_path)
deployment = ml_client.online_deployments.begin_create_or_update(deployment_definition)
```

- Now wait for deployment to be succeeded

```
import time

terminal_states = {"Succeeded", "Failed", "Canceled"}
success_state = "Succeeded"

while True:
    deployment = ml_client.online_deployments.get(
        name=deployment_definition.name,
        endpoint_name=endpoint_definition.name,
    )

    state = deployment.provisioning_state
    print(f'Deployment "{deployment.name}" provisioning state: "{state}"')

    if state == success_state:
        print("✅ Deployment is ready (Succeeded).")
        break

    if state in terminal_states and state != success_state:
        raise RuntimeError(f"❌ Deployment ended in terminal state: {state}")
  

    # Not done yet (e.g., Creating/Updating)
    time.sleep(15)
```

- set the traffic to 100 percent

```
endpoint = ml_client.online_endpoints.get(endpoint_definition.name)

endpoint.traffic = {
    "main": 100
}

ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print("✅ 100% traffic assigned to deployment 'main'")
```

- Function to process input through the endpoint

```
import numpy as np

# Define numeric and categotical feature columns
NUMERIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]
CATEGORICAL_FEATURES = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]


def generate_sample_inference_data(df_production, number_of_records=20):
    # Sample records
    df_sample = df_production.sample(n=number_of_records, replace=True)

    # Generate numeric features with random noise
    df_numeric_generated = pd.DataFrame(
        {
            feature: np.random.normal(
                0, df_production[feature].std(), number_of_records
            ).astype(np.int64)
            for feature in NUMERIC_FEATURES
        }
    ) + df_sample[NUMERIC_FEATURES].reset_index(drop=True)

    # Take categorical columns
    df_categorical = df_sample[CATEGORICAL_FEATURES].reset_index(drop=True)

    # Combine numerical and categorical columns
    df_combined = pd.concat([df_numeric_generated, df_categorical], axis=1)

    return df_combined
```

- Create the data set

```
import mltable
import pandas as pd
from azure.ai.ml import MLClient

# Load production / inference data
data_asset = ml_client.data.get("credit-default-production", version="1")
tbl = mltable.load(data_asset.path)
df_production = tbl.to_pandas_dataframe()

# Generate sample data for inference
number_of_records = 20
df_generated = generate_sample_inference_data(df_production, number_of_records)
```

- now invoke the endpoint

```
import json
import os

request_file_name = "request.json"

# Request sample data
data = {"data": df_generated.to_dict(orient="records")}

# Write sample data
with open(request_file_name, "w") as f:
    json.dump(data, f)

# Call online endpoint
result = ml_client.online_endpoints.invoke(
    endpoint_name=endpoint_definition.name,
    deployment_name=deployment_definition.name,
    request_file=request_file_name,
)

# Delete sample data
os.remove(request_file_name)
```

- print the result

```
print(result)
```

- output

```
"{\"DEFAULT_NEXT_MONTH\": [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false]}"
```

- now lets view the input and output logs

```
# pip install -U azureml-fsspec==1.3.1 pandas

import pandas as pd
from azureml.fsspec import AzureMachineLearningFileSystem

def browse_and_preview(uri, max_list=100):
    fs = AzureMachineLearningFileSystem(uri)

    items = fs.find("/")  # recursive
    print(f"\nURI: {uri}")
    print(f"Found {len(items)} items")
    for p in items[:max_list]:
        print("  ", p)

    # preview first parquet/csv/jsonl found
    candidates = [p for p in items if p.lower().endswith((".parquet",".csv",".jsonl"))]
    if not candidates:
        print("No parquet/csv/jsonl files found to preview.")
        return

    sample = candidates[0]
    print("\nPreviewing:", sample)

    if sample.lower().endswith(".parquet"):
        df = pd.read_parquet(fs.open(sample))
        print(df.head(10))
    elif sample.lower().endswith(".csv"):
        df = pd.read_csv(fs.open(sample))
        print(df.head(10))
    else:  # jsonl
        import json
        rows = []
        with fs.open(sample) as f:
            for _ in range(20):
                rows.append(json.loads(next(f)))
        df = pd.DataFrame(rows)
        print(df.head(10))

# ---- Example URIs (replace endpoint/deployment with yours) ----
endpoint = endpoint_definition.name
deployment = deployment_definition.name

base = f"azureml://subscriptions/subid/resourcegroups/rgname/workspaces/amlworkspacename/datastores/workspaceblobstore/paths/modelDataCollector/{endpoint}/{deployment}"
print(base)

browse_and_preview(f"{base}/model_inputs/")

#browse_and_preview(f"{base}/inputs_outputs")
```

- output

```
Previewing: data/credit-default/production/01.csv
   LIMIT_BAL  SEX  EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2  PAY_3  PAY_4  \
0      50000    1          2         2   23      2      2      0      0   
1      60000    1          2         2   26      0      0      0      0   
2     400000    1          2         2   27      0      0      0      0   
3      20000    1          5         2   27      5      4      3      2   
4      50000    1          3         2   27      0      0     -2     -2   
5     110000    1          2         2   27      0      0      0      0   
6      30000    1          3         2   23      0      0     -2     -1   
7     230000    1          2         2   27      0      0      0      0   
8      20000    1          3         3   23      0      0      0      0   
9      30000    1          1         2   24      0      0      0      2   

   PAY_5  ...  BILL_AMT5  BILL_AMT6  PAY_AMT1  PAY_AMT2  PAY_AMT3  PAY_AMT4  \
0      0  ...      21247      20066         8      2401      2254      2004   
1      0  ...      26958      28847      2282      2324      2049      2000   
2      0  ...      20737       9545      2501     10009      1437      1105   
3      2  ...      20113      19840         0         0         0       900   
4     -1  ...         70        120         0       100         0        70   
5      0  ...     105988     108617      5500      6000      6000      4000   
6      0  ...       7704      20204       430       400       601      7504   
7      0  ...       9811       9865      1816      5105      1293      2000   
8      0  ...      19807      12294      2000     20000      1612      1121   
9      0  ...       4430        906      1440      2259         0      1500   

   PAY_AMT5  PAY_AMT6  DEFAULT_NEXT_MONTH            TIMESTAMP  
0       704       707                   0  2025-12-14T17:15:39  
1      3000      1120                   1  2025-12-14T17:25:39  
2       510       959                   0  2025-12-14T17:35:39  
3         0         0                   0  2025-12-14T17:45:39  
4       200       100                   0  2025-12-14T17:55:39  
5      5000      4000                   0  2025-12-14T18:05:39  
6     15005      5674                   0  2025-12-14T18:15:39  
7       528      3000                   0  2025-12-14T18:25:39  
8       702      1000                   0  2025-12-14T18:35:39  
9       425       895                   0  2025-12-14T18:45:39  

[10 rows x 25 columns]
```

- now display output logs

```
browse_and_preview(f"{base}/model_outputs/")
```

- output

```
Previewing: data/credit-default/production/01.csv
   LIMIT_BAL  SEX  EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2  PAY_3  PAY_4  \
0      50000    1          2         2   23      2      2      0      0   
1      60000    1          2         2   26      0      0      0      0   
2     400000    1          2         2   27      0      0      0      0   
3      20000    1          5         2   27      5      4      3      2   
4      50000    1          3         2   27      0      0     -2     -2   
5     110000    1          2         2   27      0      0      0      0   
6      30000    1          3         2   23      0      0     -2     -1   
7     230000    1          2         2   27      0      0      0      0   
8      20000    1          3         3   23      0      0      0      0   
9      30000    1          1         2   24      0      0      0      2   

   PAY_5  ...  BILL_AMT5  BILL_AMT6  PAY_AMT1  PAY_AMT2  PAY_AMT3  PAY_AMT4  \
0      0  ...      21247      20066         8      2401      2254      2004   
1      0  ...      26958      28847      2282      2324      2049      2000   
2      0  ...      20737       9545      2501     10009      1437      1105   
3      2  ...      20113      19840         0         0         0       900   
4     -1  ...         70        120         0       100         0        70   
5      0  ...     105988     108617      5500      6000      6000      4000   
6      0  ...       7704      20204       430       400       601      7504   
7      0  ...       9811       9865      1816      5105      1293      2000   
8      0  ...      19807      12294      2000     20000      1612      1121   
9      0  ...       4430        906      1440      2259         0      1500   

   PAY_AMT5  PAY_AMT6  DEFAULT_NEXT_MONTH            TIMESTAMP  
0       704       707                   0  2025-12-14T17:15:39  
1      3000      1120                   1  2025-12-14T17:25:39  
2       510       959                   0  2025-12-14T17:35:39  
3         0         0                   0  2025-12-14T17:45:39  
4       200       100                   0  2025-12-14T17:55:39  
5      5000      4000                   0  2025-12-14T18:05:39  
6     15005      5674                   0  2025-12-14T18:15:39  
7       528      3000                   0  2025-12-14T18:25:39  
8       702      1000                   0  2025-12-14T18:35:39  
9       425       895                   0  2025-12-14T18:45:39  

[10 rows x 25 columns]
```

## Cleanup

- set the trafic to 0 percent

```

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

endpoint_name = "credit-default-bb26"   # <-- your endpoint name

endpoint = ml_client.online_endpoints.get(endpoint_name)

# Set traffic to 0 for the deployment(s) you currently route to
# If you know the deployment name:
endpoint.traffic = {"main": 0}  # replace "blue"

ml_client.begin_create_or_update(endpoint).result()
```

- delete the deployment

```
deployment_name = "main"   # <-- deployment to delete

ml_client.online_deployments.begin_delete(
    name=deployment_name,
    endpoint_name=endpoint_name
).result()
```

- delete all the deployments

```
deployments = ml_client.online_deployments.list(endpoint_name=endpoint_name)
for d in deployments:
    ml_client.online_deployments.begin_delete(
        name=d.name,
        endpoint_name=endpoint_name
    ).result()
```

- now delete the endpoint

```
ml_client.online_endpoints.begin_delete(name=endpoint_name).result()
```

## conclusion

- In this notebook we saw how to log inputs and outputs of managed online endpoints in Azure Machine Learning.
- We also saw how to view the logged inputs and outputs using code.
- This helps in debugging and auditing machine learning workflows effectively.
- Make sure to clean up resources to avoid unnecessary costs.