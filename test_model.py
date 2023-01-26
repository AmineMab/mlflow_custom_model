# Databricks notebook source
# MAGIC %pip install --upgrade mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC #1) import the required libraries

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from custom_model.my_custom_model import AddN

# COMMAND ----------

mlflow.__version__

# COMMAND ----------

# MAGIC %md
# MAGIC #2) Initialize the imported custom model to add 5 to any input given

# COMMAND ----------

# Construct and save the model
model_path = "add_n_model"
add5_model = AddN(n=5)

# COMMAND ----------

# MAGIC %md
# MAGIC #3) Testing the local model logging feature.

# COMMAND ----------

# #save locally if not exists
try:
  mlflow.pyfunc.save_model(path=model_path, python_model=add5_model)
except Exception as e:
  print("model already present locally")

# COMMAND ----------

#test loading model from local as pyfunc
loaded_model = mlflow.pyfunc.load_model(model_path)

# COMMAND ----------

# Evaluate the model
import pandas as pd
model_input = pd.DataFrame([range(10)])
model_output = loaded_model.predict(model_input)
assert model_output.equals(pd.DataFrame([range(5, 15)]))

# COMMAND ----------

model_input

# COMMAND ----------

model_output

# COMMAND ----------

# MAGIC %md
# MAGIC #4) Testing model to the MLflow model registry

# COMMAND ----------

#register model to the model registry. Adding path to the custom model in the package.
mlflow.pyfunc.log_model(artifact_path = "model", python_model=add5_model, code_path = ["./custom_model/"], )

# COMMAND ----------

from mlflow.models.utils import add_libraries_to_model
model_uri = "models:/custom_model_add_n/production"

# COMMAND ----------

add_libraries_to_model(model_uri)

# COMMAND ----------

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# Evaluate the model
import pandas as pd
model_input = pd.DataFrame([range(10)])
model_output = loaded_model.predict(model_input)
assert model_output.equals(pd.DataFrame([range(5, 15)]))

# COMMAND ----------

loaded_model.predict(model_input)

# COMMAND ----------

# MAGIC %md
# MAGIC #5) Using the custom model to do inference from REST endpoint

# COMMAND ----------

import numpy as np
import pandas as pd
import requests

import mlflow 
# We need both a token for the API, which we can get from the notebook.
# Recall that we discuss the method below to retrieve tokens is not the best practice. We recommend you create your personal access token and save it in a secret scope. 
DATABRICKS_API_TOKEN = mlflow.utils.databricks_utils._get_command_context().apiToken().get()


# Next we need an endpoint at which to execute our request which we can get from the Notebook's context
api_url = mlflow.utils.databricks_utils.get_webapp_url()

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://adb-441590990085545.5.azuredatabricks.net/model/custom_model_add_n/5/invocations'
  headers = {'Authorization': f'Bearer {DATABRICKS_API_TOKEN}', 'Content-Type': 'application/json'}
  ds_dict = dataset.to_dict(orient='records') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  rec_dict = {"dataframe_records": ds_dict}
  data_json = json.dumps(rec_dict, allow_nan=True)
  
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()


# Scoring a model that accepts pandas DataFrames
data =  pd.DataFrame([{
  "0": 5.1,
  "1": 3.5,
  "2": 1.4,
  "3": 0.2
},
{
  "0": 10.1,
  "1": 143.5,
  "2": 66.4,
  "3": 10.2
}])

score_model(data)

# # Scoring a model that accepts tensors
# data = np.asarray([[5.1, 3.5, 1.4, 0.2]])
# score_model(MODEL_VERSION_URI, DATABRICKS_API_TOKEN, data)

# COMMAND ----------


