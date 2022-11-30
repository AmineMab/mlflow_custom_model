# Databricks notebook source
# MAGIC %pip install --upgrade mlflow

# COMMAND ----------

import mlflow
import mlflow.pyfunc
from custom_model.my_custom_model import AddN

# COMMAND ----------

mlflow.__version__

# COMMAND ----------

# Construct and save the model
model_path = "add_n_model"
add5_model = AddN(n=5)

# COMMAND ----------

# #save locally
# mlflow.pyfunc.save_model(path=model_path, python_model=add5_model)

# COMMAND ----------

#save to mlflow registry
# mlflow.pyfunc.log_model(artifact_path = "model", python_model=add5_model, code_path=["./custom_model"])

# COMMAND ----------

# Load the model in `python_function` format
loaded_model = mlflow.pyfunc.load_model("models:/custom_model_add_n/production")

# COMMAND ----------

# Evaluate the model
import pandas as pd
model_input = pd.DataFrame([range(10)])
model_output = loaded_model.predict(model_input)
assert model_output.equals(pd.DataFrame([range(5, 15)]))

# COMMAND ----------

loaded_model.predict(model_input)
