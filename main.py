import io
import numpy as np
import pandas as pd

import keras

import plotly.express as px
import plotly.subplots as make_subplots
import plotly.graph_objects as go
import seaborn as sns

data_link = "https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv"

chicago_taxi_dataset = pd.read_csv(data_link)

training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
training_df.head(200)