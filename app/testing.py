import tensorflow as tf
import pandas as pd
import numpy as np
import ktrain
from ktrain import text

# Set a seed for reproducibility
tf.random.set_seed(42)


predictor_load=ktrain.load_predictor('C:/Users/Lenovo/Downloads/bert_model')

data=['I am good']
my_prediction = predictor_load.predict(data)
print(my_prediction)