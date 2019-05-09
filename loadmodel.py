from keras.models import load_model
import numpy as np


model = load_model('../models/my_model.h5')
print(model.layers)
print(model.get_config())

print(model.get_weights())