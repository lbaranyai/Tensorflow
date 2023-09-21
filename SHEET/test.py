# SHEET data processing / TensorFlow Lite model
# by lbaranyai@github
# video demonstration at https://youtu.be/cU-_Eo0AmQ8?si=WLIymyyXCpU1zZLM

# Start python environment in RStudio
no

# Libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Read data
ds = pd.read_csv('data.csv',sep=';')
ds = ds.astype('float32')
features = tf.convert_to_tensor(ds.iloc[:,0:10])

# Apply model
ipt = tf.lite.Interpreter(model_path='model.tflite')
ipt.resize_tensor_input(ipt.get_input_details()[0]['index'], [ds.shape[0], 10])
ipt.allocate_tensors()
ipt.set_tensor(ipt.get_input_details()[0]['index'], features)

ipt.invoke()

# Get result
prediction = ipt.get_tensor(ipt.get_output_details()[0]['index'])
answer = prediction[:, 0]
answer[np.where(answer < 0.5)] = 0
answer[np.where(answer > 0)] = 1
idx = np.where(answer - ds.iloc[:,10] == 0)[0]
print("Correct {0} of {1} = {2:.2f}%".format(idx.shape[0],ds.shape[0],100*idx.shape[0]/ds.shape[0]))
