## Grape data processing using TensorFlow and NN
## ICT-AGRI-FOOD project SHEET
## Copyright CC-BY: lbaranyai@github

## Load libraries
import numpy as np
import pandas as pd
import tensorflow as tf

## Load data from file
ds = pd.read_csv('grape_training_data.csv',sep=';')
# Split data to 80% training 20% validation
nrow = ds.shape[0]
idx = np.random.choice(nrow,size=int(round(0.2*nrow,0)),replace=False)
train_data = ds.iloc[-idx]
test_data  = ds.iloc[idx]
print('Validation data:')
print(test_data)
## Make TensorFlow dataframe
features = tf.convert_to_tensor(train_data.iloc[:,0:10])
target = train_data.pop('DAMAGE_future')
# Validation data
vfeatures = tf.convert_to_tensor(test_data.iloc[:,0:10])
vtarget = test_data.pop('DAMAGE_future')

## Create NN model
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(features)

def get_basic_model():
  model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(5, activation=tf.nn.relu),
    tf.keras.layers.Dense(3, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='SGD',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

## Run optimization
model = get_basic_model()
hist = model.fit(features, target, epochs=1500, verbose=0)

## Validation
print('Calibration performance:')
model.evaluate(features, target)
print('Validation performance:')
model.evaluate(vfeatures, vtarget)

# Save model
m = hist.history['accuracy'][-1]
if m > 0.9:
  ## Save to disk
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  # Save the model.
  with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
  pass
