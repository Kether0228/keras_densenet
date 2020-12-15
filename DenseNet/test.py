import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications.densenet import DenseNet121
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers

#設定
image_size = (240,240)

batch_size = 32


#載入資料並切割為訓練資料及驗證資料

model = keras.models.load_model(
    'model/',
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'test/',
    seed=7414,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)
test_class_names = test_data.class_names
print("test_label:",test_class_names)

# #evaluate model 
test_loss,test_acc = model.evaluate(test_data,batch_size=batch_size)
print("test acc:",test_acc)

# pred_label = model.predict(test_data,batch_size=batch_size)
# print(np.argmax(pred_label, axis=1))

# model.summary()
