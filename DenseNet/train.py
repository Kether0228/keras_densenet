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
epochs = 100
num_classes = 2


#載入資料並切割為訓練資料及驗證資料
tdata = tf.keras.preprocessing.image_dataset_from_directory(
    'train/',
    validation_split=0.2,
    subset="training",
    seed=7414,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)
vdata = tf.keras.preprocessing.image_dataset_from_directory(
    'train/',
    validation_split=0.2,
    subset="validation",
    seed=7414,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

train_class_names = tdata.class_names
print("train_label:",train_class_names)

val_class_names = vdata.class_names
print("val_label:",val_class_names)


#預處理資料
tdata = tdata.prefetch(buffer_size=batch_size)
vdata = vdata.prefetch(buffer_size=batch_size)




input_size = (image_size[0],image_size[1],3)
#載入模型 可參考https://avacheng.github.io/post/20190817/

model = DenseNet121(
    include_top=True,#是否包含全連階層
    weights=None,#是否載入imagenet權重
    input_shape=input_size, #輸入圍度
    classes=num_classes, #類別數量
    #classifier_activation="softmax",
)
 

"""
dropout_rate = 0.2
model = models.Sequential()
model.add(conv)
model.add(layers.GlobalMaxPooling2D(name="gap"))
# model.add(layers.Flatten(name="flatten"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
# model.add(layers.Dense(256, activation='relu', name="fc1"))
model.add(layers.Dense(2, activation="sigmoid", name="fc_out"))
"""
#訓練參數設定

model.compile(
    optimizer="adam",
    loss='categorical_crossentropy',#categorical_crossentropy #binary_crossentropy
    metrics='accuracy',
)


history = model.fit(
    tdata,
    epochs=epochs,
    validation_data=vdata,
    batch_size=batch_size
)

model.save(
    filepath='model/',
    overwrite=True,
    save_format='tf',
)

#model.summary()
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()