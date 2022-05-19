### EX NO : 09
### DATE  : 23.05.2022
# <p align="center"> KNN CLASSIFICATION ALGORITHM </p>
## Aim:
   To implement KNN classification algorithm in python.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner /Google Colab

## Related Theory Concept:

## Algorithm
1.
2.
3.
4.

## Program:
```
/*
Program to implement KNN classification algorithm.
Developed by   : Vijayaragavan ARR
RegisterNumber :  212220230059
*/
```
```python
import splitfolders  # or import split_folders
splitfolders.ratio("Raw", output="output", seed=1337, ratio=(.9, .1), group_prefix=None) # default values
import matplotlib.pyplot as plt
import matplotlib.image as mping
img = mping.imread("output/val/BREAD_KNIFE/breadkniferaw2.jpg")
plt.imshow(img)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)
train = train_datagen.flow_from_directory("output/train/",target_size=(224,224),seed=42,batch_size=32,class_mode="categorical")
test = train_datagen.flow_from_directory("output/val/",target_size=(224,224),seed=42,batch_size=32,class_mode="categorical")
from tensorflow.keras.preprocessing import image
test_image = image.load_img('output/val/BREAD_KNIFE/breadkniferaw2.jpg', target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = tf.expand_dims(test_image,axis=0)
test_image = test_image/255.
import tensorflow_hub as hub
m = tf.keras.Sequential([
hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"),
tf.keras.layers.Dense(20, activation='softmax')
])
m.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"])
history = m.fit(train,epochs=5,steps_per_epoch=len(train),validation_data=test,validation_steps=len(test))
classes=train.class_indices
classes=list(classes.keys())
m.predict(test_image)
classes[tf.argmax(m.predict(test_image),axis=1).numpy()[0]]
import pandas as pd
pd.DataFrame(history.history).plot()
m.summary()

```
## Output:

![image](https://user-images.githubusercontent.com/74660507/169310428-cf241697-9655-4dbe-8558-d7fdb641cc84.png)

![image](https://user-images.githubusercontent.com/74660507/169310545-ab9cde07-95fa-4ef9-83c6-5426f23631d5.png)

![image](https://user-images.githubusercontent.com/74660507/169310600-85855370-9648-4790-b384-c58e2e2c2430.png)

![image](https://user-images.githubusercontent.com/74660507/169310688-d9b7eb76-9d67-4b14-888d-0b623d3d1496.png)

![image](https://user-images.githubusercontent.com/74660507/169310772-23759a09-e856-4a29-a3f7-6a760ad1fc5c.png)

![image](https://user-images.githubusercontent.com/74660507/169310866-32449759-2e34-4015-9851-6e39bf3b3c37.png)

## Result:
Thus the python program successully implemented KNN classification algorithm.
