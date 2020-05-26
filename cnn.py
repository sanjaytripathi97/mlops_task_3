#!/usr/bin/env python
# coding: utf-8

# In[35]:

import os
from keras.layers import Convolution2D


# In[36]:


from keras.layers import MaxPooling2D


# In[37]:


from keras.layers import Flatten


# In[38]:


from keras.layers import Dense


# In[39]:


from keras.models import Sequential


# In[40]:


model = Sequential()


# In[41]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(64, 64, 3)
                       ))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())


model.add(Dense(units=128, activation='relu'))


# In[47]:


model.add(Dense(units=1, activation='sigmoid'))


# In[48]:


model.summary()


# In[49]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[50]:


from keras_preprocessing.image import ImageDataGenerator


# In[51]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '/root/task3/cnn_dataset/training_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        '/root/task3/cnn_dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
history = model.fit(
        training_set,
        steps_per_epoch=1,
        epochs=2,
        validation_data=test_set,
        validation_steps=800)


# In[86]:


accuracy = history.history['accuracy'][1]
#os.system("echo $accuracy > accuracy.txt")

accuracy = int(accuracy*100)

with open("accuracy.txt" , "w+") as fle:
    fle.write(str(accuracy))

print("\nFinal Accuracy:-" , accuracy)


# In[ ]:




