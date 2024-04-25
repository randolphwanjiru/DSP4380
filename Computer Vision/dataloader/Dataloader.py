#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

# In[4]:


#cd "archive (17)"


# In[5]:


#cd pizza_not_pizza


# In[6]:


get_ipython().system('ls')


# In[1]:


#import os

#base_dir = "/home/randolpwanjiru/DSP4380/Computer Vision/archive (17)/pizza_not_pizza"
#num_skipped = 0

#for folder_name in ("not_pizza", "pizza"):
    #max_images = 100
    #folder_path = os.path.join(base_dir, folder_name)
    
   # for image_i, fname in enumerate(os.listdir(folder_path)):
      #  fpath = os.path.join(folder_path, fname)
       # file_ext = os.path.splitext(fname)[1]
        #try:
          #  with open(fpath, "rb") as fpbj:
               # is_jfif = b"JFIF" in fpbj.peek(10)
        #except Exception as e:
            # Log or handle the exception appropriately
           # num_skipped += 1
            #print(f"Error processing {fpath}: {e}")
            # Delete corrupted file
           # os.remove(fpath)
            
        #if image_i > max_images or not is_jfif:
            #num_skipped += 1
            # Delete corrupted file
            #os.remove(fpath)

#print(f"Deleted {num_skipped} images.")
# it did succesfully delete 1704 images forgot to comment out 
#so it wouldnt run twice


# In[14]:


image_size = (224, 224)
batch_size = 10

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "/home/randolpwanjiru/DSP4380/Computer Vision/archive (17)/pizza_not_pizza",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


# ## 2 classes: 
# * 0 - not pizza
# * 1 - pizza
# 
# 1 file for pizza while 6 files for non pizza. imbalanced dataset detected 

# In[15]:


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
        


from tensorflow.keras.preprocessing import image_dataset_from_directory

def load_image_dataset(directory):
    train_ds = image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=(224, 224),
        batch_size=32,
        class_names= None # Specify the class names
    )

    val_ds = image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=(224, 224),
        batch_size=32,
        class_names= None  # Specify the class names
    )

    return train_ds, val_ds

def show_images(dataset, class_names=None, num_images=None):
    plt.figure(figsize=(15, 15))  # Increase the figure size for better visualization
    images_shown = 0
    num_cols = 3  # Number of columns per row
    for images, labels in dataset:
        for i in range(images.shape[0]):
            if num_images is not None and images_shown >= num_images:
                break
            num_rows = (images_shown // num_cols) + 1  # Calculate the number of rows needed
            plt.subplot(num_rows, num_cols, images_shown + 1)  # Adjust the subplot grid
            plt.imshow(images[i].numpy().astype("uint8"), aspect='equal')  # Preserve aspect ratio
            if class_names is not None:
                plt.title(class_names[int(labels[i])])  # Display class names
            else:
                plt.title(int(labels[i]))  # Display integer labels (0 or 1)
            plt.axis("off")
            images_shown += 1
    plt.show()





