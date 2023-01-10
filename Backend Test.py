#!/usr/bin/env python
# coding: utf-8

# In[16]:


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


# In[20]:


model = ResNet50(weights='imagenet')


# In[18]:


img_path = r'C:\Users\Lynne\OneDrive - Strathmore University\Pictures\pug.jpg'


# In[9]:


import matplotlib.pyplot as plt


# In[12]:


import sys  
get_ipython().system('{sys.executable} -m pip install --user matplotlib')


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


from skimage.io import imread


# In[18]:


pip install scikit-image


# In[19]:


from skimage.io import imread


# In[21]:


img = imread(img_path)


# In[22]:


plt.imshow(img)


# In[14]:


from PIL import Image


# In[13]:


pip install pillow


# In[21]:


img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)


# In[22]:


preds


# In[23]:


print('Predicted:', decode_predictions(preds, top=3)[0])


# In[ ]:




