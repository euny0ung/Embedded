#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.python.keras.models import load_model
import numpy as np
import cv2
# import RPi.GPIO as GPIO
import time
import winsound as sd


# GPIO.setwarnings(False)
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(12, GPIO.OUT)

# p=GPIO.PWM(18,100)

model = load_model('keras_model.h5', compile=False)

cap = cv2.VideoCapture(0)

size = (224, 224)

classes = ['Upright', 'Right', 'Left']
Frq=[263,294,330,392]


def beepsound():
    fr = 2000    # range : 37 ~ 32767
    du = 1000     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
            

    h, w, _ = img.shape
    cx = h / 2
    img = img[:, 200:200+img.shape[0]]
    img = cv2.flip(img, 1)

    img_input = cv2.resize(img, size)
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input = (img_input.astype(np.float32) / 127.0) - 1
    img_input = np.expand_dims(img_input, axis=0)

    prediction = model.predict(img_input)
    idx = np.argmax(prediction)

    cv2.putText(img, text=classes[idx], org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)
  
    if classes[idx]=="Right" or classes[idx]=="Left":
#         p.start(50)
#         time.sleep(1)
#         p.stop()
         beepsound()


        
    
    cv2.imshow('result', img)     
    
    if cv2.waitKey(1) == ord('q'):
        break


# In[ ]:





# In[ ]:





# In[ ]:




