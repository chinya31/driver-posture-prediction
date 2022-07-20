from tkinter import *
from PIL import Image,ImageTk
from tkinter.filedialog import askopenfile, askopenfilename, asksaveasfile
import os
import shutil
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from tensorflow.python.client import device_lib
print("num GPU ",len(tf.config.experimental.list_physical_devices('GPU')))
print(device_lib.list_local_devices())

test_datagen = ImageDataGenerator(rescale=1./255)
global test_img_path

model = keras.models.load_model(r'C:\Users\DELL\Desktop\mini project\vgg_model2.h5')

class_def = {0: 'safe driving',
             1: 'texting - right',
             2: 'talking on the phone - right',
             3: 'texting - left',
             4: 'talking on the phone - left',
             5: 'operating the radio',
             6: 'drinking',
             7: 'reaching behind',
             8: 'hair and makeup',
             9: 'talking to passenger'}

def upload_file():
    global test_img_path
    test_img_path = askopenfilename(filetypes=[('Image Files', '*.jpg')])
    #print(test_img_path)
    #print(type(test_img_path))
    img = Image.open(test_img_path)
    img.thumbnail((420,550))
    img = ImageTk.PhotoImage(img)
    im_label.configure(image=img)
    im_label.image = img
    shr.configure(state=NORMAL)


def showresult():
    dest_path = "C:/Users/DELL/Desktop/mini project/test_image"
    for f in os.listdir(dest_path):
        os.remove(os.path.join(dest_path, f))

    global test_img_path
    src_dir = test_img_path
    shutil.copy(src_dir, dest_path)

    image_size = (128, 128)
    test_path = "../mini project/"
    test_generator = test_datagen.flow_from_directory( directory=test_path, 
                                                  target_size=image_size, 
                                                  color_mode="rgb", 
                                                  batch_size=1, 
                                                  class_mode='categorical', 
                                                  shuffle=False, 
                                                  classes=['test_image'] )

    pred = model.predict(test_generator)
    predicted_class = np.argmax(pred,axis=1)
    #t = predicted_class[0]
    #print(t)
    result = class_def[predicted_class[0]]
    #print(class_def[0])

    re_label.configure(text=result)
    

root = Tk()
root.geometry("450x450+540+110")
root.maxsize(450,450)
root.minsize(450,450)
root.title("Driver Prediction")
root.configure(background="#e3edf6")
root.configure(highlightbackground="#3777ac")
root.configure(highlightcolor="black")

frm = Frame(root,background="#e3edf6")
frm.pack(side=BOTTOM,padx=15,pady=15)

im_label = Label(root)
im_label.pack(pady=15)

re_label = Label(frm,text="Here yol get the resutl")
re_label.pack(padx=8,pady=15)

uplo = Button(frm,text="Upload image",command=upload_file)
uplo.pack(padx=8,pady=8)

shr = Button(frm,text="Show Result",command=showresult,state=DISABLED)
shr.pack(padx=8,pady=8)

root.mainloop()