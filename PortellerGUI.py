import tkinter as tk
from keras.preprocessing.image import img_to_array
import numpy as np
from tkinter import filedialog
from tensorflow import keras
from PIL import ImageTk, Image
import os
import sys
img_dimension = 255
classnames = ['DVI', 'DisplayPort', 'HDMI', 'PS2', 'USBC', 'VGA']
predicted_model = ""

def resource_path(relative_path):
#"Get absolute path to resource, works for dev and for PyInstaller"
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

model_path = resource_path("PortellerModel")
model = keras.models.load_model(model_path)

def file_predict(photo):
    image = Image.open(photo)
    image = image.resize((img_dimension, img_dimension), Image.ANTIALIAS)
    numpy_image = img_to_array(image)
    image_batch = np.expand_dims(numpy_image, axis=0)
    predictions = model.predict(image_batch)
    predicted_model = ("This port is most likely:", classnames[np.argmax(predictions[0])])
    text.delete("1.0", "end")
    text.insert(tk.END, predicted_model)

def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    file_predict(filename)

    
root = tk.Tk()
root["bg"]="green"
root.title("Proteller")

border_color = tk.Frame(root, background="red")
button = tk.Button(root, text='Open', command=UploadAction, padx=20, pady=5, font=("Ariel",20))
text = tk.Text(border_color, height=5, width=40)

text.insert(tk.END, "Please select a photo with a close up of a computer port")

text.pack()
border_color.pack(padx=40, pady=40)
button.pack()

root.mainloop()