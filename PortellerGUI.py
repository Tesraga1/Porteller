import tkinter as tk
from keras.preprocessing.image import img_to_array
import numpy as np
from tkinter import filedialog
from tensorflow import keras
from PIL import ImageTk, Image
img_dimension = 255
classnames = ['DVI', 'DisplayPort', 'HDMI', 'PS2', 'USBC', 'VGA']
model = keras.models.load_model("PortellerModel")
predicted_model = ""

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
button = tk.Button(root, text='Open', command=UploadAction)
text = tk.Text(root, height=2, width=40)
text["bg"]="orange"
button.pack()
text.pack()


root.mainloop()