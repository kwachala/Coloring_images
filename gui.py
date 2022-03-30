import tkinter as tk
from tkinter import E, W, Label, filedialog
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image, ImageTk
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from keras.models import load_model
from skimage.io import imsave
import os
import glob


class GUI:

    def __init__(self) -> None:
        self.filename = None
        # self.obj = LoadModel()
        # \self.obj.main()
        self.result = None

    def main(self):
        root = tk.Tk()

        root.geometry("1000x600")  # Size of the window
        root.title('Colouring')
        root.resizable(1,1)
        root.columnconfigure(0, minsize=500)
        root.columnconfigure(1, minsize=500)

        text2 = tk.StringVar()
        text2.set(f"Welcome! Here you can colour your picture!")
        label2 = Label(root, textvariable=text2, font=("Arial", 22))
        label2.grid(row=0, column=0, columnspan=2, pady=10, padx=10)

        self.text = tk.StringVar()
        self.text.set(f"Prediction: ")
        label = Label(root, textvariable=self.text)
        label.grid(row=2, column=1, pady=10, padx=10)

        self.text3 = tk.StringVar()
        self.text3.set(f"Uploaded file: ")
        label3 = Label(root, textvariable=self.text3)
        label3.grid(row=2, column=0, pady=10, padx=10)

        b1 = tk.Button(
            root,
            text='Upload image that you want to colour',
            command=lambda: self.get_photo(),
            background='white')

        b2 = tk.Button(
            root,
            text='Colour my picture',
            command=lambda: self.return_photo(),
            background='white'
        )

        b1.grid(row=1, column=0, pady=5, padx=10)
        b2.grid(row=1, column=1, pady=5, padx=10)

        root.mainloop()  # Keep the window open

    def get_photo(self):
        global img
        global image_first
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        # self.result = self.obj.prepdata(filename)
        image_first = Image.open(filename)
        image = Image.open(filename)
        image = image.resize((256, 256), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)
        photo1 = Label(image=img)
        photo1.grid(row=3, column=0)


    def return_photo(self):
        global coloured_image
        global coloured_img


        color_me = []
        color_me.append(img_to_array(image_first))
        color_me = np.array(color_me, dtype=float)
        color_me = rgb2lab(1.0 / 255 * color_me)[:, :, :, 0]
        color_me = color_me.reshape(color_me.shape + (1,))


        files = glob.glob('result_gui/*')
        for f in files:
            os.remove(f)

        model = load_model("model_save")

        output = model.predict(color_me)
        output = output * 128

        saved_file_path= ""

        for i in range(len(output)):
            cur = np.zeros((128, 128, 3))
            cur[:, :, 0] = color_me[i][:, :, 0]
            cur[:, :, 1:] = output[i]
            saved_file_path = "result_gui/img_" + str(i) + ".png"
            imsave(saved_file_path, lab2rgb(cur))

        coloured_image = Image.open(saved_file_path)
        coloured_image = coloured_image.resize((256, 256), Image.ANTIALIAS)
        coloured_img = ImageTk.PhotoImage(coloured_image)
        photo2 = Label(image=coloured_img)
        photo2.grid(row=3, column=1)


if __name__ == '__main__':
    obj = GUI()
    obj.main()