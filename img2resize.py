from PIL import Image


image = Image.open("./1.JPG")
image = image.resize((1280, 720))
image.save('./moon.tiff')

