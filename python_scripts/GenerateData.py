from PIL import Image
import numpy as np

if __name__ == '__main__':
  file = r'C:\Users\ttart\PycharmProjects\Test\Conv\grey_gwagon.png'

  # Read Image
  img = Image.open(file)
  # Convert Image to Numpy as array
  img = np.array(img)

  img.flatten("C").astype(np.float32).tofile('grey_lin_wagon.bin')