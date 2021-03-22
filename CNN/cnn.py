import numpy as np
import cv2
import matplotlib.pyplot as plt

class ConvolutionalLayer:
  def __init__(self, num_filters, filter_size, stride):
    self.filters = np.random.randn(num_filters, filter_size, filter_size) * 0.1
    self.stride = stride
    self.number_of_filters = num_filters
    self.last_input = None

  def forward(self, image):
    self.last_input = image

    input_dimension = image.shape[1]
    output_dimension = int((input_dimension - self.number_of_filters) / self.stride) + 1

    out = np.zeros((self.filters.shape[0], output_dimension, output_dimension))
    #print('Filters shape', self.filters.shape[0])
    #print('Out shape', out.shape)
    for f in range(self.filters.shape[0]):  # convolve each filter over the image,
      tmp_y = out_y = 0  # moving it vertically first and then horizontally
      while tmp_y + self.number_of_filters <= input_dimension:
        tmp_x = out_x = 0
        while tmp_x + self.number_of_filters <= input_dimension:
          patch = image[tmp_y:tmp_y + self.number_of_filters, tmp_x:tmp_x + self.number_of_filters, :]
          #print(patch * self.filters[f])
          out[f, out_y, out_x] = np.sum(self.filters[f] * patch)
          tmp_x += self.stride
          out_x += 1
        tmp_y += self.stride
        out_y += 1
    return out

img = cv2.imread('test.png')
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
num_filters = 3
filter_size = 3
strides = 2
print('Image shape;', img.shape)
conn = ConvolutionalLayer(num_filters, filter_size, strides)
out = conn.forward(img)
expected_shape = (img.shape[0] - filter_size)//strides +1
for filter in range(0,num_filters):
  plt.imshow(out[filter, :, :],cmap='gray')
  plt.show()