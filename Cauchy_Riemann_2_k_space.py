import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from numpy import linspace, arange, reshape, zeros


img = mpimg.imread('/Users/daltonhbermudez/Downlaods/brain_tumor_img.JPG')
plt.imshow(img)
plt.show()

x_min = 0
x_max = 530
y_min = 0
y_max = 612

x_array = zeros((612,530), dtype = float)
y_array = zeros((612, 530), dtype = float)

for row , y_value in enumerate(linspace(y_min, y_max, num = 612):
    for column , x_value in enumerate(x_min, x_max, num = 530):
        x_array[row][column] = x_value
        y_array[row][column] = y_value

a = np.sum(img,2)
k_space = fft(a)
n_values_y = fftfreq(612, (1.0/612.0))

n_values_x = fftfreq(530, (1.0/530.0))

kx_array = zeros((612, 530), dtype = float)
ky_array = zeros((612,530), dtype = float)

x_length = 530
y_length = 612

for row in arange(612):
    for column in arange(530):
        kx_array[row][column] = (2*pi*n_values_x[column])/x_length
        ky_array[row][column] = (2*pi*n_values_y[row])/y_length

u = np.real(k_space)
v = np.imag(k_space)

plt.quiver(kx_array,ky_array, u, v)
plt.show()


dux = np.gradient(u, kx_array[1,:],axis=1)
dvx = np.gradient(v, kx_array[1,:],axis=1)
k_prime_space = dux + dvx*1j # complex derivative definition 1 
img_2 = np.abs(scipy.fft.ifft2(k_prime_space))
plt.imshow(img_2)
plt.show()


dfx = np.gradient(u, kx_array[1,:],axis=1)
dfy = np.gradient(v, ky_array[:,1], axis = 0)

dfz = .5*(dfx-dfy*1j) # real-derivative definition
img_2_2 = np.abs(scipy.fft.ifft2(dfz))
plt.imshow(img_2_2)
plt.show()                               
