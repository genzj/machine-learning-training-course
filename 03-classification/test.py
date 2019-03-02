import matplotlib.pyplot as plt
from scipy import ndimage

pixel_depth = 255
test_png = (ndimage.imread('./test.png').astype(float) - pixel_depth / 2) / pixel_depth
print(test_png.shape)
plt.figure()
plt.imshow(test_png)  # display it
plt.show()

test_png = (ndimage.imread('./notMNIST_small/A/MDEtMDEtMDAudHRm.png').astype(float) - pixel_depth / 2) / pixel_depth
print(test_png.shape)
plt.figure()
plt.imshow(test_png)  # display it
plt.show()
