from PIL import Image
import matplotlib.pyplot as plt
from glob import glob

fig = plt.figure()

image1 = Image.open('montage/21.png')
ax1 = fig.add_subplot(1,2,1)
ax1.set_title('montage')
ax1.imshow(image1)

image2 = Image.open('L/21_1771.jpg')
ax2 = fig.add_subplot(1,2,2)
ax2.set_title('sketch')
ax2.imshow(image2)

plt.show()

trainA_list = glob('montage/*.png')
trainB_list = glob('L/*.png') + glob('L/*.jpg') + glob('L/*.jpeg')
print(len(trainA_list))
print(len(trainB_list))