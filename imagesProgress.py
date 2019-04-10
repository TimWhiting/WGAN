import numpy as np
import glob
from PIL import Image
initialPath = "./mnist_dcgan_dcgan/images/"

list_im = []
for k in range(1, 101):
    imageFolderPath = initialPath + "epoch_{}/image_1.png".format(k)
    imgPath = glob.glob(imageFolderPath)
    if imgPath is None or len(imgPath) < 1 or imgPath[0] is None or imgPath[0] == "":
        continue
    list_im.append(imgPath[0])

print(list_im)
imgs = [Image.open(i) for i in list_im]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted([(np.sum(i.size), i.size) for i in imgs[:10]])[0][1]
prevRows = None
for j in range(int(len(list_im)/10)):
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape))
                           for i in imgs[10*j:10*(j+1)]))
    if prevRows is None:
        prevRows = imgs_comb
    else:
        prevRows = np.vstack((prevRows, imgs_comb))

    # save that beautiful picture
imgs_comb = Image.fromarray(prevRows)
imgs_comb.save(initialPath + "timelinearray.png")
