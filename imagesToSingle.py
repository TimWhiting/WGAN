import numpy as np
import glob
from PIL import Image
initialPath = "./images/sNORB_mlp/"

for k in range(1, 101):
    imageFolderPath = initialPath + "image_epoch_{}/".format(k)
    print(imageFolderPath)
    list_im = glob.glob(imageFolderPath + '/image_*.png')
    imgs = [Image.open(i) for i in list_im]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs[:10]])[0][1]
    prevRows = None
    for j in range(4):
        imgs_comb = np.hstack((np.asarray(i.resize(min_shape))
                               for i in imgs[10*j:10*(j+1)]))
        if prevRows is None:
            prevRows = imgs_comb
        else:
            prevRows = np.vstack((prevRows, imgs_comb))

    # save that beautiful picture
    imgs_comb = Image.fromarray(prevRows)
    imgs_comb.save(initialPath + "image_epoch_{}/fullarray.png".format(k))
