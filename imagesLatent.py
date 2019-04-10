import numpy as np
import glob
from PIL import Image
initialPath = "./mnist_mlp_dcgan/imageSweep/epoch_100/"

list_im = []
list_values = ["-1.0", "-0.5", "0.0", "0.5", "1.0"]
latentVariables = 20
for k in list_values:
    for j in range(1, latentVariables+1):
        imageFolderPath = initialPath + \
            "latentIndex_{}_value_{}.png".format(j, k)
        imgPath = glob.glob(imageFolderPath)
        if imgPath is None or len(imgPath) < 1 or imgPath[0] is None or imgPath[0] == "":
            continue
        list_im.append(imgPath[0])

print(list_im)
imgs = [Image.open(i) for i in list_im]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
min_shape = sorted([(np.sum(i.size), i.size) for i in imgs[:20]])[0][1]
prevRows = None
for j in range(int(len(list_im)/20)):
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape))
                           for i in imgs[20*j:20*(j+1)]))
    if prevRows is None:
        prevRows = imgs_comb
    else:
        prevRows = np.vstack((prevRows, imgs_comb))

    # save that beautiful picture
imgs_comb = Image.fromarray(prevRows)
imgs_comb.save(initialPath + "latentImage.png")
