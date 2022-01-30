import matplotlib.pyplot as plt
import numpy as np

from Dataset import train_loader, train_data

for i, data in enumerate(train_loader):
    if i == 1:
        break
    else:
        image, labels = data
        image = image.numpy()[:, ::-1, :, :]
        image = np.transpose(image, [0, 2, 3, 1])
        f, array = plt.subplots(2, 2, figsize=(12, 10))
        for j in range(2):
            array[j][0].imshow(image[j])
            array[j][0].axes.get_xaxis().set_visible(False)
            array[j][0].axes.get_yaxis().set_visible(False)
            array[j][1].imshow(train_data.decode_segmap(labels.numpy()[j]))
            array[j][1].axes.get_xaxis().set_visible(False)
            array[j][1].axes.get_yaxis().set_visible(False)
        plt.show()
