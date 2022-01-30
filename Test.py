import os
import re
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Config import args
from Dataset import CityscapesLoader
from Train import model
from torchvision.transforms import Compose, ToPILImage

test_data_img = CityscapesLoader(args.local_path, split="test", is_transform=True, augmentations=None)
test_loader = DataLoader(test_data_img, batch_size=1, shuffle=args.test_shuffle)


class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


cityscapes_trainIds2labelIds = Compose([
    Relabel(19, 255),
    Relabel(18, 33),
    Relabel(17, 32),
    Relabel(16, 31),
    Relabel(15, 28),
    Relabel(14, 27),
    Relabel(13, 26),
    Relabel(12, 25),
    Relabel(11, 24),
    Relabel(10, 23),
    Relabel(9, 22),
    Relabel(8, 21),
    Relabel(7, 20),
    Relabel(6, 19),
    Relabel(5, 17),
    Relabel(4, 13),
    Relabel(3, 12),
    Relabel(2, 11),
    Relabel(1, 8),
    Relabel(0, 7),
    Relabel(250, 0),
    ToPILImage(),
])

root_pre = r"./XXX"
root_labelids = r"./XXXX"


def predict(img, label):
    model.eval()
    img = Variable(img).cuda()

    outputs = model(img)

    predict_results = outputs.max(1)[1].squeeze().cpu().data.numpy()
    pred_1 = outputs.max(1)[1].byte().squeeze().cpu().data
    label_1 = label.byte().squeeze().cpu()

    label_cityscapes = cityscapes_trainIds2labelIds(pred_1.unsqueeze(0))

    decoded = test_data_img.decode_segmap(predict_results) * 255
    decoded = decoded.astype(np.uint8)

    labels = label_1.cpu().numpy()
    labels = test_data_img.decode_segmap(labels) * 255
    labels = labels.astype(np.uint8)

    return decoded, label_cityscapes, labels


for i, datas in enumerate(test_loader):
    test_data, test_label, filename = datas
    name = filename[0].split("\\")[-1]
    pred, labelids, _ = predict(test_data, test_label)
    pred = Image.fromarray(pred).convert('RGB')
    pred.save(os.path.join(root_pre, '%s_test_labelIds.png' % re.sub("_\D.+", "", name)))
    labelids.save(os.path.join(root_labelids, '%s_pred_labelIds.png' % re.sub("_\D.+", "", name)))
