import os

import torch
from scipy import misc
from torch.utils import data
from torch.utils.data import DataLoader

from utils.Process import *
from utils.Config import get_parser


def recursive_glob(rootdir='.', suffix=''):
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


class CityscapesLoader(data.Dataset):
    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]
        # [0, 0, 0]
    ]

    label_colours = dict(zip(range(19), colors))

    def __init__(self, root, split="train", gt="gtFine", img_size=(512, 1024),
                 is_transform=False, augmentations=None):

        self.root = root
        self.gt = gt
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations

        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([73.16, 82.91, 72.39])
        self.files = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, gt, self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.png')

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle']

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("> No files for split=[%s] found in %s" % (split, self.images_base))

        print("> Found %d %s images..." % (len(self.files[split]), split))

    def __len__(self):

        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + '{}_labelIds.png'.format(self.gt))

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))

        img = misc.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))

        lbl = misc.imread(lbl_path)

        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def transform(self, img, lbl):

        img = img[:, :, ::-1]
        img = img.astype(float)
        img -= self.mean
        img /= 255.0
        img = img.transpose(2, 0, 1)

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            raise ValueError("> Segmentation map contained invalid class values.")

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


# Load Data
parser = get_parser()
args = parser.parse_args()

height, width = int(512 * args.crop_ratio), int(1024 * args.crop_ratio)

augment_train = Compose([
    RandomHorizontallyFlip(),
    RandomSized((0.5, 1.5)),
    RandomRotate(3),
    AdjustGamma(0.2),
    AdjustSaturation(0.2),
    AdjustContrast(0.2),
    RandomCrop((height, width))
])

augment_valid = Compose([Scale((height, width)), CenterCrop((height, width))])

train_data = CityscapesLoader(args.local_path, split="train", gt="gtFine", is_transform=True,
                              augmentations=augment_train)
valid_data = CityscapesLoader(args.local_path, split="val", gt="gtFine", is_transform=True,
                              augmentations=augment_valid)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=args.train_shuffle)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=args.val_shuffle)

print(len(train_loader), len(valid_loader))

print(train_data[0], valid_data[0])
