"""A modified image folder class
We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.

Modified from: https://github.com/PaddlePaddle/PaddleGAN/blob/master/ppgan/datasets/image_folder.py

Added labels in make_dataset
"""

from PIL import Image
import os
import os.path
from core.dataset import Dataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, max_dataset_size=float("inf")):
    images = []
    class_indexes = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        classname = os.path.basename(root)
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                class_index = class_to_idx[classname]
                images.append(path)
                class_indexes.append(class_index)

    return images[:min(float(max_dataset_size), len(images))], class_indexes[:min(float(max_dataset_size), len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(Dataset):

    def __init__(self, root, transform=None, img_size=None, return_paths=False,
                 loader=default_loader):
        classes, class_to_idx = self._find_classes(root)
        imgs, class_indexes = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.indx = class_indexes
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader
        self.img_size = img_size

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path = self.imgs[index]
        class_index = self.indx[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img, self.img_size)
        if self.return_paths:
            return img, class_index, path
        else:
            return img, class_index

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    root = '/Users/jiangli/Work/projects/github-projects/stargan-v2/stargan-v2-pytorch/data/celeba_hq/train'
    dataset = ImageFolder(root)
    print(len(dataset), dataset[0], dataset[-1])
