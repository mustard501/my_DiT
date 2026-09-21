"""ImageFolder loading with ADM / official DiT center-crop."""
import os

import numpy as np
import torchvision
import torchvision.transforms as TF
from PIL import Image
from torch.utils.data import DataLoader

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def is_imagefolder(root):
    """True if root/class_name/ contains at least one image."""
    if not os.path.isdir(root):
        return False
    for cls in os.listdir(root):
        p = os.path.join(root, cls)
        if not os.path.isdir(p) or cls.startswith("."):
            continue
        try:
            names = os.listdir(p)
        except OSError:
            continue
        if any(f.lower().endswith(_IMG_EXTS) for f in names):
            return True
    return False


def center_crop_arr(pil_image, image_size):
    """Center crop from ADM / official DiT train.py."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def get_loader(data_dir, batch_size, size=256, num_workers=4):
    """256^2 images in [-1, 1]. ADM center-crop + flip (official DiT train.py)."""
    if not is_imagefolder(data_dir):
        raise FileNotFoundError(
            f"no ImageFolder at {data_dir} (expected {data_dir}/<class>/*.jpg)")

    def _crop(img):
        return center_crop_arr(img, size)

    tf = TF.Compose([
        TF.Lambda(_crop),
        TF.RandomHorizontalFlip(),
        TF.ToTensor(),
        TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    ds = torchvision.datasets.ImageFolder(data_dir, transform=tf)
    print(f"{len(ds)} images, {len(ds.classes)} classes from {data_dir}")
    kw = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers,
              drop_last=True, pin_memory=True)
    if num_workers > 0:
        kw["persistent_workers"] = True
    return DataLoader(ds, **kw), ds
