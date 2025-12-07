from functools import partial

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def def_dset_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))  # normalize to [-1, 1]
    ])
    val_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))  # normalize to [-1, 1]
    ])
    return train_transform, val_transform


def build_dataloaders(batch_size, tokenizer):
    train_transform, val_transform = def_dset_transforms()
    train_set = dset.CocoCaptions(root="../Datasets/COCO_subset/train2017",
                                  annFile="../Datasets/COCO_subset/annotations_trainval2017/captions_train2017.json",
                                  transform=train_transform)
    val_set = dset.CocoCaptions(root="../Datasets/COCO_subset/val2017",
                                annFile="../Datasets/COCO_subset/annotations_trainval2017/captions_val2017.json",
                                transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=False,  # no shuffle for easy training on the same subset
                                               collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             collate_fn=partial(collate_fn, tokenizer=tokenizer))
    return train_loader, val_loader


def collate_fn(batch, tokenizer):
    imgs, all_captions = zip(*batch)
    captions = [caps[0] for caps in all_captions]
    tokens = tokenizer(captions)
    imgs = torch.stack(imgs)
    return imgs, tokens
