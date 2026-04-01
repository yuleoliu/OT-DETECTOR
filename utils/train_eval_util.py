
import sys
import os
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from transformers import CLIPModel
import timm

from torchvision import datasets, transforms
import torchvision.transforms as transforms
from dataloaders import StanfordCars, Food101, OxfordIIITPet, Cub2011
import clip

def set_model_clip(args):
    '''
    load Huggingface CLIP
    '''
    ckpt_mapping = {"ViT-B/16":"openai/clip-vit-base-patch16", 
                    "ViT-B/32":"openai/clip-vit-base-patch32",
                    "ViT-L/14":"openai/clip-vit-large-patch14",
                    "RN50":"RN50",
                    "RN101": "RN101"
                    }

    args.ckpt = ckpt_mapping[args.CLIP_ckpt]
    model = CLIPModel.from_pretrained(args.ckpt)
    model.eval()
    model = model.cuda()
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    return model, val_preprocess

def set_val_set(args, preprocess=None):
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 8, 'pin_memory': True}
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet')
        val_dataset = datasets.ImageFolder(path, transform=preprocess)
            
    elif args.in_dataset in ["ImageNet10", "ImageNet20", "ImageNet100"]:
        val_dataset = datasets.ImageFolder(os.path.join(
                root, args.in_dataset, 'val'), transform=preprocess)
    elif args.in_dataset == "car196":
        val_dataset = StanfordCars(root, split="test", download=False, transform=preprocess)
    elif args.in_dataset == "food101":
        val_dataset = Food101(root, split="test", download=True, transform=preprocess)
    elif args.in_dataset == "pet37":
        val_dataset = OxfordIIITPet(root, split="test", download=True, transform=preprocess)
    elif args.in_dataset == "bird200":
        val_dataset = Cub2011(root, train = False, transform=preprocess)
    elif args.in_dataset == "ImageNet_A":
        val_dataset = datasets.ImageFolder(os.path.join(
                root,'imagenet-a'), transform=preprocess)
    elif args.in_dataset == "ImageNet_R":
        val_dataset = datasets.ImageFolder(os.path.join(
                root,'imagenet-r'), transform=preprocess)
    return val_dataset


def set_oodset_ImageNet(args, out_dataset, preprocess, root):
    '''
    set OOD loader for ImageNet scale datasets
    '''
    if out_dataset == 'iNaturalist':
        testsetout = datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
    elif out_dataset == 'places365': # filtered places
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'Places'),transform=preprocess)  
    elif out_dataset == 'placesbg': 
        testsetout = torchvision.datasets.ImageFolder(root= os.path.join(root, 'placesbg'),transform=preprocess)  
    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd'),
                                        transform=preprocess)
        # testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Textures'),
        #                                 transform=preprocess)
    elif out_dataset == 'ImageNet10':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet10', 'train'), transform=preprocess)
    elif out_dataset == 'ImageNet20':
        testsetout = datasets.ImageFolder(os.path.join(args.root_dir, 'ImageNet20', 'val'), transform=preprocess)
    return testsetout
