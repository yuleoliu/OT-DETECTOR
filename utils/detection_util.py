import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
import torchvision
import sklearn.metrics as sk
from transformers import CLIPTokenizer
from torchvision import datasets
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
import clip
def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def softmax(x, T):
    x = x/T
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x


def print_measures(log, auroc, aupr, fpr, method_name='Ours', recall_level=0.95):
    if log == None: 
        print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
        print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
        print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))
    else:
        log.debug('\t\t\t\t' + method_name)
        log.debug('  FPR{:d} AUROC AUPR'.format(int(100*recall_level)))
        log.debug('& {:.2f} & {:.2f} & {:.2f}'.format(100*fpr, 100*auroc, 100*aupr))

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_ood_scores_clip_dist(args, net, loader, test_labels, dataset, device, in_dist=False):
    to_np = lambda x: x.data.cpu().numpy()
    _score = []
    tokenizer = CLIPTokenizer.from_pretrained(args.ckpt) 
    cos_matrix = []
    text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels], padding=True, return_tensors="pt")
    with torch.no_grad():
        text_encoder = net.module if isinstance(net, DDP) else net
        text_features = text_encoder.get_text_features(
            input_ids = text_inputs['input_ids'].to(device),
            attention_mask = text_inputs['attention_mask'].to(device)
        ).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    tqdm_object = tqdm(loader, total=len(loader), desc=f"Rank {args.local_rank} Processing", disable=args.rank != 0)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            bz = images.size(0)
            images = images.to(device)
            images = images.view((-1,3,224,224)).contiguous()
            image_features = net.module.get_image_features(pixel_values = images).float() 
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.reshape(-1, 257, args.feat_dim)
            refined_feature = feature_calibration(
                ori_feature=image_features[:,0,:], 
                crop_feature=image_features[:,1:,:], 
                text_feature=text_features, 
                k = 20
            )
            refined_feature /= torch.norm(refined_feature, p=2, dim=1, keepdim=True) 
            output = refined_feature @ text_features.T
            cos_matrix.append(output)
        cos_matrix = torch.cat(cos_matrix)
    return to_np(cos_matrix)

def get_ood_scores_clip(args, net, loader, test_labels, dataset,in_dist=False):
    '''
    used for scores based on img-caption product inner products: MIP, entropy, energy score. 
    '''
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    tokenizer = CLIPTokenizer.from_pretrained(args.ckpt)
    cos_matrix = []
    tqdm_object = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            bz = images.size(0)
            labels = labels.long().cuda()
            images = images.cuda()
            images = images.view((-1,3,224,224)).contiguous()
            image_features = net.get_image_features(pixel_values = images).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.reshape(-1, 257, args.feat_dim)
            text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels], padding=True, return_tensors="pt")
            text_features = net.get_text_features(input_ids = text_inputs['input_ids'].cuda(),
                                            attention_mask = text_inputs['attention_mask'].cuda()).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            # image_features = image_features.unsqueeze(0)
            refined_feature = feature_calibration(ori_feature=image_features[:,0,:], crop_feature=image_features[:,1:,:], text_feature=text_features, k = 20)
            refined_feature /= torch.norm(refined_feature, p=2)
            output = refined_feature @ text_features.T
            cos_matrix.append(output)
        cos_matrix = torch.cat(cos_matrix)
    return to_np(cos_matrix)


    
def get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list):
    '''
    1) evaluate detection performance for a given OOD test set (loader)
    2) print results (FPR95, AUROC, AUPR)
    '''
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(-in_score, -out_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    print(f'in score samples (random sampled): {in_score[:3]}, out score samples: {out_score[:3]}')
    # print(f'in score samples (min): {in_score[-3:]}, out score samples: {out_score[-3:]}')
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr) # used to calculate the avg over multiple OOD test sets
    print_measures(log, auroc, aupr, fpr, args.score)



def feature_calibration(ori_feature, crop_feature, text_feature, k):
    batch = crop_feature.size(0)
    cos_sim_crop = torch.matmul(crop_feature, text_feature.t())  # [batch, 256, 1000]
    cos_sim_ori = torch.matmul(ori_feature, text_feature.t())  # [batch, 1000]

    ori_class = torch.argmax(cos_sim_ori, dim=1)  # [batch]
    crop_class = torch.argmax(cos_sim_crop, dim=2)  # [batch, 256]

    cls_mask = (ori_class.unsqueeze(1) == crop_class)  # [batch, 256]
    cos_sim_crop_masked = cls_mask.unsqueeze(2) * cos_sim_crop  # [batch, 256, 1000]
    del cos_sim_crop, cos_sim_ori, ori_class, crop_class

    sorted_cos, _ = torch.sort(cos_sim_crop_masked, dim=2)  # [batch, 256, 1000]
    crop_max_margin = sorted_cos[:, :, -1] - sorted_cos[:, :, -2]  # [batch, 256]
    del sorted_cos

    crop_max_margin_idx = torch.argsort(-crop_max_margin, dim=1)
    first_indice = torch.arange(batch).unsqueeze(1)

    crop_feature_sorted = crop_feature[first_indice, crop_max_margin_idx]
    crop_max_margin_sorted = -torch.sort(-crop_max_margin, dim=1)[0]

    topk_mask = torch.zeros_like(crop_max_margin_sorted)
    topk_mask[:, :k] = 1

    crop_feature_sorted *= topk_mask.unsqueeze(2)
    crop_max_margin_sorted *= topk_mask

    new_feat = torch.sum(crop_feature_sorted * crop_max_margin_sorted.unsqueeze(2), dim=1)
    margin = torch.sum(crop_max_margin_sorted, dim=1)
    zero_mask = (margin == 0)
    new_feat[zero_mask] = ori_feature[zero_mask]
    margin[zero_mask] = 1
    new_feat /= margin.unsqueeze(1)
    return new_feat

    
class RandomCrop_ori(object):
    def __init__(self, n_crop=2):
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        self.n_crop = n_crop
        self.random_crop = transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.no = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, x):
        
        views = [self.random_crop(x).unsqueeze(dim=0) for _ in range(self.n_crop)]
        views = torch.cat(views, dim=0)
        x = self.no(x)
        x = x.unsqueeze(dim=0)
        views = torch.cat([x,views],dim=0)
        return views
