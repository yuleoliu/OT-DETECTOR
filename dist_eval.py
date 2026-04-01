import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler 
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import ot
from utils.common import setup_seed, get_num_cls, get_test_labels
from utils.train_eval_util import set_model_clip, set_val_set, set_oodset_ImageNet
from utils.detection_util import *
from transformers import CLIPTokenizer
from tqdm import tqdm


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates MCM Score for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                        choices=['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet100',
                                 'pet37', 'food101', 'car196', 'bird200','Cub100'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default="/home/liuyu/code/csp_adaneg/data/images_largescale", type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="eval_ood",
                        type=str, help="unique ID for the run")
    parser.add_argument('--seed', default=5, type=int, help="random seed")
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='mini-batch size per GPU') # Note: per GPU
    parser.add_argument('--T', type=int, default=1,
                        help='temperature parameter')
    parser.add_argument('--model', default='CLIP', type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14','RN50','RN101'], help='which pretrained img encoder to use')
    parser.add_argument('--score', default='MCM', type=str, choices=[
        'MCM', 'energy', 'max-logit', 'entropy', 'var', 'maha'], help='score options')
    parser.add_argument('--feat_dim', type=int, default=512, help='feat dim; 512 for ViT-B and 768 for ViT-L')
    parser.add_argument('--normalize', type = bool, default = False, help='whether use normalized features for Maha score')
    parser.add_argument('--generate', type = bool, default = True, help='whether to generate class-wise means or read from files for Maha score')
    parser.add_argument('--template_dir', type = str, default = 'img_templates', help='the loc of stored classwise mean and precision matrix')
    parser.add_argument('--subset', default = False, type =bool, help = "whether uses a subset of samples in the training set")
    parser.add_argument('--max_count', default = 250, type =int, help = "how many samples are used to estimate classwise mean and precision matrix")
    parser.add_argument('--crop_num', default = 256, type =int, help = "crop num for SaCR")

    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')
    parser.add_argument('--load', type=bool, default=False, help='load extracted feature')
    parser.add_argument('--load_path', type=str, default='./features')

    args = parser.parse_args()
    LOG_ROOT_DIR = "./log"
    os.makedirs(LOG_ROOT_DIR, exist_ok=True) 
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.n_cls = get_num_cls(args)
    return args

def setup_ddp(args):
    dist.init_process_group(backend='nccl')
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    torch.cuda.set_device(args.local_rank)
    print(f"[{os.getpid()}] Rank {args.rank} initialized (Local Rank: {args.local_rank}, World Size: {args.world_size})")

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    args = process_args()
    setup_seed(args.seed + args.local_rank) 
    
    if args.local_rank != -1:
        setup_ddp(args)
        is_main_process = (args.rank == 0)
    else:
        is_main_process = True
        
    assert torch.cuda.is_available()

    if args.local_rank != -1:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Running in non-DDP mode, using {device}.")

    crop_transform = RandomCrop_ori(args.crop_num)
    net_module, preprocess = set_model_clip(args)
    net_module = net_module.to(device)
    if args.local_rank != -1:
        net = DDP(net_module, device_ids=[args.local_rank])
    else:
        net = net_module
    net.eval()

    if args.in_dataset in ['ImageNet10']: 
        ood_datasets = ['ImageNet20']
    elif args.in_dataset in ['ImageNet20']: 
        ood_datasets = ['ImageNet10']
    elif args.in_dataset in [ 'ImageNet', 'ImageNet100', 'bird200', 'car196', 'food101', 'pet37']:
        ood_datasets = ['iNaturalist','SUN', 'places365', 'dtd']

    test_labels = get_test_labels(args)
    if is_main_process and args.load:
        cos_id = np.load(os.path.join(args.load_path, f'cos_id_imagenet_vit16_crop.npy'))
    else:
        id_dataset = set_val_set(args, crop_transform) 
        id_sampler = DistributedSampler(id_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.local_rank != -1 else None
        id_loader = torch.utils.data.DataLoader(id_dataset, batch_size=args.batch_size, sampler=id_sampler, shuffle=(id_sampler is None), num_workers=4)
        cos_id_part = get_ood_scores_clip_dist(args, net, id_loader, test_labels, args.in_dataset, device)
        cos_id = gather_results(cos_id_part, args) 
    mean_fpr = []
    mean_auroc = []
    for ood_dataset in ood_datasets:
        if is_main_process and args.load:
            cos_ood = np.load(os.path.join(args.load_path, f'cos_ood_{ood_dataset}_vit16_crop.npy'))
        else:
            ood_dataset_obj = set_oodset_ImageNet(args, ood_dataset, crop_transform,args.root-dir )
            ood_sampler = DistributedSampler(ood_dataset_obj, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.local_rank != -1 else None
            ood_loader = torch.utils.data.DataLoader(ood_dataset_obj, batch_size=args.batch_size, sampler=ood_sampler, shuffle=(ood_sampler is None), num_workers=4)
            cos_ood_part = get_ood_scores_clip_dist(args, net, ood_loader, test_labels, args.in_dataset, device)
            cos_ood = gather_results(cos_ood_part, args)

        if is_main_process:
            id_sim = cos_id
            id_size = id_sim.shape[0]
            lambd = 90
            bsauroc = 0.
            bsfpr = 1.            
            ood_sim = cos_ood
            cos_sim = np.concatenate([id_sim, ood_sim], axis=0)
            score = np.max(softmax(cos_sim, 1), axis=1)
            auroc, _, fpr = get_measures(score[:id_size], score[id_size:]) 
            print(f'the mcm score for dataset:{ood_dataset} ,fpr:{fpr}, auroc:{auroc} ')

            M = 1 - cos_sim
            a = np.full(cos_sim.shape[0], 1 / cos_sim.shape[0])
            b = np.full(cos_sim.shape[1], 1 / cos_sim.shape[1])
            a_t = torch.from_numpy(a).to(device).float()
            b_t = torch.from_numpy(b).to(device).float()
            M_t = torch.from_numpy(M).to(device).float()
            Gs = ot.sinkhorn(a_t, b_t, M_t, 1/lambd, verbose=False, warn = False, numItermax=10000)
            Gs = Gs.cpu().numpy()
            M = M_t.cpu().numpy()
            score1 = np.max(Gs, axis=1)
            score2 = 1 - np.sum(Gs * M, axis=1)
            scores_combined = np.vstack((score1, score2)).T
            scaler = MinMaxScaler()
            scores_normalized = scaler.fit_transform(scores_combined)
            score1 = scores_normalized[:, 0]
            score2 = scores_normalized[:, 1]
            for alpha in np.linspace(0, 1, 11):
                beta = 1 - alpha
                score = alpha * score1 + beta * score2
                auroc, aupr, fpr = get_measures(score[:id_size], score[id_size:])
                if auroc > bsauroc and bsfpr > fpr:
                    bsauroc = auroc
                    bslam = lambd
                    bsfpr = fpr
                    best_alpha = alpha
                    best_beta = beta
            mean_fpr.append(bsfpr)
            mean_auroc.append(bsauroc)
            with open(f"./log/result_{args.in_dataset}.txt", 'a') as f:
                f.write(f'the OT score for dataset:{ood_dataset}, best_alpha:{best_alpha:.1f}, best_beta:{best_beta:.1f}, fpr:{bsfpr:.4f},  auroc:{bsauroc:.4f}\n')
            print(f'the OT score for dataset:{ood_dataset}, best_alpha:{best_alpha:.1f}, best_beta:{best_beta:.1f}, fpr:{bsfpr:.4f},  auroc:{bsauroc:.4f}')

    if is_main_process:
        mean_fpr_val = np.mean(np.array(mean_fpr))
        mean_auroc_val = np.mean(np.array(mean_auroc))
        print(f'mean_fpr:{mean_fpr_val:.4f},mean_auroc:{mean_auroc_val:.4f}\n')
        with open(f"./log/result_{args.in_dataset}.txt", 'a') as f:
            f.write(f'mean_fpr:{mean_fpr_val:.4f}, mean_auroc:{mean_auroc_val:.4f}\n')
            
    if args.local_rank != -1:
        cleanup_ddp()


def gather_results(local_scores, args):
    if args.local_rank == -1:
        return local_scores 
    local_tensor = torch.from_numpy(local_scores).to(args.local_rank)
    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(args.world_size)]
    dist.all_gather(gathered_tensors, local_tensor)
    if args.rank == 0:
        full_scores = torch.cat(gathered_tensors, dim=0).cpu().numpy()
        return full_scores
    else:
        return None

if __name__ == '__main__':
    main()