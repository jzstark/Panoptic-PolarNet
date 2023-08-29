import os
import time
import argparse
import sys
import yaml
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import errno

from network.BEV_Unet import BEV_Unet
from network.ptBEV import ptBEVnet
from dataloader.dataset import collate_fn_BEV,SemKITTI,SemKITTI_label_name,spherical_dataset,voxel_dataset,collate_fn_BEV_test
from network.instance_post_processing import get_panoptic_segmentation
from utils.eval_pq import PanopticEval
from utils.configs import merge_configs
#ignore weird np warning
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', '--data_dir', default='data')
parser.add_argument('-p', '--pretrained_model', default='pretrained_weight/Panoptic_SemKITTI_PolarNet.pt')
parser.add_argument('-c', '--configs', default='configs/SemanticKITTI_model/Panoptic-PolarNet.yaml')
    
args = parser.parse_args()
with open(args.configs, 'r') as s:
    new_args = yaml.safe_load(s)
args = merge_configs(args,new_args)

data_path = args['dataset']['path']
test_batch_size = args['model']['test_batch_size']
pretrained_model = args['model']['pretrained_model']
output_path = args['dataset']['output_path']
compression_model = args['dataset']['grid_size'][2]
grid_size = args['dataset']['grid_size']
visibility = args['model']['visibility']
pytorch_device = torch.device('cuda:0')

assert args['model']['polar'] == True
fea_dim = 9
circular_padding = True

unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
unique_label_str=[SemKITTI_label_name[x] for x in unique_label+1]

my_BEV_model=BEV_Unet(n_class=len(unique_label), n_height = compression_model, input_batch_norm = True, dropout = 0.5, circular_padding = circular_padding, use_vis_fea=visibility)
my_model = ptBEVnet(my_BEV_model, pt_model = 'pointnet', grid_size =  grid_size, fea_dim = fea_dim, max_pt_per_encode = 256,
                            out_pt_fea_dim = 512, kernal_size = 1, pt_selection = 'random', fea_compre = compression_model)
if os.path.exists(pretrained_model):
    my_model.load_state_dict(torch.load(pretrained_model))

my_model.to(pytorch_device)
my_model.eval()


val_pt_dataset = SemKITTI(data_path + '/sequences/', imageset = 'val', return_ref = True, instance_pkl_path=args['dataset']['instance_pkl_path'])
val_dataset=spherical_dataset(val_pt_dataset, args['dataset'], grid_size = grid_size, ignore_label = 0)
val_dataset_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                    batch_size = test_batch_size,
                                                    collate_fn = collate_fn_BEV,
                                                    shuffle = False,
                                                    num_workers = 4)

pbar = tqdm(total=len(val_dataset_loader))
time_list = []
pp_time_list = []
evaluator = PanopticEval(len(unique_label)+1, None, [0], min_points=50)

# One single item
val_vox_fea,val_vox_label,val_gt_center,val_gt_offset,val_grid,val_pt_labels,val_pt_ints,val_pt_fea = next(iter(val_dataset_loader))

"""
vox_fea: [1, 32, 480, 360]; most -1, some 0

vox_label: [1, 480, 360, 32]; mostly 0
gt_center: [1, 1, 480, 360]; 0~1 float, mostly 0, sums up to 2028.2290
gt_offset: [1, 2, 480, 360]; 0~1 float, mostly 0, sums up to -468.9375

val_grid:
[array([[478, 184,  30],
        [478, 184,  30],
        [478, 185,  30],
        ...,
        [ 10, 159,   8],
        [ 10, 159,   8],
        [ 10, 159,   8]])]

pt_label: vector of len 123389; 0-19
pt_ints: vector of len 123389; 0-12976138
pt_fea:  [123389, 9], -80 ~ 80, float

"""

def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def SemKITTI2train_single(label):
    return label - 1 # uint8 trick

with torch.no_grad():
    val_vox_fea_ten = val_vox_fea.to(pytorch_device)
    val_vox_label = SemKITTI2train(val_vox_label)
    val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
    val_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in val_grid]
    val_label_tensor=val_vox_label.type(torch.LongTensor).to(pytorch_device)
    val_gt_center_tensor = val_gt_center.to(pytorch_device)
    val_gt_offset_tensor = val_gt_offset.to(pytorch_device)


    torch.cuda.synchronize()
    predict_labels,center,offset = my_model(val_pt_fea_ten, val_grid_ten, val_vox_fea_ten)
    torch.cuda.synchronize()

    """
    predict_labels: torch.Size([1, 19, 480, 360, 32])
    center: torch.Size([1, 1, 480, 360])
    offset: torch.Size([1, 2, 480, 360])
    """