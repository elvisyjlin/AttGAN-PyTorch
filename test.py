# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Entry point for testing AttGAN network."""

import argparse
import json
import os
from os.path import join

import torch
import torch.utils.data as data
import torchvision.utils as vutils

from attgan import AttGAN
from data import check_attribute_conflict
from helpers import Progressbar
from utils import find_model


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', required=True)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--num_test', dest='num_test', type=int)
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--custom_img', action='store_true')
    parser.add_argument('--custom_data', type=str, default='./data/custom')
    parser.add_argument('--custom_attr', type=str, default='./data/list_attr_custom.txt')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    return parser.parse_args(args)

args_ = parse()
print(args_)

with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

args.test_int = args_.test_int
args.num_test = args_.num_test
args.gpu = args_.gpu
args.load_epoch = args_.load_epoch
args.multi_gpu = args_.multi_gpu
args.custom_img = args_.custom_img
args.custom_data = args_.custom_data
args.custom_attr = args_.custom_attr
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)

print(args)


if args.custom_img:
    output_path = join('output', args.experiment_name, 'custom_testing')
    from data import Custom
    test_dataset = Custom(args.custom_data, args.custom_attr, args.img_size, args.attrs)
else:
    output_path = join('output', args.experiment_name, 'sample_testing')
    if args.data == 'CelebA':
        from data import CelebA
        test_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'test', args.attrs)
    if args.data == 'CelebA-HQ':
        from data import CelebA_HQ
        test_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'test', args.attrs)
os.makedirs(output_path, exist_ok=True)
test_dataloader = data.DataLoader(
    test_dataset, batch_size=1, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)
if args.num_test is None:
    print('Testing images:', len(test_dataset))
else:
    print('Testing images:', min(len(test_dataset), args.num_test))


attgan = AttGAN(args)
attgan.load(find_model(join('output', args.experiment_name, 'checkpoint'), args.load_epoch))
progressbar = Progressbar()

attgan.eval()
for idx, (img_a, att_a) in enumerate(test_dataloader):
    if args.num_test is not None and idx == args.num_test:
        break
    
    img_a = img_a.cuda() if args.gpu else img_a
    att_a = att_a.cuda() if args.gpu else att_a
    att_a = att_a.type(torch.float)
    
    att_b_list = [att_a]
    for i in range(args.n_attrs):
        tmp = att_a.clone()
        tmp[:, i] = 1 - tmp[:, i]
        tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
        att_b_list.append(tmp)

    with torch.no_grad():
        samples = [img_a]
        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
            samples.append(attgan.G(img_a, att_b_))
        samples = torch.cat(samples, dim=3)
        if args.custom_img:
            out_file = test_dataset.images[idx]
        else:
            out_file = '{:06d}.jpg'.format(idx + 182638)
        vutils.save_image(
            samples, join(output_path, out_file),
            nrow=1, normalize=True, range=(-1., 1.)
        )
        print('{:s} done!'.format(out_file))
