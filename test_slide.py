import argparse
import json
import os
from os.path import join
from glob import glob

import torch
import torch.utils.data as data
import torchvision.utils as vutils

from attgan import AttGAN
from data import CelebA, CelebA_HQ, check_attribute_conflict
from helpers import Progressbar


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', required=True)
    parser.add_argument('--test_att', dest='test_att', help='test_att')
    parser.add_argument('--test_int_min', dest='test_int_min', type=float, default=-1.0, help='test_int_min')
    parser.add_argument('--test_int_max', dest='test_int_max', type=float, default=1.0, help='test_int_max')
    parser.add_argument('--n_slide', dest='n_slide', type=int, default=10, help='n_slide')
    parser.add_argument('--num_test', dest='num_test', type=int)
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args(args)

def find_model(path, epoch='latest'):
    if epoch == 'latest':
        files = glob(join(path, '*.pth'))
        file = sorted(files, key=lambda x: int(x.rsplit('.', 2)[1]))[-1]
    else:
        file = join(path, 'weights.{:d}.pth'.format(int(epoch)))
    assert os.path.exists(file), 'File not found: ' + file
    print('Find model of {} epoch: {}'.format(epoch, file))
    return file

args_ = parse()
print(args_)

with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

args.test_att = args_.test_att
args.test_int_min = args_.test_int_min
args.test_int_max = args_.test_int_max
args.n_slide = args_.n_slide
args.num_test = args_.num_test
args.load_epoch = args_.load_epoch
args.gpu = args_.gpu
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)

assert args.test_att is not None, 'test_att should be chosen in %s' % (str(args.attrs))

output_path = join('output', args.experiment_name, 'sample_testing_slide_' + args.test_att)
os.makedirs(output_path, exist_ok=True)

if args.data == 'CelebA':
    test_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'test', args.attrs)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=1, num_workers=args.num_workers, 
        shuffle=False, drop_last=False
    )
if args.data == 'CelebA-HQ':
    test_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'test', args.attrs)
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
    att_b = att_a.clone()

    with torch.no_grad():
        samples = [img_a]
        for i in range(args.n_slide):
            test_int = (args.test_int_max - args.test_int_min) / (args.n_slide - 1) * i + args.test_int_min
            att_b_ = (att_b * 2 - 1) * args.thres_int
            att_b_[..., args.attrs.index(args.test_att)] = test_int
            samples.append(attgan.G(img_a, att_b_))
        samples = torch.cat(samples, dim=3)
        vutils.save_image(
            samples, output_path + '/{:06d}.jpg'.format(idx + 182638), 
            nrow=1, normalize=True, range=(-1., 1.)
        )
        print('{:06d}.jpg done!'.format(idx + 182638))