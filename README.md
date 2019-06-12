# AttGAN-PyTorch

A PyTorch implementation of AttGAN - [Arbitrary Facial Attribute Editing: Only Change What You Want](https://arxiv.org/abs/1711.10678)

![Teaser](https://github.com/elvisyjlin/AttGAN-PyTorch/blob/master/pics/teaser.jpg)
Test on the CelebA validating set

![Custom](https://github.com/elvisyjlin/AttGAN-PyTorch/blob/master/pics/custom_images.jpg)
Test on my custom set

Inverting 13 attributes respectively. From left to right: _Input, Reconstruction, Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Bushy_Eyebrows, Eyeglasses, Male, Mouth_Slightly_Open, Mustache, No_Beard, Pale_Skin, Young_

The original TensorFlow version can be found [here](https://github.com/LynnHo/AttGAN-Tensorflow).


## Requirements

* Python 3
* PyTorch 0.4.0
* TensorboardX

```bash
pip3 install -r requirements.txt
```

If you'd like to train with __multiple GPUs__, please install PyTorch __v0.4.0__ instead of v1.0.0 or above. The so-called stable version of PyTorch has a bunch of problems with regard to `nn.DataParallel()`. E.g. https://github.com/pytorch/pytorch/issues/15716, https://github.com/pytorch/pytorch/issues/16532, etc.

```bash
pip3 install --upgrade torch==0.4.0
```

* Dataset
  * [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
    * [Images](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNUu0bseoCKuxuI5ZeTl1a/Img?dl=0&preview=img_align_celeba.zip) should be placed in `./data/img_align_celeba/*.jpg`
    * [Attribute labels](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA8YmAHNNU6BEfWMPMfM6r9a/Anno?dl=0&preview=list_attr_celeba.txt) should be placed in `./data/list_attr_celeba.txt`
  * [HD-CelebA](https://github.com/LynnHo/HD-CelebA-Cropper) (optional)
    * Please see [here](https://github.com/LynnHo/HD-CelebA-Cropper).
  * [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) dataset (optional)
    * Please see [here](https://github.com/willylulu/celeba-hq-modified).
    * _Images_ should be placed in `./data/celeba-hq/celeba-*/*.jpg`
    * _Image list_ should be placed in `./data/image_list.txt`
* [Pretrained models](http://bit.ly/attgan-pretrain): download the models you need and unzip the files to `./output/` as below,
  ```text
  output
  ├── 128_shortcut1_inject0_none
  ├── 128_shortcut1_inject1_none
  ├── 256_shortcut1_inject0_none
  ├── 256_shortcut1_inject1_none
  ├── 256_shortcut1_inject0_none_hq
  ├── 256_shortcut1_inject1_none_hq
  ├── 384_shortcut1_inject0_none_hq
  └── 384_shortcut1_inject1_none_hq
  ```

## Usage

#### To train an AttGAN on CelebA 128x128

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--img_size 128 \
--shortcut_layers 1 \
--inject_layers 1 \
--experiment_name 128_shortcut1_inject1_none \
--gpu
```

#### To train an AttGAN on CelebA-HQ 256x256 with multiple GPUs

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--data CelebA-HQ \
--img_size 256 \
--shortcut_layers 1 \
--inject_layers 1 \
--experiment_name 256_shortcut1_inject1_none_hq \
--gpu \
--multi_gpu
```

#### To visualize training details

```bash
tensorboard \
--logdir ./output
```

#### To test with single attribute editing

![Test](https://github.com/elvisyjlin/AttGAN-PyTorch/blob/master/pics/sample_testing.jpg)

```bash
CUDA_VISIBLE_DEVICES=0 \
python test.py \
--experiment_name 128_shortcut1_inject1_none \
--test_int 1.0 \
--gpu
```

#### To test with multiple attributes editing

![Test Multi](https://github.com/elvisyjlin/AttGAN-PyTorch/blob/master/pics/sample_testing_multi.jpg)

```bash
CUDA_VISIBLE_DEVICES=0 \
python test_multi.py \
--experiment_name 128_shortcut1_inject1_none \
--test_atts Pale_Skin Male \
--test_ints 0.5 0.5 \
--gpu
```

#### To test with attribute intensity control

![Test Slide](https://github.com/elvisyjlin/AttGAN-PyTorch/blob/master/pics/sample_testing_slide.jpg)

```bash
CUDA_VISIBLE_DEVICES=0 \
python test_slide.py \
--experiment_name 128_shortcut1_inject1_none \
--test_att Male \
--test_int_min -1.0 \
--test_int_max 1.0 \
--n_slide 10 \
--gpu
```

#### To test with your custom images (supports `test.py`, `test_multi.py`, `test_slide.py`)

```bash
CUDA_VISIBLE_DEVICES=0 \
python test.py \
--experiment_name 384_shortcut1_inject1_none_hq \
--test_int 1.0 \
--gpu \
--custom_img
```

Your custom images are supposed to be in `./data/custom` and you also need an attribute list of the images `./data/list_attr_custom.txt`. Please crop and resize them into square images in advance.