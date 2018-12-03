# AttGAN-PyTorch

A PyTorch implementation of AttGAN - [Arbitrary Facial Attribute Editing: Only Change What You Want](https://arxiv.org/abs/1711.10678)

![Teaser](https://github.com/elvisyjlin/AttGAN-PyTorch/blob/master/pics/teaser.jpg)

Inverting 13 attributes respectively. From left to right: _Input, Reconstruction, Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Bushy_Eyebrows, Eyeglasses, Male, Mouth_Slightly_Open, Mustache, No_Beard, Pale_Skin, Young_

The original TensorFlow version can be found [here](https://github.com/LynnHo/AttGAN-Tensorflow).


## Requirements

* Python 3
* PyTorch 0.4.0
* TensorboardX

```bash
pip3 install -r requirements.txt
```

* Dataset
  * [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset
    * [Images](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNUu0bseoCKuxuI5ZeTl1a/Img?dl=0&preview=img_align_celeba.zip) should be placed in `./data/img_align_celeba/*.jpg`
    * [Attribute labels](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA8YmAHNNU6BEfWMPMfM6r9a/Anno?dl=0&preview=list_attr_celeba.txt) should be placed in `./data/list_attr_celeba.txt`
  * [HD-CelebA](https://github.com/LynnHo/HD-CelebA-Cropper) (optional)
    * Please see [here](https://github.com/LynnHo/HD-CelebA-Cropper).
* [Pretrained models](https://drive.google.com/open?id=1_E5YCb4XOTZpt6KBwBzSaJdofoqPViN8): download the models you need and unzip the files to `./output/` as below,
  ```text
  output
  ├── 128_shortcut1_inject0_none
  └── 128_shortcut1_inject1_none
  ```

## Usage

To train an AttGAN

```bash
CUDA_VISIBLE_DEVICES=0 \ 
python train.py \ 
--img_size 128 \ 
--shortcut_layers 1 \ 
--inject_layers 1 \ 
--experiment_name 128_shortcut1_inject1_none \ 
--gpu
```

To visualize training details

```bash
tensorboard \
--logdir ./output
```

To test with single attribute editing

![Test](https://github.com/elvisyjlin/AttGAN-PyTorch/blob/master/pics/sample_testing.jpg)

```bash
CUDA_VISIBLE_DEVICES=0 \ 
python test.py \ 
--experiment_name 128_shortcut1_inject1_none \ 
--test_int 1.0 \ 
--gpu
```

To test with multiple attributes editing

![Test Multi](https://github.com/elvisyjlin/AttGAN-PyTorch/blob/master/pics/sample_testing_multi.jpg)

```bash
CUDA_VISIBLE_DEVICES=0 \ 
python test_multi.py \ 
--experiment_name 128_shortcut1_inject1_none \ 
--test_atts Pale_Skin Male \ 
--test_ints 0.5 0.5 \ 
--gpu
```

To test with attribute intensity control

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