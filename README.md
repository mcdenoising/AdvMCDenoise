# Instructions to use the code.

This code relies a lot on the projects GauGAN, SRGAN, BasicSR,pix2pix. Credit to these pytorch projects.
Some preprocessing code for Tungsten scenes credits to Disney [KPCN](http://cvc.ucsb.edu/graphics/Papers/SIGGRAPH2017_KPCN/) project.


# model weights
**experiments** dir can be downloaded from [googledrive](https://drive.google.com/open?id=1ql_ti30l4UUcLv3W_ooAp1WXpR5ibJZC)   


We provide two set of model weights. 
"./experiment/kjl/models/opt_XX.pth" is trained from KJL large indoor room datasets without seperating Specular and Diffuse components. You can finetune the weigths to your own dataset based on this pretrained weights.

"./experiments/tungsten_diffuse/models/opt_XX.pth" and "./experiments/tungsten_specular/models/opt_XX.pth" are finetuned on [Tungsten](https://github.com/tunabrain/tungsten) scenes. 





# data
**data** dir can be downloaded from [googledrive](https://drive.google.com/open?id=1aWCbbUqkdxvNl_VZvzomaPKFsvEfAdOw)

This directory contain some samples from Tungsten scenes.
It also provides the utility scripts to do data processing, for example, to process EXR files, Tungsten data pre-processing ect..

Large scale indoor dataset from Kujiale.com will be published soon.

# runscripts
**script** dir contains json files for training/testing  

**Attention!!!   Change the settings in json files to include your own data path and project root
Specifically, remember to change the "dataroot_NOISY" "dataroot_GT" "root" "val_root" to your paths**


## to run 

#### training

```
python train_diffuse.py -opt script/train/train_seperate_denoiser_diffuse.json
```

```
python train_specular.py -opt script/train/train_seperate_denoiser_specular.json
```

#### testing

```
python test_diffuse.py -opt script/test/test_seperate_denoiser_diffuse.json
```


```
python test_specular.py -opt script/test/test_seperate_denoiser_specular.json
```
