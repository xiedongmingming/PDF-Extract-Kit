# [dit: self-supervised pre-training for document image transformer](https://arxiv.org/abs/2203.02378)

dit (document image transformer) is a self-supervised pre-trained document image transformer model using large-scale unlabeled text images for document ai tasks, which is essential since no supervised counterparts ever exist due to the lack of human labeled document images. 

<div align="center">
	<img src="https://user-images.githubusercontent.com/45008728/157173825-0949218a-61f5-4acb-949b-bbc499ab49f2.png" width="500" /><img src="https://user-images.githubusercontent.com/45008728/157173843-796dc878-2607-48d7-85cb-f54a2c007687.png" width="500"/>
    <br/>model outputs with publaynet (left) and icdar 2019 ctdar (right)</div>


## what's new
- demos on huggingface: [document layout analysis](https://huggingface.co/spaces/nielsr/dit-document-layout-analysis), [document image classification](https://huggingface.co/spaces/microsoft/document-image-transformer) 第一个貌似不可用（已下载到本地）
- march 2022: release pre-trained [checkpoints](https://mail2sysueducn-my.sharepoint.com/:f:/g/personal/huangyp28_mail2_sysu_edu_cn/Eh7si2lpRqZLsmHASlnsw00BewkWLtqtQ3YGw5b0oi4YzQ?e=qKdZ4A) and fine-tuning [checkpoints](https://mail2sysueducn-my.sharepoint.com/:f:/g/personal/huangyp28_mail2_sysu_edu_cn/EsK1oCJRNnZBi_VfswfD2j8BKCMad94BDmJWlX5rKx_L0Q?e=NVDb1k) & codes (dit-base and dit-large)
- march 2022: release preprint in [arxiv](https://arxiv.org/abs/2203.02378)

## [pretrained models](https://mail2sysueducn-my.sharepoint.com/:f:/g/personal/huangyp28_mail2_sysu_edu_cn/Eh7si2lpRqZLsmHASlnsw00BewkWLtqtQ3YGw5b0oi4YzQ?e=qKdZ4A)

we provide two dit weights pretrained on [iit-cdip test collection 1.0](https://dl.acm.org/doi/10.1145/1148170.1148307). the models were pretrained with 224x224 resolution.

- `diT-base`:   #layer=12; hidden=768;    ffn factor=4x; #head=12; patch=16x16 (#parameters: 86m)
- `diT-large`: #layer=24; hidden=1024; ffn factor=4x; #head=16; patch=16x16 (#parameters: 304m)

download checkpoints that are **self-supervised pretrained** on iit-cdip test collection 1.0:
- dit-base: [dit_base_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ERfMnkSl2mZFlv68E6xxhH4BvJVQOYcwpna3QMbbbLDQCA?e=tOPZRx)
- dit-large: [dit_large_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/Ec3vqfjSDLBMoVI9WIK6vZUBEtEFlOD163ChojDQzPoUvQ?e=ccEeNc)


## setup

first, clone the repo and install required packages:
```
git clone https://github.com/microsoft/unilm.git

cd unilm/dit

pip install -r requirements.txt
```

the required packages including: [pytorch](https://pytorch.org/) version 1.9.0, [torchvision](https://pytorch.org/vision/stable/index.html) version 0.10.0 and [timm](https://github.com/rwightman/pytorch-image-models) version 0.5.4, etc.

for mixed-precision training, please install [apex](https://github.com/NVIDIA/apex)
```
git clone https://github.com/NVIDIA/apex

cd apex

pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
for object detection, please additionally install detectron2 library and shapely. refer to the [detectron2's install.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

```bash
# install `detectron2`
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# install `shapely`
pip install shapely
```

## fine-tuning on rvl-cdip (document image classification)

we summarize the validation results as follows. we also provide the fine-tuned weights. the detailed instructions to reproduce the results can be found at [`classification/readme.md`](classification/README.md).

| name | initialized checkpoint | resolution | accuracy  | weight |
|------------|:----------------------------------------|:----------:|:-------:|-----|
| dit-base | [dit_base_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ERfMnkSl2mZFlv68E6xxhH4BvJVQOYcwpna3QMbbbLDQCA?e=tOPZRx) | 224x224 | 92.11 | [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/EdZaGg6kCnxAtGyWUFfVQTABbMsp6zcW78JKVpk0XsnEmw) |
| dit-large | [dit_large_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/Ec3vqfjSDLBMoVI9WIK6vZUBEtEFlOD163ChojDQzPoUvQ?e=ccEeNc) | 224x224 | 92.69 | [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/EWHOJYI1bZVKr-saboJTWt8B-cS2RRuCZ-TNEfvcvriG3g?e=ykRYDm) |


## fine-tuning on publaynet (document layout analysis)

we summarize the validation results as follows. we also provide the fine-tuned weights. the detailed instructions to reproduce the results can be found at [`object_detection/readme.md`](object_detection/README.md).

| name | initialized checkpoint | detection algorithm  |  mAP| weight |
|------------|:----------------------------------------|:----------:|-------------------|-----|
| dit-base | [dit_base_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ERfMnkSl2mZFlv68E6xxhH4BvJVQOYcwpna3QMbbbLDQCA?e=tOPZRx) | Mask R-CNN | 0.935 |  [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/EdRCgV60BxNJkMW-DUwg5toBYrfZ13Taiuwidx24LL9uCQ?e=zXH3et) |
| dit-large | [dit_large_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/Ec3vqfjSDLBMoVI9WIK6vZUBEtEFlOD163ChojDQzPoUvQ?e=ccEeNc) | Mask R-CNN | 0.941 |  [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/EbiF6b0pushDuMGeL13uEFMBI-tO8UdGuOaKF1jAJhqjWg?e=hKhaL7) |
| dit-base | [dit_base_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ERfMnkSl2mZFlv68E6xxhH4BvJVQOYcwpna3QMbbbLDQCA?e=tOPZRx) | Cascade R-CNN | 0.945 |  [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/Ebu8BdNRx6ZBp4t1KHWyy8cBeytG3evrVm8ae7XVf8Ox0A?e=hlfaML) |
| dit-large | [dit_large_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/Ec3vqfjSDLBMoVI9WIK6vZUBEtEFlOD163ChojDQzPoUvQ?e=ccEeNc) | Cascade R-CNN | 0.949 |  [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ESKnk2I_O09Em52V1xb2Ux0BrO_Z-7cuzL3H1KQRxipb7Q?e=iqTfGc) |

## fine-tuning on icdar 2019 ctdar (table detection)

we summarize the validation results as follows. we also provide the fine-tuned weights. the detailed instructions to reproduce the results can be found at [`object_detection/readme.md`](object_detection/README.md).

**modern**

| name | initialized checkpoint | detection algorithm  |  Weighted Average F1 | weight |
|------------|:----------------------------------------|:----------:|-------------------|-----|
| dit-base | [dit_base_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ERfMnkSl2mZFlv68E6xxhH4BvJVQOYcwpna3QMbbbLDQCA?e=tOPZRx) | Mask R-CNN | 94.74 |  [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/EfJ-YwqPB99FjDL4M-Vc8SgBDLlU3A-YOVlmGy6Cpkwc4A?e=2hr83V) |
| dit-large | [dit_large_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/Ec3vqfjSDLBMoVI9WIK6vZUBEtEFlOD163ChojDQzPoUvQ?e=ccEeNc) | Mask R-CNN | 95.50 |  [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/EfzBky0q6ulGr5AWFDMJOAUBkQu1HWCfeL8Cuuqvdcb5ow?e=tZysTu) |
| dit-base | [dit_base_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ERfMnkSl2mZFlv68E6xxhH4BvJVQOYcwpna3QMbbbLDQCA?e=tOPZRx) | Cascade R-CNN | 95.85 |  [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/EeK-N5V6FFhKkReVk6VPWg8Be_PRySGwSBSUyNVjz8Djew?e=fnVxUg) |
| dit-large | [dit_large_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/Ec3vqfjSDLBMoVI9WIK6vZUBEtEFlOD163ChojDQzPoUvQ?e=ccEeNc) | Cascade R-CNN | 96.29 |  [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/EWZYlqMX8HpMoSTkdBOBCY0BWtYlXPfXfWavDsXUjG1xWg?e=dR4YUc) |

**archival**

| name | initialized checkpoint | detection algorithm  |  Weighted Average F1 | weight |
|------------|:----------------------------------------|:----------:|-------------------|-----|
| dit-base | [dit_base_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ERfMnkSl2mZFlv68E6xxhH4BvJVQOYcwpna3QMbbbLDQCA?e=tOPZRx) | Mask R-CNN | 96.24 |  [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/EWAzBbNFunZBvPXVo7oeLK0B52fqOofE60JhzYVmEo4SSg?e=IOQgbl) |
| dit-large | [dit_large_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/Ec3vqfjSDLBMoVI9WIK6vZUBEtEFlOD163ChojDQzPoUvQ?e=ccEeNc) | Mask R-CNN | 96.46 |  [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ETI0TwK9MsdBlK5k2KDPRzcBDxpQ1tlS6H2qS94SI0s2rQ?e=fx3htJ) |
| dit-base | [dit_base_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ERfMnkSl2mZFlv68E6xxhH4BvJVQOYcwpna3QMbbbLDQCA?e=tOPZRx) | Cascade R-CNN | 96.63 |  [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/EWmpA4gS58tOj8kPmIl72ZQBWh11eqsjRKR30ZGTRLYbOg?e=yXdfp8) |
| dit-large | [dit_large_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/Ec3vqfjSDLBMoVI9WIK6vZUBEtEFlOD163ChojDQzPoUvQ?e=ccEeNc) | Cascade R-CNN | 97.00 |  [link](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/EQY8PZMPCIVKrtJuwJOdSRABOwwsP937dr0EHKCILhCOgQ?e=zeCuNa) |

**combined (combine the inference results of modern and archival)**

| name | initialized checkpoint | detection algorithm  |  Weighted Average F1 | weight |
|------------|:----------------------------------------|:----------:|-------------------|-----|
| diT-tase | [dit_base_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ERfMnkSl2mZFlv68E6xxhH4BvJVQOYcwpna3QMbbbLDQCA?e=tOPZRx) | Mask R-CNN | 95.30 |  - |
| dit-large | [dit_large_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/Ec3vqfjSDLBMoVI9WIK6vZUBEtEFlOD163ChojDQzPoUvQ?e=ccEeNc) | Mask R-CNN | 95.85 |  - |
| dit-base | [dit_base_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ERfMnkSl2mZFlv68E6xxhH4BvJVQOYcwpna3QMbbbLDQCA?e=tOPZRx) | Cascade R-CNN | 96.14 |  - |
| dit-large | [dit_large_patch16_224](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/Ec3vqfjSDLBMoVI9WIK6vZUBEtEFlOD163ChojDQzPoUvQ?e=ccEeNc) | Cascade R-CNN | 96.55 |  - |


## citation

if you find this repository useful, please consider citing our work:
```
@misc{li2022dit,
    title={DiT: Self-supervised Pre-training for Document Image Transformer},
    author={Junlong Li and Yiheng Xu and Tengchao Lv and Lei Cui and Cha Zhang and Furu Wei},
    year={2022},
    eprint={2203.02378},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


## acknowledgement

this repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, the [detectron2](https://github.com/facebookresearch/detectron2) library, the [deit](https://github.com/facebookresearch/deit) repository, the [dino](https://github.com/facebookresearch/dino) repository, the [beit](https://github.com/microsoft/unilm/tree/master/beit) repository and the [mpvit](https://github.com/youngwanLEE/MPViT) repository.


### contact information

for help or issues using dit models, please submit a github issue.

for other communications related to dit, please contact lei cui (`lecu@microsoft.com`), furu wei (`fuwei@microsoft.com`).
