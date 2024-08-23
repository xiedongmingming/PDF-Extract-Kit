# dit for object detection

this folder contains mask r-cnn cascade mask r-cnn running instructions on top of [detectron2](https://github.com/facebookresearch/detectron2) for publaynet and icdar 2019 ctdar.

## usage

### inference

the quickest way to try out dit for document layout analysis is the web demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/nielsr/dit-document-layout-analysis).

one can run inference using the `inference.py` script. it can be run as follows (from the root of the unilm repository):

```
python ./dit/object_detection/inference.py \
--image_path ./dit/object_detection/publaynet_example.jpeg \
--output_file_name output.jpg \
--config ./dit/object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml \
--opts MODEL.WEIGHTS https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_mrcnn.pth \
```

make sure that the configuration file (yaml) and pytorch checkpoint match. the example above uses dit-base with the mask r-cnn framework fine-tuned on publaynet.

### data preparation

**publaynet**

download the data from this [link](https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz?_ga=2.218138265.1825957955.1646384196-1495010506.1633610665) (~96GB). then extract it to `PATH-to-PubLayNet`.

a soft link needs to be created to make the data accessible for the program:`ln -s PATH-to-PubLayNet publaynet_data`.

**icdar 2019 ctdar**

Download the data from this [link](https://github.com/cndplab-founder/ICDAR2019_cTDaR) (~4GB). Assume path to this repository is named as `PATH-to-ICDARrepo`.

Then run `python convert_to_coco_format.py --root_dir=PATH-to-ICDARrepo --target_dir=PATH-toICDAR`. Now the path to processed data is `PATH-to-ICDAR`.

Run the following command to get the adaptively binarized images for archival subset.

```
cp -r PATH-to-ICDAR/trackA_archival PATH-to-ICDAR/at_trackA_archival
python adaptive_binarize.py --root_dir PATH-to-ICDAR/at_trackA_archival
```

The binarized archival subset will be in `PATH-to-ICDAR/at_trackA_archival`.

According to the subset you want to evaluate/fine-tune, a soft link should be created:`ln -s PATH-to-ICDAR/trackA_modern data` or `ln -s PATH-to-ICDAR/at_trackA_archival data`.

### evaluation

Following commands provide two examples to evaluate the fine-tuned checkpoints.

The config files can be found in `icdar19_configs` and `publaynet_configs`.

1) Evaluate the fine-tuned checkpoint of DiT-Base with Mask R-CNN on PublayNet:
```bash
python train_net.py --config-file publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml --eval-only --num-gpus 8 MODEL.WEIGHTS <finetuned_checkpoint_file_path or link> OUTPUT_DIR <your_output_dir> 
```

2) Evaluate the fine-tuned checkpoint of DiT-Large with Cascade Mask R-CNN on ICDAR 2019 cTDaR archival subset (make sure you have created a soft link from `PATH-to-ICDAR/at_trackA_archival` to `data`):
```bash
python train_net.py --config-file icdar19_configs/cascade/cascade_dit_large.yaml --eval-only --num-gpus 8 MODEL.WEIGHTS <finetuned_checkpoint_file_path or link> OUTPUT_DIR <your_output_dir> 
```

**Note**: We have fixed the **bug** in the [ICDAR2019 measurement tool](https://github.com/cndplab-founder/ctdar_measurement_tool) during integrating the tool into our code. If you use the tool to get the evaluation score, please modify the [code](https://github.com/cndplab-founder/ctdar_measurement_tool/blob/738456d3164a838ffaeefe7d1b5e64f3a4368a0e/evaluate.py#L146
) as follows:
```bash
    ...
    # print(each_file)

# for file in gt_file_lst:
#     if file.split(".") != "xml":
#         gt_file_lst.remove(file)
#     # print(gt_file_lst)

# Comment the code above and add the code below
for i in range(len(gt_file_lst) - 1, -1, -1):
    if gt_file_lst[i].split(".")[-1] != "xml":
        del gt_file_lst[i]

if len(gt_file_lst) > 0:
    ...
```

### training
the following commands provide two examples to train the mask r-cnn/cascade mask r-cnn with dit backbone on 8 32gb nvidia v100 gpus.

1) fine-tune dit-base with cascade mask r-cnn on publaynet:
```bash
python train_net.py --config-file publaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 8 MODEL.WEIGHTS <DiT-Base_file_path or link> OUTPUT_DIR <your_output_dir> 
```


2) fine-tune dit-large with mask r-cnn on icdar 2019 ctdar modern:
```bash
python train_net.py --config-file icdar19_configs/markrcnn/maskrcnn_dit_large.yaml --num-gpus 8 MODEL.WEIGHTS <DiT-Large_file_path or link> OUTPUT_DIR <your_output_dir> 
```



[detectron2's document](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) may help you for more details.


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



## acknowledgment
thanks to [detectron2](https://github.com/facebookresearch/detectron2) for mask r-cnn and cascade mask r-cnn implementation.
