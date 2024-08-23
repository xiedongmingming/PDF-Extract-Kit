#!/usr/bin/env python
# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# --------------------------------------------------------------------------------

"""
Detection Training Script for MPViT.
"""

import json
import os
import itertools

import torch

from typing import Any, Dict, List, Set

from detectron2.data import build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import detection_utils as utils

from ditod import add_vit_config
from ditod import DetrDatasetMapper

from detectron2.data.datasets import register_coco_instances
import logging
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.engine.defaults import create_ddp_model
import weakref
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from ditod import MyDetectionCheckpointer, ICDAREvaluator
from ditod import MyTrainer
from detectron2.engine import DefaultPredictor # 源码调整 -- 单独引用（之前和上面的混合在一起了）

from detectron2.structures import BoxMode

from PIL import Image, ImageDraw

import numpy as np

from detectron2.evaluation.coco_evaluation import instances_to_coco_json


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # add_coat_config(cfg)
    add_vit_config(cfg)

    cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    cfg.freeze()

    default_setup(cfg, args)

    return cfg


def parser_instance(instances, img_width, img_height):

    category_id_maping = [
        {
            "supercategory": "",
            "id": 1,
            "name": "Text"
        },
        {
            "supercategory": "",
            "id": 2,
            "name": "Title"
        },
        {
            "supercategory": "",
            "id": 3,
            "name": "List"
        },
        {
            "supercategory": "",
            "id": 4,
            "name": "Table"
        },
        {
            "supercategory": "",
            "id": 5,
            "name": "Figure"
        }
    ]

    num_instance = len(instances)

    if num_instance == 0:

        return []

    boxes = instances.pred_boxes.tensor.cpu().numpy()  # 源码调整：tensor.numpy()->tensor.cpu().numpy()

    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)

    boxes = boxes.tolist()

    scores = instances.scores.tolist()

    classes = instances.pred_classes.tolist()

    results = []

    for k in range(num_instance):

        output_label = category_id_maping[classes[k] - 1]["name"]

        bbox = boxes[k]

        x, y, width, heigh = bbox[:4]

        result = {
            'from_name': "label",
            'to_name': "image",
            'type': 'rectanglelabels',
            "value": {
                'rectanglelabels': [output_label],
                'x': float(x) / img_width * 100,
                'y': float(y) / img_height * 100,
                'width': float(width) / img_width * 100,
                'height': float(heigh) / img_height * 100
            },
            "score": scores[k],
        }

        results.append(
            result
        )

    avg_score = sum(scores) / max(1.0, len(scores))

    return [{
        'result': results,
        'score': avg_score
    }]


def main(args):

    cfg = setup(args)

    """
    register publaynet first
    """
    register_coco_instances(
        "publaynet_train",
        {},
        cfg.PUBLAYNET_DATA_DIR_TRAIN + ".json",
        cfg.PUBLAYNET_DATA_DIR_TRAIN
    )

    register_coco_instances(
        "publaynet_val",
        {},
        cfg.PUBLAYNET_DATA_DIR_TEST + ".json",
        cfg.PUBLAYNET_DATA_DIR_TEST
    )

    register_coco_instances(
        "icdar2019_train",
        {},
        cfg.ICDAR_DATA_DIR_TRAIN + ".json",
        cfg.ICDAR_DATA_DIR_TRAIN
    )

    register_coco_instances(
        "icdar2019_test",
        {},
        cfg.ICDAR_DATA_DIR_TEST + ".json",
        cfg.ICDAR_DATA_DIR_TEST
    )

    image_path = args.input

    original_image = Image.open(image_path)

    img_width, img_height = original_image.width, original_image.height

    image = np.asarray(original_image)  # {ndarray: (1754, 1240, 3)}

    model = DefaultPredictor(cfg)

    res = model(image)

    ret_res = parser_instance(res["instances"], img_width, img_height)

    print(json.dumps(ret_res, indent=2))

    ########################################################################################
    instances = res["instances"]

    draw = ImageDraw.Draw(original_image)

    boxes = []

    for index in range(0, len(instances)):
        #
        boxes.append({
            'box': instances[index].pred_boxes.tensor.cpu().numpy()[0],
            'score': instances[index].scores.data.cpu().numpy()[0],
            'class': instances[index].pred_classes.cpu().numpy()[0],
        })

    color = 50

    for box in boxes:

        if color > 255:
            #
            color = 50

        draw.rectangle(xy=(
            box['box'][0],
            box['box'][1],
            box['box'][2],
            box['box'][3]
        ),
            fill=None,
            outline=str(hex(color * 255 * 255 + color * 255 + color)).replace("0x", "#"),  # outline="red",
            width=1
        )

        color += 30

    original_image.save("./images-test.png")

    return ret_res


if __name__ == "__main__":

    parser = default_argument_parser()

    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--input", help="image path")

    args = parser.parse_args()

    print("Command Line Args:", args)

    if args.debug:
        #
        import debugpy

        # 允许DEBUG-SH启动的程序了。
        print("Enabling attach starts.")

        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()

        print("Enabling attach ends.")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

# python train_net.py --config-file cascade_layoutlmv3.yaml --num-gpus 0 MODEL.WEIGHTS ./publaynet/model_final.pth OUTPUT_DIR output --eval-only PUBLAYNET_DATA_DIR_TEST /media/liukun/7764-4284/ai/publaynet/val
# python infer_net.py --config-file cascade_layoutlmv3.yaml --num-gpus 0 MODEL.WEIGHTS ./publaynet/model_final.pth OUTPUT_DIR output --input C:\Users\XM\Desktop\demo\png\xxxx-000003.png