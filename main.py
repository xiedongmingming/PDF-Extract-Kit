import os
import json
import fitz

import yaml
import pytz
import datetime
import argparse
import logging
import uvicorn

from copy import deepcopy

from PIL import Image, ImageDraw, ImageFont

from pymupdf import TOOLS, Rect, Point

from torch.utils.data import Dataset

from ultralytics import YOLO

from unimernet.common.config import Config

import unimernet.tasks as tasks

from unimernet.processors import load_processor

from modules.extract_pdf import load_file_fitz, load_image_fitz
from modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from modules.post_process import get_croped_image

from fastapi import FastAPI, UploadFile, File

logger = logging.getLogger(__name__)

TOOLS.set_small_glyph_heights(True)

# TEXT输出：
def parse2(path: str, dpi: int = 72):

    with fitz.open(path) as pdf:

        for page in pdf:

            # 1. 获取文本信息
            words = page.get_text_words()

            # 2. 生成结构体：块->[行]->[索引]
            blocks = {  #

            }

            for w in words:  # 示例：(26.243999481201172, 637.5908203125, 210.36801147460938, 654.8192138671875, '基金项目:国家重点研发计划项目(2018YFC2002204)', 0, 2, 0)

                # 位置信息：w[:4]
                # 文本信息：w[4]
                # 块号信息：w[5]
                # 行号信息：w[6]
                # 索引信息：w[7]

                result = {
                    "x1": w[0],
                    "y1": w[1],
                    "x2": w[2],
                    "y2": w[3],
                    "content": w[4],
                    "block": w[5],
                    "line": w[6],
                    "index": w[7]
                }

                if result["block"] in blocks:
                    if result["line"] in blocks[result["block"]]:
                        blocks[result["block"]][result["line"]].append(result)
                    else:
                        blocks[result["block"]][result["line"]] = [result]
                else:
                    blocks[result["block"]] = {
                        result["line"]: [result]
                    }

                print(json.dumps(result))

            print(blocks)

            # 3. 按行统计方式1 -- 按行统计（行内的索引合并）
            lines1 = []

            for block in blocks.values():

                for line in block.values():

                    x1 = line[0]["x1"]
                    y1 = line[0]["y1"]
                    x2 = line[0]["x2"]
                    y2 = line[0]["y2"]

                    content = line[0]["content"]

                    for data in line[1:]:

                        content += data["content"]

                        if data["x1"] < x1:
                            x1 = data["x1"]
                        if data["y1"] < y1:
                            y1 = data["y1"]
                        if data["x2"] > x2:
                            x2 = data["x2"]
                        if data["y2"] > y2:
                            y2 = data["y2"]

                    lines1.append({
                        "bboxes": [x1, y1, x2, y2],
                        "content": content
                    })

            print(lines1)

            # 4. 按行统计方式2 -- 按块统计（块内的行合并）
            lines2 = []

            num = 0

            for block in blocks.values():

                num += 1

                x1 = float("inf")
                y1 = float("inf")
                x2 = float("-inf")
                y2 = float("-inf")

                content = ""

                for line in block.values():

                    for data in line:

                        content += data["content"]

                        if data["x1"] < x1:
                            x1 = data["x1"]
                        if data["y1"] < y1:
                            y1 = data["y1"]
                        if data["x2"] > x2:
                            x2 = data["x2"]
                        if data["y2"] > y2:
                            y2 = data["y2"]

                lines2.append({
                    "bboxes": [x1, y1, x2, y2],
                    "content": content
                })


            # pages.append({
            #     "width": page.width,
            #     "height": page.height,
            #     "num": page.number,
            #     "lines": lines2
            # })

            print(lines2)

def parse3(path: str, dpi: int = 72):
    #
    with fitz.open(path) as pdf:
        #
        pages = []

        for page in pdf:
            #
            text = page.get_text(option="json")  # {"blocks":[{"lines":["spans":[]]}]}

            lines = []
            images = []

            for block in json.loads(text)['blocks']:
                #
                if block['type'] == 0:  # 文本

                    for line in block['lines']:

                        text = ""

                        font = line['spans'][0]['font']
                        size = line['spans'][0]['size']

                        for span in line['spans']:

                            text += span["text"]

                        if not text.isspace():  # 略过空行

                            bbox = line['bbox']

                            dir = 0

                            if line['dir'][0] == 1 and line['dir'][1] == 0:
                                dir = 0
                            elif line['dir'][0] == 1 and line['dir'][1] == 0:
                                dir = 90
                            elif line['dir'][0] == 1 and line['dir'][1] == 0:
                                dir = 180
                            elif line['dir'][0] == 1 and line['dir'][1] == 0:
                                dir = 270
                            else:
                                dir = 0

                            lines.append({
                                'uuid': "",
                                'score': 1,
                                'page_num': page.number,
                                'content': text,
                                'x1': bbox[0],
                                'y1': bbox[1],
                                'x2': bbox[2],
                                'y2': bbox[3],
                                'height': bbox[3] - bbox[1],
                                'width': bbox[2] - bbox[0],
                                'font_size': size,
                                'font_name': font,
                                'direction': dir
                            })

                elif block['type'] == 1:  # 图片

                    images.append({
                        'uuid': "",
                        'score': 1,
                        'page_num': page.number,
                        'x1': bbox[0],
                        'y1': bbox[1],
                        'x2': bbox[2],
                        'y2': bbox[3],
                        'height': bbox[3] - bbox[1],
                        'width': bbox[2] - bbox[0]
                    })

            pages.append({
                "page_num": page.number,
                "page_width": page.cropbox.width,
                "page_height": page.cropbox.height,
                "texts": lines,
                "images": images
            })

        return pages


def mfd_model_init(weight):
    #
    mfd_model = YOLO(weight)

    return mfd_model


def mfr_model_init(weight_dir, device='cpu'):
    #
    args = argparse.Namespace(cfg_path="modules/UniMERNet/configs/demo.yaml", options=None)

    cfg = Config(args)

    cfg.config.model.pretrained = os.path.join(weight_dir, "pytorch_model.bin")
    cfg.config.model.model_config.model_name = weight_dir
    cfg.config.model.tokenizer_config.path = weight_dir

    task = tasks.setup_task(cfg)

    model = task.build_model(cfg)
    model = model.to(device)

    vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)

    return model, vis_processor


def layout_model_init(weight):
    #
    model = Layoutlmv3_Predictor(weight)

    return model


class MathDataset(Dataset):

    def __init__(self, image_paths, transform=None):

        self.image_paths = image_paths

        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):
        # if not pil image, then convert to pil image
        if isinstance(self.image_paths[idx], str):
            raw_image = Image.open(self.image_paths[idx])
        else:
            raw_image = self.image_paths[idx]

        if self.transform:
            image = self.transform(raw_image)

        return image


tz = pytz.timezone('Asia/Shanghai')

now = datetime.datetime.now(tz)

print(now.strftime('%Y-%m-%d %H:%M:%S'))

# 模型初始化
with open('configs/model_configs.yaml') as f:
    #
    model_configs = yaml.load(f, Loader=yaml.FullLoader)

####################################################################################
# 读取配置文件中的配置参数
img_size = model_configs['model_args']['img_size']

conf_thres = model_configs['model_args']['conf_thres']

iou_thres = model_configs['model_args']['iou_thres']

device = model_configs['model_args']['device']

dpi = model_configs['model_args']['pdf_dpi']

print(now.strftime('%Y-%m-%d %H:%M:%S'))
print('model init done!')


####################################################################################
# 初始化模型
# 1. 公式检测
# mfd_model = mfd_model_init(model_configs['model_args']['mfd_weight'])
# 2. 公式识别
# mfr_model, mfr_vis_processors = mfr_model_init(model_configs['model_args']['mfr_weight'], device=device)
# 3. 公式识别
# mfr_transform = transforms.Compose([mfr_vis_processors, ])
# 4. 布局检测
# layout_model = layout_model_init(model_configs['model_args']['layout_weight'])
# 5. OCR检测
# ocr_model = ModifiedPaddleOCR(show_log=True)

def get_layout_model():
    #
    global layout_model

    if layout_model == None:
        #
        layout_model = layout_model_init(model_configs['model_args']['layout_weight'])

    return layout_model


def get_mfd_model():
    #
    global mfd_model

    if mfd_model == None:
        #
        mfd_model = mfd_model_init(model_configs['model_args']['mfd_weight'])

    return mfd_model

def sorted_layout_boxes(res, w):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        res(list):ppstructure results
    return:
        sorted results(list)
    """
    num_boxes = len(res)

    if num_boxes == 1:

        res[0]["layout"] = "single"

        return res

    sorted_boxes = sorted(res, key=lambda x: (x["poly"][1], x["poly"][0]))  # 先按Y1和X1排序

    _boxes = list(sorted_boxes)

    new_res = []
    res_left = []
    res_right = []

    i = 0

    while True:

        if i >= num_boxes:

            break

        if i == num_boxes - 1: # 最后一个数据

            if (_boxes[i]["poly"][1] > _boxes[i - 1]["poly"][5] and _boxes[i]["poly"][0] < w / 2 and _boxes[i]["poly"][4] > w / 2):

                new_res += res_left
                new_res += res_right

                _boxes[i]["layout"] = "single"

                new_res.append(_boxes[i])

            else:

                if _boxes[i]["poly"][4] > w / 2:

                    _boxes[i]["layout"] = "double"

                    res_right.append(_boxes[i])

                    new_res += res_left
                    new_res += res_right

                elif _boxes[i]["poly"][0] < w / 2:

                    _boxes[i]["layout"] = "double"

                    res_left.append(_boxes[i])

                    new_res += res_left
                    new_res += res_right

            res_left = []
            res_right = []

            break

        elif _boxes[i]["poly"][0] < w / 4 and _boxes[i]["poly"][4] < 3 * w / 4:  # 双列左

            _boxes[i]["layout"] = "double"

            res_left.append(_boxes[i])

            i += 1

        elif _boxes[i]["poly"][0] > w / 4 and _boxes[i]["poly"][4] > w / 2:  # 双列右

            _boxes[i]["layout"] = "double"

            res_right.append(_boxes[i])

            i += 1

        else:  # 单列

            new_res += res_left
            new_res += res_right

            _boxes[i]["layout"] = "single"  # 单列

            new_res.append(_boxes[i])  # 直接添加

            res_left = []
            res_right = []

            i += 1

    if res_left:
        new_res += res_left
    if res_right:
        new_res += res_right

    return new_res

mfd_model = None

layout_model = None

#################################################################################
app = FastAPI()

async def save(file: UploadFile):

    fn = file.filename

    save_path = f'temp/'

    if not os.path.exists(save_path):
        #
        os.mkdir(save_path)

    save_file = os.path.join(save_path, fn)

    f = open(save_file, 'wb')

    data = await file.read()

    f.write(data)

    f.close()

    return save_file

def get_image_list(save_file: str, pdf: bool):

    if pdf:

        logger.info("-------------> 当前处理为PDF文件：{}".format(save_file))

        try:

            img_list = load_file_fitz(save_file, dpi=dpi)  # 将PDF读取成图片（2倍尺寸）

        except:

            logger.error("-------------> 加载PDF文件异常：{}".format(save_file))

            img_list = None

    else:

        logger.info("-------------> 当前处理为PDF页面：{}".format(save_file))

        img_list = load_image_fitz(save_file)

    return img_list

def parse(file: UploadFile, ispdf: bool, save_file: str, img_list: list, mfd_model, layout_model):

    # LAYOUT检测 + 公式检测
    results = []

    if ispdf:
        pages = parse3(save_file)
    else:
        pages = []

    for idx, image in enumerate(img_list):

        img_H, img_W = image.shape[0], image.shape[1]

        regions = layout_model(image, ignore_catids=[])

        ####################################################################################
        # 公式检测
        formulas = mfd_model.predict(image, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=True)[0]

        for xyxy, conf, cla in zip(formulas.boxes.xyxy.cpu(), formulas.boxes.conf.cpu(), formulas.boxes.cls.cpu()):
            #
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]

            item = {
                'category_id': 13 + int(cla.item()),
                'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                'score': round(float(conf.item()), 2),
                'latex': '',
            }

            regions['layout_dets'].append(item)

        if ispdf:
            page = pages[idx]
            page_h = page["page_height"]
            page_w = page["page_width"]
        else:
            page = None
            page_h = 0
            page_w = 0

        # result_cp = deepcopy(layout_res['layout_dets'])
        #
        # result_sorted = sorted_layout_boxes(result_cp, img_W)
        #
        # layout_res['layout_dets'] = result_sorted

        results.append(dict(
            page_no=idx,
            height=img_H,
            width=img_W,
            page_data=page,
            page_height=page_h,
            page_width=page_w,
            regions=regions['layout_dets']
        ))

    logger.info("处理结果：{}，{}".format(file.filename, json.dumps(results)))

    return results


@app.post("/parse/file")
async def file_parse(file: UploadFile = File(...)):
    #
    global mfd_model
    global layout_model

    get_mfd_model()
    get_layout_model()

    logger.info("-------------> 处理识别请求：{}".format(file.filename))

    save_file = await save(file)

    img_list = get_image_list(save_file, True)

    return parse(file, True, save_file, img_list, mfd_model, layout_model)

@app.post("/parse/page")
async def page_parse(file: UploadFile = File(...)):
    #
    global mfd_model
    global layout_model

    get_mfd_model()
    get_layout_model()

    logger.info("-------------> 处理识别请求：{}".format(file.filename))

    save_file = await save(file)

    img_list = get_image_list(save_file, False)

    return parse(file, False, save_file, img_list, mfd_model, layout_model)

if __name__ == '__main__':
    #
    uvicorn.run("main:app", host="0.0.0.0", port=9000)
