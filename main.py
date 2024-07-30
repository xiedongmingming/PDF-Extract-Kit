import os
import json
import yaml
import pytz
import datetime
import argparse
import logging
import uvicorn

from PIL import Image

from torch.utils.data import Dataset

from ultralytics import YOLO

from unimernet.common.config import Config

import unimernet.tasks as tasks

from unimernet.processors import load_processor

from modules.extract_pdf import load_pdf_fitz, load_image_fitz
from modules.layoutlmv3.model_init import Layoutlmv3_Predictor
from modules.post_process import get_croped_image

from fastapi import FastAPI, UploadFile, File

logger = logging.getLogger(__name__)


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


mfd_model = None

layout_model = None

#################################################################################
app = FastAPI()


@app.get("/")
def index():
    #
    global mfd_model
    global layout_model

    get_mfd_model()
    get_layout_model()

    return {"Hello": "World"}


@app.post("/parse/file")
async def file_parse(file: UploadFile = File(...)):
    #
    global mfd_model
    global layout_model

    get_mfd_model()
    get_layout_model()

    logger.info("-------------> 处理识别请求：{}".format(file.filename))

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

    all_pdfs = [save_file]

    print("total files:", len(all_pdfs))

    try:

        img_list = load_pdf_fitz(save_file, dpi=dpi)  # 将PDF读取成图片（2倍尺寸）

    except:

        img_list = None

        print("unexpected pdf file:", save_file)

    if img_list is None:
        #
        return

    print("pdf index:", 0, "pages:", len(img_list))

    ####################################################################################
    # LAYOUT检测 + 公式检测
    doc_layout_result = []

    latex_filling_list = []

    mf_image_list = []

    for idx, image in enumerate(img_list):

        img_H, img_W = image.shape[0], image.shape[1]

        layout_res = layout_model(image, ignore_catids=[])

        ####################################################################################
        # 公式检测
        mfd_res = mfd_model.predict(image, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=True)[0]

        for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()):
            #
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]

            new_item = {
                'category_id': 13 + int(cla.item()),
                'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                'score': round(float(conf.item()), 2),
                'latex': '',
            }

            layout_res['layout_dets'].append(new_item)

            latex_filling_list.append(new_item)

            bbox_img = get_croped_image(Image.fromarray(image), [xmin, ymin, xmax, ymax])

            mf_image_list.append(bbox_img)

        layout_res['page_info'] = dict(
            page_no=idx,
            height=img_H,
            width=img_W
        )

        doc_layout_result.append(layout_res)

    logger.info("处理结果：{}，{}".format(file.filename, json.dumps(doc_layout_result)))

    return doc_layout_result

@app.post("/parse/page")
async def page_parse(file: UploadFile = File(...)):
    #
    global mfd_model
    global layout_model

    get_mfd_model()
    get_layout_model()

    logger.info("-------------> 处理识别请求：{}".format(file.filename))

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
    ####################################################################################

    img_list = load_image_fitz(save_file)

    print("total files:", len(img_list))

    ####################################################################################
    # LAYOUT检测 + 公式检测
    doc_layout_result = []

    latex_filling_list = []

    mf_image_list = []

    for idx, image in enumerate(img_list):

        img_H, img_W = image.shape[0], image.shape[1]

        layout_res = layout_model(image, ignore_catids=[])

        ####################################################################################
        # 公式检测
        mfd_res = mfd_model.predict(image, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=True)[0]

        for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()):
            #
            xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]

            new_item = {
                'category_id': 13 + int(cla.item()),
                'poly': [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                'score': round(float(conf.item()), 2),
                'latex': '',
            }

            layout_res['layout_dets'].append(new_item)

            latex_filling_list.append(new_item)

            bbox_img = get_croped_image(Image.fromarray(image), [xmin, ymin, xmax, ymax])

            mf_image_list.append(bbox_img)

        layout_res['page_info'] = dict(
            page_no=idx,
            height=img_H,
            width=img_W
        )

        doc_layout_result.append(layout_res)

    logger.info("处理结果：{}，{}".format(file.filename, json.dumps(doc_layout_result)))

    return doc_layout_result

if __name__ == '__main__':
    #
    uvicorn.run("main:app", host="0.0.0.0", port=9000)
