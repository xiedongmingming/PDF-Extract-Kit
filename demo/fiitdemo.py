import os
import json
import fitz

import yaml
import pytz
import datetime
import argparse
import logging
import uvicorn


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

                        if 'spans' in line and len(line['spans']) > 0:

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

                        else:

                            print(line)

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

if __name__ == "__main__" :
    #

    # 读取json文件内容,返回字典格式
    with open('G:\\datasets\\CDLA_DATASET\\coco\\val.json', 'r', encoding='utf8') as fp:

        json_data = json.load(fp)

        print('这是文件中的json数据：', json_data)
        print('这是读取到文件数据的数据类型：', type(json_data))

    parse3("D:\\临时\\cdd5232eead637f1213953c9279f6f8c.pdf", 144)