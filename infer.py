import time
from PIL import Image

# from vietocr.tool.predictor import Predictor
# from vietocr.tool.config import Cfg
from PaddleOCR.paddleocr import PaddleOCR

import numpy as np
from PIL import Image
import pandas as pd
import fitz


# config = Cfg.load_config_from_name('vgg_seq2seq')
# config['predictor']['beamsearch'] = False
# config['device'] = 'cpu'
# config['weights'] = '/content/drive/MyDrive/ Image CVs/vietocr/transformerocr.pth'
# detector = Predictor(config)


ocr = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir='data',
                rec_model_dir='data',
                cls_model_dir='data')


def img2text(img):

    # Input: array image
    # Output: a dataframe with bounding box and text

    # Detect line
    result = ocr.ocr(img, rec=False)
    result.reverse()

    # Get bbox
    xs = []
    ys = []
    ws = []
    hs = []
    list_text = []

    # Create df
    for line in result:
        bbox = (line[0][0]-5, line[0][1]-5, line[2][0]+5, line[2][1]+5)
        xs.append(bbox[0])
        ys.append(bbox[1])
        ws.append(bbox[2])
        hs.append(bbox[3])

        new_image = img.crop(bbox)
        list_text.append(detector.predict(new_image))
    bbox_df = pd.DataFrame(
        {'x': xs, 'y': ys, 'w': ws, 'h': hs, 'text': list_text})
    return bbox_df


def img2text(img):
    # Detect line
    result = ocr.ocr(img, rec=False)
    result.reverse()

    # Get bbox
    xs = []
    ys = []
    ws = []
    hs = []
    list_text = []

    img = Image.fromarray(img)
    # Create df
    for line in result:
        bbox = (line[0][0]-5, line[0][1]-5, line[2][0]+5, line[2][1]+5)
        xs.append(bbox[0])
        ys.append(bbox[1])
        ws.append(bbox[2])
        hs.append(bbox[3])
        new_image = img.crop(bbox)
        list_text.append(detector.predict(new_image))
    bbox_df = pd.DataFrame(
        {'x': xs, 'y': ys, 'w': ws, 'h': hs, 'text': list_text})
    return bbox_df


def pix2np(pix):
    if pix.n > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, pix.n)
    return im


def pdfimage2text(file):
    doc = fitz.Document(file)
    pages = doc.pageCount
    block_df = pd.DataFrame()
    for num_page in range(pages):
        pix = doc[num_page].getPixmap(alpha=False)
        pix = pix2np(pix)
        block_df = pd.concat([block_df, img2text(pix)])
    return block_df


# start = time.time()
# pdf_path = '/content/drive/MyDrive/ Image CVs/cvSample4.pdf'
# print(pdfimage2text(pdf_path))
# print(time.time()-start)
