import os

import fitz
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from PIL import Image

from vietocr.predictor import Predictor
from vietocr.utils import config

root_data = "data"
config["predictor"]["beamsearch"] = False
config["weights"] = os.path.join(root_data, "transformerocr.pth")


class OCR(object):
    def __init__(self) -> None:
        self.detector = Predictor(config)
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            det_model_dir=root_data,
            rec_model_dir=root_data,
            cls_model_dir=root_data,
            use_gpu=False,
        )

    def img2text(self, img: np.array, page: int) -> pd.DataFrame:
        # Input: array image
        # Output: a dataframe with bounding box and text

        # Detect line
        result = self.ocr.ocr(np.array(img), rec=False)
        result.reverse()

        # Get bbox
        boxes = []
        list_text = []
        img = Image.fromarray(img)
        # Create df
        for line in result:
            bbox = (line[0][0] - 5, line[0][1] - 5, line[2][0] + 5, line[2][1] + 5)
            new_image = img.crop(bbox)
            boxes.append(bbox)
            list_text.append(self.detector.predict(new_image))
        bbox_df = pd.DataFrame({"bbox": boxes, "text": list_text})
        bbox_df["page"] = page
        return bbox_df

    @staticmethod
    def pix2np(pix):
        if pix.n > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        return im

    def pdfimage2text(self, file: str) -> pd.DataFrame:
        zoom = 1
        mat = fitz.Matrix(zoom, zoom)
        doc = fitz.Document(file)
        pages = doc.pageCount
        block_df = pd.DataFrame()
        for page in range(pages):
            pix = doc[page].getPixmap(matrix=mat, alpha=False)
            img = self.pix2np(pix)
            block_df = pd.concat([block_df, self.img2text(img, page)])
        return block_df
