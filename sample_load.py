from PaddleOCR.tools.infer.predict_det import TextDetector
import fitz
import pandas as pd
import numpy as np


def parse_args(mMain=True, add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    if mMain:
        parser = argparse.ArgumentParser(add_help=add_help)
        # params for prediction engine
        parser.add_argument("--use_gpu", type=str2bool, default=True)
        parser.add_argument("--ir_optim", type=str2bool, default=True)
        parser.add_argument("--use_tensorrt", type=str2bool, default=False)
        parser.add_argument("--gpu_mem", type=int, default=8000)

        # params for text detector
        parser.add_argument("--image_dir", type=str)
        parser.add_argument("--det_algorithm", type=str, default='DB')
        parser.add_argument("--det_model_dir", type=str, default=None)
        parser.add_argument("--det_limit_side_len", type=float, default=960)
        parser.add_argument("--det_limit_type", type=str, default='max')

        # DB parmas
        parser.add_argument("--det_db_thresh", type=float, default=0.3)
        parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
        parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
        parser.add_argument("--use_dilation", type=bool, default=False)
        parser.add_argument("--det_db_score_mode", type=str, default="fast")

        # EAST parmas
        parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
        parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
        parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

        # params for text recognizer
        parser.add_argument("--rec_algorithm", type=str, default='CRNN')
        parser.add_argument("--rec_model_dir", type=str, default=None)
        parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
        parser.add_argument("--rec_char_type", type=str, default='ch')
        parser.add_argument("--rec_batch_num", type=int, default=6)
        parser.add_argument("--max_text_length", type=int, default=25)
        parser.add_argument("--rec_char_dict_path", type=str, default=None)
        parser.add_argument("--use_space_char", type=bool, default=True)
        parser.add_argument("--drop_score", type=float, default=0.5)

        # params for text classifier
        parser.add_argument("--cls_model_dir", type=str, default=None)
        parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
        parser.add_argument("--label_list", type=list, default=['0', '180'])
        parser.add_argument("--cls_batch_num", type=int, default=6)
        parser.add_argument("--cls_thresh", type=float, default=0.9)

        parser.add_argument("--enable_mkldnn", type=bool, default=False)
        parser.add_argument("--use_zero_copy_run", type=bool, default=False)
        parser.add_argument("--use_pdserving", type=str2bool, default=False)

        parser.add_argument("--lang", type=str, default='ch')
        parser.add_argument("--det", type=str2bool, default=True)
        parser.add_argument("--rec", type=str2bool, default=True)
        parser.add_argument("--use_angle_cls", type=str2bool, default=False)
        return parser.parse_args()
    else:
        return argparse.Namespace(
            use_gpu=False,
            ir_optim=True,
            use_tensorrt=False,
            gpu_mem=8000,
            image_dir='',
            det_algorithm='DB',
            det_model_dir=None,
            det_limit_side_len=960,
            det_limit_type='max',
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            use_dilation=False,
            det_db_score_mode="fast",
            det_east_score_thresh=0.8,
            det_east_cover_thresh=0.1,
            det_east_nms_thresh=0.2,
            rec_algorithm='CRNN',
            rec_model_dir=None,
            rec_image_shape="3, 32, 320",
            rec_char_type='ch',
            rec_batch_num=6,
            max_text_length=25,
            rec_char_dict_path=None,
            use_space_char=True,
            drop_score=0.5,
            cls_model_dir=None,
            cls_image_shape="3, 48, 192",
            label_list=['0', '180'],
            cls_batch_num=6,
            cls_thresh=0.9,
            enable_mkldnn=False,
            use_zero_copy_run=False,
            use_pdserving=False,
            lang='ch',
            det=True,
            rec=False,
            use_angle_cls=False)



postprocess_params = parse_args(mMain=False, add_help=False)
postprocess_params.use_angle_cls = True
postprocess_params.lang = "en"
postprocess_params.det_model_dir = "data"
postprocess_params.rec_model_dir = "data"
postprocess_params.cls_model_dir = "data"

text_detector = TextDetector(postprocess_params)



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
        result = text_detector(pix)
        break
    return block_df

import time
def main():
    individual_file = "/Users/quan/Desktop/jobhop_2/2021/7/image-cv/cvSample4.pdf"
    start = time.time()
    pdfimage2text(individual_file)
    print(time.time() - start)

if __name__ == '__main__':
    main()