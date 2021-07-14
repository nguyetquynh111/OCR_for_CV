config = {
    "vocab": "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ",
    "device": "cpu",
    "seq_modeling": "seq2seq",
    "transformer": {
        "encoder_hidden": 256,
        "decoder_hidden": 256,
        "img_channel": 256,
        "decoder_embedded": 256,
        "dropout": 0.1,
    },
    "optimizer": {"max_lr": 0.001, "pct_start": 0.1},
    "trainer": {
        "batch_size": 32,
        "print_every": 200,
        "valid_every": 4000,
        "iters": 100000,
        "export": "./weights/transformerocr.pth",
        "checkpoint": "./checkpoint/transformerocr_checkpoint.pth",
        "log": "./train.log",
        "metrics": None,
    },
    "dataset": {
        "name": "data",
        "data_root": "./img/",
        "train_annotation": "annotation_train.txt",
        "valid_annotation": "annotation_val_small.txt",
        "image_height": 32,
        "image_min_width": 32,
        "image_max_width": 512,
    },
    "dataloader": {"num_workers": 3, "pin_memory": True},
    "aug": {"image_aug": True, "masked_language_model": True},
    "predictor": {"beamsearch": False},
    "quiet": False,
    "pretrain": {
        "id_or_url": "1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4",
        "md5": "efcabaa6d3adfca8e52bda2fd7d2ee04",
        "cached": "/tmp/tranformerorc.pth",
    },
    "weights": "./vietocr/transformerocr.pth",
    "backbone": "vgg19_bn",
    "cnn": {
        "ss": [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
        "ks": [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
        "hidden": 256,
    },
}