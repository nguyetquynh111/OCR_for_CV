import math

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import softmax

from .model.transformerocr import VietOCR
from .model.vocab import Vocab


def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token] * len(img)]
        char_probs = [[1] * len(img)]

        max_length = 0

        while max_length <= max_seq_length and not all(
            np.any(np.asarray(translated_sentence).T == eos_token, axis=1)
        ):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)

            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)
            output = output.to("cpu")

            values, indices = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence > 3)
        char_probs = np.sum(char_probs, axis=-1) / (char_probs > 0).sum(-1)

    return translated_sentence, char_probs


def build_model(config):
    vocab = Vocab(config["vocab"])
    device = config["device"]

    model = VietOCR(
        len(vocab),
        config["backbone"],
        config["cnn"],
        config["transformer"],
        config["seq_modeling"],
    )

    model = model.to(device)

    return model, vocab


def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w / round_to) * round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def process_image(image, image_height, image_min_width, image_max_width):
    img = image.convert("RGB")

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.ANTIALIAS)

    img = np.asarray(img).transpose(2, 0, 1)
    img = img / 255
    return img


def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img


class Predictor:
    def __init__(self, config):

        device = config["device"]

        model, vocab = build_model(config)
        weights = config["weights"]

        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab

    def predict(self, img, return_prob=False):
        img = process_input(
            img,
            self.config["dataset"]["image_height"],
            self.config["dataset"]["image_min_width"],
            self.config["dataset"]["image_max_width"],
        )
        img = img.to(self.config["device"])

        s, prob = translate(img, self.model)
        s = s[0].tolist()
        prob = prob[0]

        s = self.vocab.decode(s)

        return s
