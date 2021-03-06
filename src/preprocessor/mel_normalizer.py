import numpy as np
import torch
import json
import os


class MelNormalizer(object):
    def __init__(self, dataloader, savefile_path="./normalizer.json"):
        self.dataloader = dataloader
        self.savefile_path = savefile_path

        # load params
        self._load_params()
        print(
            f"Params: s_a:{self.s_a:.4f}, s_b:{self.s_b:.4f}, p_a:{self.p_a:.4f}, p_b:{self.p_b:.4f}"
        )

    def _save_params(self):
        params = {"s_a": self.s_a, "s_b": self.s_b, "p_a": self.p_a, "p_b": self.p_b}

        with open(self.savefile_path, mode="w") as file:
            json.dump(params, file)

    def _load_params(self):
        if not os.path.isfile(self.savefile_path):
            self._range_normalizer(magnitude_margin=0.8, IF_margin=1.0)
            return

        with open(self.savefile_path, mode="r") as file:
            params = json.load(file)

            self.s_a = params["s_a"]
            self.s_b = params["s_b"]
            self.p_a = params["p_a"]
            self.p_b = params["p_b"]

    def _range_normalizer(self, magnitude_margin, IF_margin):
        min_spec = 10000
        max_spec = -10000
        min_IF = 10000
        max_IF = -10000

        for batch_idx, data_dict in enumerate(self.dataloader):
            spec = data_dict["log_mel_spec"]
            IF = data_dict["mel_if"]

            if spec.min() < min_spec:
                min_spec = spec.min()
            if spec.max() > max_spec:
                max_spec = spec.max()

            if IF.min() < min_IF:
                min_IF = IF.min()
            if IF.max() > max_IF:
                max_IF = IF.max()

        self.s_a = (magnitude_margin * (2.0 / (max_spec - min_spec))).item()
        self.s_b = (
            magnitude_margin * (-2.0 * min_spec / (max_spec - min_spec) - 1.0)
        ).item()

        self.p_a = (IF_margin * (2.0 / (max_IF - min_IF))).item()
        self.p_b = (IF_margin * (-2.0 * min_IF / (max_IF - min_IF) - 1.0)).item()

        self._save_params()

    def __call__(self, spec, IF):
        spec = spec * self.s_a + self.s_b
        IF = IF * self.p_a + self.p_b

        return spec, IF


class MelDeNormalizer(object):
    def __init__(self, dataloader, savefile_path="./normalizer.json"):
        self.dataloader = dataloader
        self.savefile_path = savefile_path

        # load params
        self._load_params()

    def _save_params(self):
        params = {"s_a": self.s_a, "s_b": self.s_b, "p_a": self.p_a, "p_b": self.p_b}

        with open(self.savefile_path, mode="w") as file:
            json.dump(params, file)

    def _load_params(self):
        if not os.path.isfile(self.savefile_path):
            self._range_normalizer(magnitude_margin=0.8, IF_margin=1.0)
            return

        with open(self.savefile_path, mode="r") as file:
            params = json.load(file)

            self.s_a = params["s_a"]
            self.s_b = params["s_b"]
            self.p_a = params["p_a"]
            self.p_b = params["p_b"]

    def _range_normalizer(self, magnitude_margin, IF_margin):
        min_spec = 10000
        max_spec = -10000
        min_IF = 10000
        max_IF = -10000

        for batch_idx, data_dict in enumerate(self.dataloader):
            spec = data_dict["log_mel_spec"]
            IF = data_dict["mel_if"]

            if spec.min() < min_spec:
                min_spec = spec.min()
            if spec.max() > max_spec:
                max_spec = spec.max()

            if IF.min() < min_IF:
                min_IF = IF.min()
            if IF.max() > max_IF:
                max_IF = IF.max()

        self.s_a = (magnitude_margin * (2.0 / (max_spec - min_spec))).item()
        self.s_b = (
            magnitude_margin * (-2.0 * min_spec / (max_spec - min_spec) - 1.0)
        ).item()

        self.p_a = (IF_margin * (2.0 / (max_IF - min_IF))).item()
        self.p_b = (IF_margin * (-2.0 * min_IF / (max_IF - min_IF) - 1.0)).item()

        self._save_params()

    def __call__(self, spec, IF):
        spec = (spec - self.s_b) / self.s_a
        IF = (IF - self.p_b) / self.p_a
        return spec, IF
