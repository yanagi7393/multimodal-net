import torch
import torch.nn as nn
from .utils import calc_gp, weights_init, save_model, load_model
from datasets.dataset import Dataset
from datasets.img_dataset import IMGDataset
from preprocessor import (
    MelNormalizer,
    MelDeNormalizer,
    FrameNormalizer,
    FrameDeNormalizer,
)
import torchvision
from torch.utils.data import DataLoader
from models.sound2image_v2 import Generator, Discriminator
from copy import copy
import os
from itertools import chain
import fire
from modules.imgfeat_extractor import ImageFeatureExtractor


DATA_CONFIG = {
    "load_files": ["frame", "log_mel_spec", "mel_if"],
    "mel_normalizer_savefile": "./normalizer/mel_normalizer.json",
    "normalizer_dir": "./normalizer",
    "D_checkpoint_dir": "./check_points/Discriminator",
    "G_checkpoint_dir": "./check_points/Generator",
    "test_output_dir": "./test_outputs",
}

MODEL_CONFIG = {
    "batch_size": 64,
    "lr": 0.0002,
    "beta1": 0.5,
    "beta2": 0.99,
    "epochs": 100,
    "print_epoch": 1,
    "test_epoch": 1,
    "save_epoch": 1,
    "print_iter": 10,
    "test_iter": 500,
    "recon_lambda": 10,
    "recon_feat_lambda": 10,
    "g_norm": "BN",
    "d_norm": None,
    "g_sn": True,
    "d_sn": True,
    "dropout": 0,
    "loss_type": "hinge",
    "flip": True,
    "load_strict": True,
}


def _mutate_config_path(data_config, exp_dir):
    for key in [
        "mel_normalizer_savefile",
        "normalizer_dir",
        "D_checkpoint_dir",
        "G_checkpoint_dir",
        "test_output_dir",
    ]:
        if data_config[key][0] != "/":
            data_config[key] = os.path.join(exp_dir, data_config[key])

        if "_dir" in key:
            os.makedirs(data_config[key], exist_ok=True)

    return data_config


class Sound2ImageNet:
    def __init__(
        self,
        data_dir,
        test_data_dir,
        d_config={},
        m_config={},
        exp_dir="./experiments",
        device="cuda",
    ):
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        self.exp_dir = exp_dir
        self.device = device

        self._build_config(d_config=d_config, m_config=m_config)
        self._build_data_loaders()
        self._build_models()
        self._build_optimizers()
        self._build_loss_func()

    def _build_config(self, d_config, m_config):
        # refine path with exp_dirs
        data_config = copy(DATA_CONFIG)
        model_config = copy(MODEL_CONFIG)
        if not set(m_config.keys()).issubset(set(model_config.keys())):
            raise ValueError(f"{set(m_config.keys()) - set(model_config.keys())}")

        if not set(d_config.keys()).issubset(set(data_config.keys())):
            raise ValueError(f"{set(d_config.keys()) - set(data_config.keys())}")

        data_config = {**data_config, **d_config}
        model_config = {**model_config, **m_config}

        data_config = _mutate_config_path(data_config=data_config, exp_dir=self.exp_dir)

        self.data_config = data_config
        self.model_config = model_config

    def _get_transforms(self):
        # for normalizer of mel
        mel_data_loader = None
        if not os.path.isfile(self.data_config["mel_normalizer_savefile"]):
            mel_dataset = Dataset(
                data_dir=self.data_dir,
                transforms={},
                load_files=["log_mel_spec", "mel_if"],
            )
            mel_data_loader = DataLoader(
                dataset=mel_dataset,
                batch_size=self.model_config["batch_size"],
                shuffle=False,
                pin_memory=True,
                num_workers=self.model_config["batch_size"] // 2,
            )

        # Data definitions
        frame_transforms = [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        if self.model_config["flip"] is not True:
            flip_transform = frame_transforms.pop(1)
            assert isinstance(
                flip_transform, torchvision.transforms.RandomHorizontalFlip
            )

        transforms = {
            "frame": torchvision.transforms.Compose(frame_transforms),
            "mel": MelNormalizer(
                dataloader=mel_data_loader,
                savefile_path=self.data_config["mel_normalizer_savefile"],
            ),
        }

        return transforms

    def _build_data_loaders(self):
        transforms = self._get_transforms()

        # DEFINE: DATASETS
        train_dataset = Dataset(
            data_dir=self.data_dir,
            transforms=transforms,
            load_files=self.data_config["load_files"],
        )

        test_dataset = Dataset(
            data_dir=self.test_data_dir,
            transforms=transforms,
            load_files=self.data_config["load_files"],
        )

        # DEFINE: DATA LOADER
        self.train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.model_config["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=self.model_config["batch_size"] // 2,
        )

        self.test_data_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.model_config["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=self.model_config["batch_size"] // 2,
        )

        self.test_data_loader_ = iter(self.test_data_loader)

    def _load_models(self, netG, netD):
        # load model
        g_last_epoch = load_model(
            model=netG,
            dir=self.data_config["G_checkpoint_dir"],
            strict=self.model_config["load_strict"],
        )
        d_last_epoch = load_model(
            model=netD,
            dir=self.data_config["D_checkpoint_dir"],
            strict=self.model_config["load_strict"],
        )

        last_epoch = min(g_last_epoch, d_last_epoch)
        if g_last_epoch != last_epoch:
            load_model(
                model=netG,
                dir=self.data_config["G_checkpoint_dir"],
                load_epoch=last_epoch,
                strict=self.model_config["load_strict"],
            )

        if d_last_epoch != last_epoch:
            load_model(
                model=netD,
                dir=self.data_config["D_checkpoint_dir"],
                load_epoch=last_epoch,
                strict=self.model_config["load_strict"],
            )

        return netG, netD, last_epoch

    def _save_models(self, netG, netD, epoch):
        save_model(model=netG, dir=self.data_config["G_checkpoint_dir"], epoch=epoch)
        save_model(model=netD, dir=self.data_config["D_checkpoint_dir"], epoch=epoch)

    def _build_models(self):
        # DEFINE: MODELS
        # 1. main model
        self.netG = Generator(
            sn=self.model_config["g_sn"],
            norm=self.model_config["g_norm"],
            dropout=self.model_config["dropout"],
        )
        self.netD = Discriminator(
            sn=self.model_config["d_sn"], norm=self.model_config["d_norm"],
        )
        self.imgfeat_extractor = ImageFeatureExtractor(requires_grad=False, resize=True)

        # INIT: init MODEL's weights
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # PARALLELIZE MODELS
        if torch.cuda.device_count() > 1:
            print("NOTICE: use ", torch.cuda.device_count(), "GPUs")
            self.netG = nn.DataParallel(self.netG)
            self.netD = nn.DataParallel(self.netD)

            self.imgfeat_extractor = nn.DataParallel(self.imgfeat_extractor)

        else:
            print(f"NOTICE: use {self.device}")

        # LOAD: MODELS
        self.netG, self.netD, self.last_epoch = self._load_models(
            netG=self.netG, netD=self.netD
        )

        # MODELS TO DEVICE
        self.netG.to(self.device)
        self.netD.to(self.device)
        self.imgfeat_extractor.to(self.device)

        # Use only eval mode
        self.imgfeat_extractor.eval()

    def _build_optimizers(self):
        # set optimizer
        self.optimizer_d = torch.optim.Adam(
            self.netD.parameters(),
            self.model_config["lr"],
            (self.model_config["beta1"], self.model_config["beta2"]),
        )
        self.optimizer_g = torch.optim.Adam(
            self.netG.parameters(),
            self.model_config["lr"],
            (self.model_config["beta1"], self.model_config["beta2"]),
        )

    def _build_loss_func(self):
        self.l1_criterion = nn.L1Loss().to(self.device)
        self.ce_criterion = nn.CrossEntropyLoss().to(self.device)

    def set_mode(self, mode):
        getattr(self.netD, mode)()
        getattr(self.netG, mode)()

    def step_loss(self, data_dict):
        # Set DATA
        data_dict = {key: value.to(self.device) for key, value in data_dict.items()}
        frame_data = data_dict["frame"]

        # generate feature from images
        img_feature = self.imgfeat_extractor(frame_data).detach()
        mel_data = torch.stack([data_dict["log_mel_spec"], data_dict["mel_if"]], dim=1)

        # SET TRAIN MODE
        self.set_mode("train")

        # Deplicated wgan-gp loss
        assert self.model_config["loss_type"] in ("hinge")
        if self.model_config["loss_type"] == "hinge":
            #################
            # Discriminator #
            #################
            self.netD.zero_grad()
            self.netG.zero_grad()

            # Real image
            D_real_out = self.netD(frame_data)
            D_real = torch.nn.ReLU()(1.0 - D_real_out).mean(dim=-1).mean()

            # Fake images
            gen_frames, g_latent_vec = self.netG(mel_data)
            gen_frames_by_imgfeat, _ = self.netG(img_feature, is_feature=True)

            gen_frames_detached = gen_frames.detach()
            gen_frames_by_imgfeat_detached = gen_frames_by_imgfeat.detach()

            D_fake = 0
            for gen_frames_ in [
                gen_frames_detached,
                gen_frames_by_imgfeat_detached,
            ]:
                D_fake_out = self.netD(gen_frames_)
                D_fake += torch.nn.ReLU()(1.0 + D_fake_out).mean(dim=-1).mean()

            D_fake = D_fake / 2

            d_loss = D_real + D_fake
            d_loss.backward()
            self.optimizer_d.step()

            #############
            # Generator #
            #############
            self.netD.zero_grad()
            self.netG.zero_grad()

            DG_fake = 0
            for gen_frames_ in [
                gen_frames,
                gen_frames_by_imgfeat,
            ]:
                DG_fake_out = self.netD(gen_frames_)
                DG_fake += -DG_fake_out.mean(dim=-1).mean()

            DG_fake = DG_fake / 2

            # Compute recon loss
            recon_loss = 0
            if float(self.model_config["recon_lambda"]) != 0.0:
                recon_loss += (
                    self.l1_criterion(gen_frames, frame_data)
                    * self.model_config["recon_lambda"]
                )
                recon_loss += (
                    self.l1_criterion(gen_frames_by_imgfeat, frame_data)
                    * self.model_config["recon_lambda"]
                )

                recon_loss = recon_loss / 2

            recon_feat_loss = (
                self.l1_criterion(g_latent_vec, img_feature)
                * self.model_config["recon_feat_lambda"]
            )

            g_loss = recon_loss + DG_fake + recon_feat_loss
            g_loss.backward()
            self.optimizer_g.step()

            losses = {
                "D": {"main": d_loss.item(),},
                "G": {
                    "main": g_loss.item(),
                    "recon": recon_loss.item(),
                    "recon_feat": recon_feat_loss.item(),
                },
            }

        else:
            raise NotImplementedError

        return losses

    def run_test(self, epoch_, iter_, losses):
        if epoch_ % self.model_config["print_epoch"] == 0:
            if iter_ % self.model_config["print_iter"] == 0:
                print(
                    f"""INFO: D_loss: {losses['D']['main']:.2f} || G_loss: {losses['G']['main']:.2f}
REC: {losses['G']['recon']:.2f} | REC_FEAT: {losses['G']['recon_feat']:.2f}"""
                )

        if epoch_ % self.model_config["test_epoch"] == 0:
            if iter_ % self.model_config["test_iter"] == 0:
                # EVAL MODE
                self.set_mode(mode="eval")

                try:
                    test_data_dict = next(self.test_data_loader_)
                except StopIteration:
                    self.test_data_loader_ = iter(self.test_data_loader)
                    test_data_dict = next(self.test_data_loader_)

                test_data_dict = dict(
                    [
                        (key, value.to(self.device))
                        if key in ["log_mel_spec", "mel_if"]
                        else (key, value.to("cpu"))
                        for key, value in test_data_dict.items()
                    ]
                )

                mel_test_data = torch.stack(
                    [test_data_dict["log_mel_spec"], test_data_dict["mel_if"]], dim=1,
                )

                with torch.no_grad():
                    gen_frames, _ = self.netG(mel_test_data)
                    gen_frames = gen_frames.detach()

                concat_frames = torch.stack(
                    list(
                        chain(
                            *[
                                (fake_img, real_img)
                                for fake_img, real_img in zip(
                                    gen_frames.cpu(), test_data_dict["frame"]
                                )
                            ]
                        )
                    ),
                    dim=0,
                )

                torchvision.utils.save_image(
                    concat_frames,
                    os.path.join(
                        self.data_config["test_output_dir"], f"{epoch_}-{iter_}.png"
                    ),
                    nrow=2,
                    padding=10,
                    range=(-1.0, 1.0),
                    normalize=True,
                )

    def train(self):
        for epoch in range(self.model_config["epochs"]):
            if epoch <= self.last_epoch:
                continue

            for idx, data_dict in enumerate(
                self.train_data_loader
            ):
                # OPTIMIZE
                losses = self.step_loss(data_dict=data_dict)

                # TEST
                self.run_test(epoch_=epoch, iter_=idx, losses=losses)

            if epoch % self.model_config["save_epoch"] == 0:
                self._save_models(netG=self.netG, netD=self.netD, epoch=epoch)


if __name__ == "__main__":
    fire.Fire(Sound2ImageNet)
