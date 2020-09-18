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
from models.sound2image import Generator, Discriminator
from copy import copy
import os
from itertools import chain
import fire
from modules.perceptual_loss import VGGLoss, VGGLabel, Vgg19


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
    "fm_lambda": 0.1,
    "gp_lambda": 10,
    "pl_lambda": 0.0,
    "g_ce_lambda": 0.5,
    "d_ce_lambda": 0.05,
    "extraimg_ratio": 0.5,
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
        extraimg_data_dir=None,
        extraimg_type="jpg",
        d_config={},
        m_config={},
        exp_dir="./experiments",
        device="cuda",
    ):
        self.data_dir = data_dir
        self.test_data_dir = test_data_dir
        self.extraimg_data_dir = extraimg_data_dir
        self.extraimg_type = extraimg_type
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
        extraimg_transform = torchvision.transforms.Compose(
            transforms["frame"].transforms[1:]
        )

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

        train_extraimg_dataset = IMGDataset(
            data_dir=self.extraimg_data_dir,
            data_type=self.extraimg_type,
            transform=extraimg_transform,
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

        self.train_extraimg_data_loader = DataLoader(
            dataset=train_extraimg_dataset,
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
            use_class=True if float(self.model_config["g_ce_lambda"]) != 0.0 else False,
        )
        self.netD = Discriminator(
            sn=self.model_config["d_sn"],
            norm=self.model_config["d_norm"],
            use_class=True if float(self.model_config["d_ce_lambda"]) != 0.0 else False,
        )
        if (
            float(self.model_config["pl_lambda"]) != 0.0
            or float(self.model_config["g_ce_lambda"]) != 0.0
            or float(self.model_config["d_ce_lambda"]) != 0.0
        ):
            self.vgg = Vgg19(requires_grad=False, resize=True)

        # 2. perceptual loss
        self.vgg_loss = None
        if float(self.model_config["pl_lambda"]) != 0.0:
            self.vgg_loss = VGGLoss(vgg=self.vgg)

        # 3. ce loss
        self.vgg_label = None
        if (
            float(self.model_config["g_ce_lambda"]) != 0.0
            or float(self.model_config["d_ce_lambda"]) != 0.0
        ):
            self.vgg_label = VGGLabel(vgg=self.vgg)

        # INIT: init MODEL's weights
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # PARALLELIZE MODELS
        if torch.cuda.device_count() > 1:
            print("NOTICE: use ", torch.cuda.device_count(), "GPUs")
            self.netG = nn.DataParallel(self.netG)
            self.netD = nn.DataParallel(self.netD)

            if self.vgg_loss is not None:
                self.vgg_loss = nn.DataParallel(self.vgg_loss)

            if self.vgg_label is not None:
                self.vgg_label = nn.DataParallel(self.vgg_label)
        else:
            print(f"NOTICE: use {self.device}")

        # LOAD: MODELS
        self.netG, self.netD, self.last_epoch = self._load_models(
            netG=self.netG, netD=self.netD
        )

        # MODELS TO DEVICE
        self.netG.to(self.device)
        self.netD.to(self.device)
        if self.vgg_loss is not None:
            self.vgg_loss.to(self.device)
        if self.vgg_label is not None:
            self.vgg_label.to(self.device)

            # Use only eval mode
            self.vgg_label.eval()

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

    def step_loss(self, data_dict, extraimg_data):
        # Set DATA
        data_dict = {key: value.to(self.device) for key, value in data_dict.items()}
        extraimg_data = extraimg_data.to(self.device)

        frame_data = data_dict["frame"]
        mel_data = torch.stack([data_dict["log_mel_spec"], data_dict["mel_if"]], dim=1)

        # SET TRAIN MODE
        self.set_mode("train")

        if self.vgg_label is not None:
            frame_labels = self.vgg_label(x=frame_data).detach()

        if self.model_config["loss_type"] == "wgan":
            #################
            # Discriminator #
            #################
            self.netD.zero_grad()
            self.netG.zero_grad()

            # Real_data
            feature_reals, D_real_outs, D_classes = self.netD(frame_data)
            feature_reals = [feature_real.detach() for feature_real in feature_reals]
            D_real = 0
            for D_real_out in D_real_outs:
                D_real += D_real_out.view(-1).mean()
            D_real = D_real / len(D_real_outs)

            _, D_real_outs_extra, D_classes_extra = self.netD(extraimg_data)
            D_real_extra = 0
            for D_real_out_extra in D_real_outs_extra:
                D_real_extra += D_real_out_extra.view(-1).mean()
            D_real_extra = D_real_extra / len(D_real_outs_extra)

            # Fake data
            gen_frames, G_classes = self.netG(mel_data)
            gen_frames_detached = gen_frames.detach()
            _, D_fake_outs, _ = self.netD(gen_frames_detached)
            D_fake = 0
            for D_fake_out in D_fake_outs:
                D_fake += D_fake_out.view(-1).mean()
            D_fake = D_fake / len(D_fake_outs)

            gp = calc_gp(
                discriminator=self.netD,
                real_images=frame_data,
                fake_images=gen_frames_detached,
                lambda_term=self.model_config["gp_lambda"],
                device=self.device,
            )

            d_ce_loss = 0
            if float(self.model_config["d_ce_lambda"]) != 0.0:
                extra_frame_labels = self.vgg_label(x=extraimg_data).detach()

                d_ce_loss += (
                    (
                        self.ce_criterion(D_classes.squeeze(), frame_labels)
                        + self.ce_criterion(
                            D_classes_extra.squeeze(), extra_frame_labels
                        )
                    )
                    / 2
                    * self.model_config["d_ce_lambda"]
                )

            # set loss
            d_loss = (
                D_fake
                - (D_real * (1 - self.model_config["extraimg_ratio"]))
                - (D_real_extra * self.model_config["extraimg_ratio"])
                + gp
                + d_ce_loss
            )
            d_loss.backward()
            self.optimizer_d.step()

            #############
            # Generator #
            #############
            self.netD.zero_grad()
            self.netG.zero_grad()

            feature_fakes, DG_fake_outs, _ = self.netD(gen_frames)
            DG_fake = 0
            for DG_fake_out in DG_fake_outs:
                DG_fake += -DG_fake_out.view(-1).mean()
            DG_fake = DG_fake / len(DG_fake_outs)

            # set loss
            recon_loss = 0
            if float(self.model_config["recon_lambda"]) != 0.0:
                recon_loss = (
                    self.l1_criterion(gen_frames, frame_data)
                    * self.model_config["recon_lambda"]
                )

            fm_loss = 0
            if float(self.model_config["fm_lambda"]) != 0.0:
                for feature_real, feature_fake in zip(feature_reals, feature_fakes):
                    fm_loss += self.l1_criterion(feature_fake, feature_real)
                fm_loss = (fm_loss / len(feature_reals)) * self.model_config[
                    "fm_lambda"
                ]

            pl_loss = 0
            if float(self.model_config["pl_lambda"]) != 0.0:
                pl_loss = (
                    self.vgg_loss(x=gen_frames, y=frame_data).view(-1).mean()
                    * self.model_config["pl_lambda"]
                )

            g_ce_loss = 0
            if float(self.model_config["g_ce_lambda"]) != 0.0:
                g_ce_loss += (
                    self.ce_criterion(G_classes.squeeze(), frame_labels)
                ) * self.model_config["g_ce_lambda"]

            g_loss = recon_loss + fm_loss + pl_loss + g_ce_loss + DG_fake
            g_loss.backward()
            self.optimizer_g.step()

            losses = {
                "D": {
                    "main": d_loss.item(),
                    "ce": d_ce_loss.item()
                    if isinstance(d_ce_loss, torch.Tensor)
                    else d_ce_loss,
                },
                "G": {
                    "main": g_loss.item(),
                    "fm": fm_loss.item()
                    if isinstance(fm_loss, torch.Tensor)
                    else fm_loss,
                    "pl": pl_loss.item()
                    if isinstance(pl_loss, torch.Tensor)
                    else pl_loss,
                    "recon": recon_loss.item()
                    if isinstance(recon_loss, torch.Tensor)
                    else recon_loss,
                    "ce": g_ce_loss.item()
                    if isinstance(g_ce_loss, torch.Tensor)
                    else g_ce_loss,
                },
            }

        elif self.model_config["loss_type"] == "hinge":
            #################
            # Discriminator #
            #################
            self.netD.zero_grad()
            self.netG.zero_grad()

            # Real data
            feature_reals, D_real_outs, D_classes = self.netD(frame_data)
            feature_reals = [feature_real.detach() for feature_real in feature_reals]
            D_real = 0
            for D_real_out in D_real_outs:
                D_real += torch.nn.ReLU()(1.0 - D_real_out).view(-1).mean()
            D_real = D_real / len(D_real_outs)

            _, D_real_outs_extra, D_classes_extra = self.netD(extraimg_data)
            D_real_extra = 0
            for D_real_out_extra in D_real_outs_extra:
                D_real_extra += torch.nn.ReLU()(1.0 - D_real_out_extra).view(-1).mean()
            D_real_extra = D_real_extra / len(D_real_outs_extra)

            # Fake data
            gen_frames, G_classes = self.netG(mel_data)
            gen_frames_detached = gen_frames.detach()
            _, D_fake_outs, _ = self.netD(gen_frames_detached)
            D_fake = 0
            for D_fake_out in D_fake_outs:
                D_fake += torch.nn.ReLU()(1.0 + D_fake_out).view(-1).mean()
            D_fake = D_fake / len(D_fake_outs)

            d_ce_loss = 0
            if float(self.model_config["d_ce_lambda"]) != 0.0:
                extra_frame_labels = self.vgg_label(x=extraimg_data).detach()

                d_ce_loss += (
                    (
                        self.ce_criterion(D_classes.squeeze(), frame_labels)
                        + self.ce_criterion(
                            D_classes_extra.squeeze(), extra_frame_labels
                        )
                    )
                    / 2
                    * self.model_config["d_ce_lambda"]
                )

            d_loss = (
                (D_real * (1 - self.model_config["extraimg_ratio"]))
                + (D_real_extra * self.model_config["extraimg_ratio"])
                + D_fake
                + d_ce_loss
            )
            d_loss.backward()
            self.optimizer_d.step()

            #############
            # Generator #
            #############
            self.netD.zero_grad()
            self.netG.zero_grad()

            feature_fakes, DG_fake_outs, _ = self.netD(gen_frames)
            DG_fake = 0
            for DG_fake_out in DG_fake_outs:
                DG_fake += -DG_fake_out.view(-1).mean()
            DG_fake = DG_fake / len(DG_fake_outs)

            # set loss
            recon_loss = 0
            if float(self.model_config["recon_lambda"]) != 0.0:
                recon_loss = (
                    self.l1_criterion(gen_frames, frame_data)
                    * self.model_config["recon_lambda"]
                )

            fm_loss = 0
            if float(self.model_config["fm_lambda"]) != 0.0:
                for feature_real, feature_fake in zip(feature_reals, feature_fakes):
                    fm_loss += self.l1_criterion(feature_fake, feature_real)
                fm_loss = (fm_loss / len(feature_reals)) * self.model_config[
                    "fm_lambda"
                ]

            pl_loss = 0
            if float(self.model_config["pl_lambda"]) != 0.0:
                pl_loss = (
                    self.vgg_loss(x=gen_frames, y=frame_data).view(-1).mean()
                    * self.model_config["pl_lambda"]
                )

            g_ce_loss = 0
            if float(self.model_config["g_ce_lambda"]) != 0.0:
                g_ce_loss += (
                    self.ce_criterion(G_classes.squeeze(), frame_labels)
                ) * self.model_config["g_ce_lambda"]

            g_loss = recon_loss + fm_loss + pl_loss + g_ce_loss + DG_fake
            g_loss.backward()
            self.optimizer_g.step()

            losses = {
                "D": {
                    "main": d_loss.item(),
                    "ce": d_ce_loss.item()
                    if isinstance(d_ce_loss, torch.Tensor)
                    else d_ce_loss,
                },
                "G": {
                    "main": g_loss.item(),
                    "fm": fm_loss.item()
                    if isinstance(fm_loss, torch.Tensor)
                    else fm_loss,
                    "pl": pl_loss.item()
                    if isinstance(pl_loss, torch.Tensor)
                    else pl_loss,
                    "recon": recon_loss.item()
                    if isinstance(recon_loss, torch.Tensor)
                    else recon_loss,
                    "ce": g_ce_loss.item()
                    if isinstance(g_ce_loss, torch.Tensor)
                    else g_ce_loss,
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
D_CE: {losses['D']['ce']:.2f} | G_CE: {losses['G']['ce']:.2f} || REC: {losses['G']['recon']:.2f} | FM: {losses['G']['fm']:.2f} | PL: {losses['G']['pl']:.2f}"""
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

            for idx, (data_dict, extraimg_data) in enumerate(
                zip(self.train_data_loader, self.train_extraimg_data_loader)
            ):
                # OPTIMIZE
                losses = self.step_loss(
                    data_dict=data_dict, extraimg_data=extraimg_data
                )

                # TEST
                self.run_test(epoch_=epoch, iter_=idx, losses=losses)

            if epoch % self.model_config["save_epoch"] == 0:
                self._save_models(netG=self.netG, netD=self.netD, epoch=epoch)


if __name__ == "__main__":
    fire.Fire(Sound2ImageNet)
