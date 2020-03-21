import torch
import torch.nn as nn
from .utils import calc_gp, weights_init, save_model, load_model
from datasets.dataset import Dataset
from preprocessor import (
    MelNormalizer,
    MelDeNormalizer,
    FrameNormalizer,
    FrameDeNormalizer,
)
from torch.utils.data import DataLoader
from modules.sound2image import Generator, Discriminator
from copy import copy


DATA_CONFIG = {
    "load_files": ["frame", "log_mel_spec", "mel_if"],
    "mel_normalizer_savefile": "normalizer.json",
    "D_checkpoint_dir": "check_points/Discriminator",
    "G_checkpoint_dir": "check_points/Generator",
}

MODEL_CONFIG = {
    "lr": 0.0001,
    "beta1": 0,
    "beta2": 0.99,
    "iters": 100,
    "print_epoch": 100,
    "test_epoch": 500,
}


def train(data_dir, batch_size, exp_dir="./experiments", device="cuda"):
    # refine path with exp_dir
    data_config = copy(DATA_CONFIG)

    os.makedirs(exp_dir, exist_ok=True)
    for key in ["mel_normalizer_savefile", "D_checkpoint_dir", "G_checkpoint_dir"]:
        data_config[key] = os.path.join(exp_dir, data_config[key])

    # for normalizer of mel
    mel_data_loader = None
    if not os.path.isfile(data_config["mel_normalizer_savefile"]):
        mel_dataset = Dataset(
            data_dir=data_dir, transforms={}, load_files=["log_mel_spec", "mel_if"]
        )
        mel_data_loader = DataLoader(
            dataset=mel_dataset, batch_size=batch_size, shuffle=False
        )

    # Data definitions
    transforms = {
        "frame": FrameNormalizer(),
        "mel": MelNormalizer(
            dataloader=mel_data_loader,
            savefile_path=data_config["mel_normalizer_savefile"],
        ),
    }

    dataset = Dataset(
        data_dir=data_dir, transforms=transforms, load_files=data_config["load_files"]
    )
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # model definition
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # load model
    g_last_iter = load_model(model=netG, dir=data_config["G_checkpoint_dir"])
    d_last_iter = load_model(model=netD, dir=data_config["D_checkpoint_dir"])

    if g_last_iter != d_last_iter:
        raise ValueError(f"g_last_iter: {g_last_iter} != d_last_iter: {d_last_iter}")

    # parallelize of model
    if device.type == "cuda":
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    # weight initialize
    netG.apply(weights_init)
    netD.apply(weights_init)

    # set optimizer
    optimizer_d = torch.optim.Adam(
        netD.parameters(),
        MODEL_CONFIG["lr"],
        (MODEL_CONFIG["beta1"], MODEL_CONFIG["beta2"]),
    )
    optimizer_g = torch.optim.Adam(
        netG.parameters(),
        MODEL_CONFIG["lr"],
        (MODEL_CONFIG["beta1"], MODEL_CONFIG["beta2"]),
    )

    for iter_ in range(MODEL_CONFIG["iters"]):
        if iter_ <= last_iter:
            continue

        for idx, data_dict in enumerate(data_loader):
            data_dict = {key: value.to(device) for key, value in data_dict.items()}
            mel_data = torch.cat(
                [data_dict["log_mel_spec"], data_dict["mel_if"]], dim=1
            )

            ############
            # Update D #
            ############
            netD.zero_grad()
            netG.zero_grad()

            D_real = netD(data_dict["frame"]).view(-1).mean()

            gen_frames = netG(mel_data)
            D_fake = netD(gen_frames.detach()).view(-1).mean()

            gp = calc_gp(
                discriminator=netD,
                real_images=netD(data_dict["frame"]),
                fake_images=gen_frames.detach(),
                device=device,
            )

            wasserstein_D = D_real - D_fake
            d_loss = D_fake - D_real + gp
            d_loss.backward()

            optimizer_d.step()

            ############
            # Update G #
            ############
            netD.zero_grad()
            netG.zero_grad()

            gen_frames = netG(mel_data)
            DG_fake = netD(gen_frames).view(-1).mean()
            g_loss = -1 * DG_fake
            g_loss.backward()

            optimizer_g.step()

            if idx % MODEL_CONFIG["print_epoch"] == 0:
                print(
                    f"INFO: D_loss: {d_loss.item():4f} | G_loss: {g_loss.item():4f} | W_D: {wasserstein_D.item():4f}"
                )

                if idx % MODEL_CONFIG["test_epoch"] == 0:
                    ...

        save_model(model=netG, dir=data_config["G_checkpoint_dir"], iter=iter_)
        save_model(model=netD, dir=data_config["D_checkpoint_dir"], iter=iter_)
