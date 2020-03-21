import torch
import torch.nn as nn
import os
from glob import glob


def save_model(model, dir, iter):
    os.makedirs(dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(dir, f"checkpoint-{iter}"))

    print("[+] Model is saved")


def load_model(model, dir):
    if not os.path.isdir(dir):
        print("[!] Load is failed")
        return -1

    check_points = glob(os.path.join(dir, "checkpoint-*"))
    check_points = sorted(check_points, key=lambda x: int(x.replace("checkpoint-", "")))
    check_point = check_points[-1]

    model.load_state_dict(torch.load(check_point))
    print("[+] Model is loaded")

    return int(check_point.replace("checkpoint-", ""))


def weights_init(m):
    classname = m.__class__.__name__

    if any(
        [
            classname.find(type) != -1
            for type in ["Conv", "Res", "Block", "SelfAttention", "Norm"]
        ]
    ):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def calc_gp(discriminator, real_images, fake_images, lambda_term=10, device="cpu"):
    alpha = torch.FloatTensor(real_images.size(0), 1, 1, 1).uniform_(0, 1).to(device)
    alpha = alpha.expand(
        real_images.size(0),
        real_images.size(1),
        real_images.size(2),
        real_images.size(3),
    )

    interpolated = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(
        True
    )
    prob_interpolated = discriminator(interpolated)
    grad_outputs = torch.ones(prob_interpolated.size()).to(device)

    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term

    return gradient_penalty
