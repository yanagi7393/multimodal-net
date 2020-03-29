import torch
import torch.nn as nn
import os
from glob import glob
from modules.norms import _BatchInstanceNorm2d


def save_model(model, dir, iter):
    os.makedirs(dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(dir, f"checkpoint-{iter}"))

    print("[+] Model is saved")


def load_model(model, dir, load_iter=None):
    if not os.path.isdir(dir):
        print("[!] Load is failed")
        return -1

    # If load_iter has value
    if load_iter is not None:
        if load_iter == -1:
            print("[!] Load is failed")
            return -1

        model.load_state_dict(torch.load(os.path.join(dir, f"checkpoint-{load_iter}")))
        return load_iter

    check_points = glob(os.path.join(dir, "checkpoint-*"))
    check_points = sorted(
        check_points, key=lambda x: int(x.split("/")[-1].replace("checkpoint-", ""))
    )

    # skip if there are no checkpoints
    if len(check_points) == 0:
        print("[!] Load is failed")
        return -1

    check_point = check_points[-1]

    model.load_state_dict(torch.load(check_point))
    print("[+] Model is loaded")

    return int(check_point.replace("checkpoint-", ""))


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.GroupNorm, _BatchInstanceNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias"):
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
