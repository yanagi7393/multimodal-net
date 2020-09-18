import torch
from torch import nn
import torchvision


class ImageFeatureExtractor(nn.Module):
    def __init__(self, requires_grad=False, resize=True):
        super(ImageFeatureExtractor, self).__init__()
        net = torchvision.models.vgg16(pretrained=True).eval()

        nets = list(net.features) + [net.avgpool]
        self.slice1 = torch.nn.Sequential(*nets)
        self.slice2 = torch.nn.Sequential(*list(net.classifier[:-2]))

        # for tansforms
        self.resize = resize
        self.transform = torch.nn.functional.interpolate

        self.one_tensor = torch.nn.Parameter(torch.Tensor([1, 1, 1]).view(1, 3, 1, 1))
        self.mean = torch.nn.Parameter(
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.std = torch.nn.Parameter(
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def _pre_process(self, x):
        # renormalization
        x = (x + self.one_tensor - (2 * self.mean)) / (2 * self.std)

        if self.resize:
            x = self.transform(x, mode="bilinear", size=(224, 224), align_corners=True)

        return x

    def forward(self, X):
        B, _, _, _ = X.size()
        X = self._pre_process(X)

        feature = self.slice2(self.slice1(X).view(B, -1))
        return feature
