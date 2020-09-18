import torch
from torch import nn
import torchvision


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False, resize=True):
        super(Vgg19, self).__init__()
        self.net = torchvision.models.vgg19(pretrained=True).eval()
        vgg_pretrained_features = self.net.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

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
        X = self._pre_process(X)

        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)

        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLabel(nn.Module):
    def __init__(self, vgg):
        super(VGGLabel, self).__init__()
        self.vgg = vgg

    def forward(self, x):
        x = self.vgg._pre_process(x)
        return self.vgg.net(x).argmax(dim=-1)


class VGGLoss(nn.Module):
    def __init__(self, vgg):
        super(VGGLoss, self).__init__()
        self.vgg = vgg
        self.criterion = nn.L1Loss(reduction="none")
        self.weights = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 1.0]

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        b_size = x.size()[0]
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0
        for i in range(len(x_vgg)):
            loss += (
                self.criterion(x_vgg[i], y_vgg[i].detach())
                .view([b_size, -1])
                .mean(dim=-1)
                * self.weights[i]
            )

        return loss
