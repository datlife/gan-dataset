import torch.nn as nn
import torchvision as tv
import torch.utils.model_zoo as model_zoo

__all__ = ['AlexNet', 'alexnet']
model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',}


def preprocessor():
    data_transforms = tv.transforms.Compose([
                        tv.transforms.ToPILImage(),
                        tv.transforms.Resize(256),
                        tv.transforms.CenterCrop(224),
                        tv.transforms.ToTensor(),
                        tv.transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])
                      ])
    return data_transforms


def alexnet_conv3(pretrained=True, **kwargs):
    model = AlexNetConv3(**kwargs)

    if pretrained:
        original_alexnet_weights = model_zoo.load_url(model_urls['alexnet'])

        # Filter weights
        model_dict = model.state_dict()

        # filter out unnecessary keys
        filtered_dict = {k: v for k, v in original_alexnet_weights.items() if k in model_dict}

        # overwrite entries in the existing state dict
        model_dict.update(filtered_dict)

        # load the new state dict
        model.load_state_dict(filtered_dict)

    return model


class AlexNetConv3(nn.Module):

    def __init__(self):
        super(AlexNetConv3, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, 4, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(192, 384, 3, 1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 384 * 11 * 11)
        return x


def alexnet(pretrained=False, **kwargs):
    model = AlexNet(**kwargs)

    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(model_urls['alexnet'])
        )
    return model


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)

        return x

