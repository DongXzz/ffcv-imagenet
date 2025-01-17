from typing import List

import torch as ch
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm

from cifar10_models.resnet import ResNet18, ResNet50
import os, time

# Note that statistics are wrt to uin8 range, [0,255].
CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

BATCH_SIZE = 512
NUM_CLASSES = 10
EPOCHS = 50


class Mul(ch.nn.Module):
    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight
    def forward(self, x): return x * self.weight


class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)


def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
            ch.nn.Conv2d(channels_in, channels_out,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         groups=groups, bias=False),
            ch.nn.BatchNorm2d(channels_out),
            ch.nn.ReLU(inplace=True)
    )


def main():
    # Biuld dataloader
    loaders = {}
    datasets = ['train_cifar10Generate', 'test', 'test101', 'test102', 'test_cifar10Generate']
    for i, name in enumerate(datasets):
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(ch.device('cuda:0')), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if i==0:
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2),
                Cutout(8, tuple(map(int, CIFAR_MEAN))), # Note Cutout is done before normalization.
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(ch.device('cuda:0'), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        # Create loaders
        loaders[name] = Loader(f'/home/andong.hua/datasets/cifar10/{name}.beton',
                                batch_size=BATCH_SIZE,
                                num_workers=4,
                                order=OrderOption.RANDOM,
                                seed=7,
                                drop_last=(i==0),
                                pipelines={'image': image_pipeline,
                                        'label': label_pipeline})
    
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    weight_path = f'checkpoints/{datasets[0]}/{now}'
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    acc_list = []
    for i in range(3):
        # Biuld model
        model = ResNet50()
            
        # model = ch.nn.Sequential(
        #     conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        #     conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        #     Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        #     conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        #     ch.nn.MaxPool2d(2),
        #     Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        #     conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        #     ch.nn.AdaptiveMaxPool2d((1, 1)),
        #     Flatten(),
        #     ch.nn.Linear(128, NUM_CLASSES, bias=False),
        #     Mul(0.2)
        # )
        model = model.to(memory_format=ch.channels_last).cuda()
        
        # Train

        opt = SGD(model.parameters(), lr=.5, momentum=0.9, weight_decay=5e-4)
        # iters_per_epoch = 50000 // BATCH_SIZE
        iters_per_epoch = len(loaders[datasets[0]])
        lr_schedule = np.interp(np.arange((EPOCHS+1) * iters_per_epoch),
                                [0, 5 * iters_per_epoch, EPOCHS * iters_per_epoch],
                                [0, 1, 0])
        scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
        scaler = GradScaler()
        loss_fn = CrossEntropyLoss(label_smoothing=0.1)

        for ep in range(EPOCHS):
            for ims, labs in tqdm(loaders[datasets[0]]):
                opt.zero_grad(set_to_none=True)
                with autocast():
                    out = model(ims)
                    loss = loss_fn(out, labs)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                scheduler.step()
        ch.save(model, os.path.join(weight_path, str(i)+'.pth'))

        model.eval()
        with ch.no_grad():
            acc_sublist = []
            for name in datasets[1:]:
                test_loader = loaders[name]
                total_correct, total_num = 0., 0.
                for ims, labs in tqdm(test_loader):
                    with autocast():
                        out = (model(ims) + model(ch.fliplr(ims))) / 2. # Test-time augmentation
                        total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                        total_num += ims.shape[0]
                acc = total_correct / total_num
                acc_sublist.append(acc)
                print(f'Accuracy: {total_correct / total_num * 100:.1f}%')
        acc_list.append(acc_sublist)

    ave_acc = [0]*len(acc_sublist)
    for i in range(len(acc_list)):
        for j in range(len(acc_sublist)):
            ave_acc[j] += acc_list[i][j]
    ave_acc = [i / len(acc_list) for i in ave_acc]
    with open(os.path.join(weight_path, 'res.txt'), 'w') as f:
        f.write('\t'.join(datasets[1:]) + '\n')
        for row in acc_list:
            row_str = [f'{i * 100:.1f}%' for i in row]
            f.write('\t'.join(row_str) + '\n')
        row_str = [f'{i * 100:.1f}%' for i in ave_acc]
        f.write('\t'.join(row_str) + '\n')

if __name__ == "__main__":
    main()