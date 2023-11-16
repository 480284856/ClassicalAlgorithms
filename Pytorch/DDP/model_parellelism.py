import torch
import timeit
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from typing import Callable, List, Optional, Type, Union
from torchvision.models.resnet import BasicBlock, Bottleneck,resnet50
from torch.nn import Module

class ResNetModelParellelism(models.ResNet):
    def __init__(self, 
                 block: type[BasicBlock] | type[Bottleneck], 
                 layers: List[int], 
                 num_classes: int = 1000, 
                 zero_init_residual: bool = False, 
                 groups: int = 1, 
                 width_per_group: int = 64, 
                 replace_stride_with_dilation: List[bool] | None = None, 
                 norm_layer: Callable[..., Module] | None = None) -> None:
        super(ResNetModelParellelism, self).__init__(
                block, 
                layers, 
                num_classes, 
                zero_init_residual, 
                groups, 
                width_per_group, 
                replace_stride_with_dilation,
                norm_layer)
        
        self.sub_model1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to(torch.device("cuda:0"))

        self.sub_model2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to(torch.device("cuda:1"))

        self.fc.to(torch.device("cuda:1"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.device("cuda:0"))
        x = self.sub_model1(x).to(torch.device("cuda:1"))

        x = self.sub_model2(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x

def main(model):
    num_classes = 1000
    num_batches = 3
    batch_size = 120
    image_w = 128
    image_h = 128



    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=0.001
    )

    one_hot_indices = torch.LongTensor(batch_size).random_(0, num_classes).view(batch_size,1)

    for _ in range(num_batches):
        inputs = torch.randn(
            batch_size, 3, image_w, image_h
        )
        labels = torch.zeros(
            batch_size, num_classes
        ).scatter_(1, one_hot_indices, 1)

        optimizer.zero_grad()
        output = model(inputs.to(torch.device("cuda:0")))

        labels = labels.to(output.device)
        loss = loss_fn(output, labels)
        loss.backward()

        optimizer.step()

def timeRecord():
    num_repeat = 10

    stmt = "main(model_mp)"
    setup = "model_mp = ResNetModelParellelism(block=Bottleneck,layers=[3,4,6,3],num_classes=1000)"
    mp_run_times = timeit.repeat(
        stmt=stmt,
        setup=setup,
        number=1,
        repeat=num_repeat,
        globals=globals()
    )
    mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

    setup = "model_rn = models.resnet50(num_classes=1000).to('cuda:0')"
    stmt = "main(model_rn)"
    rn_run_times = timeit.repeat(
        stmt=stmt,
        setup=setup,
        number=1,
        repeat=num_repeat,
        globals=globals()
    )
    rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)
    return mp_mean, mp_std,rn_mean, rn_std

def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)



if __name__ == "__main__":
    mp_mean, mp_std,rn_mean, rn_std = timeRecord()
    plot([mp_mean, rn_mean],
        [mp_std, rn_std],
        ['Model Parallel', 'Single GPU'],
        'mp_vs_rn.png')
