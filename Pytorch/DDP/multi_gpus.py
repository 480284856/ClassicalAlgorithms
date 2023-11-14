import os
import torch
import socket  
import random
import time

from torchvision import transforms
from single_gpu import Trainer,Model,MNIST
from torch.utils.data import DataLoader,DistributedSampler
from torch.distributed import init_process_group,destroy_process_group,get_rank
from torch.nn.parallel import DistributedDataParallel

def ddp_initialization(
        local_rank,
        world_size,
        master_ip="localhost",
        master_port="42344",
        backend="nccl"
    ):
    '''
    initialization for distributed data parallelism
    ----
    input:
        local_rank: local_rank for processor
        world_size: total number of processors
        master_ip: ip address for main node for communication
        master_port: port for main node for communication
    '''

    os.environ["MASTER_ADDR"]=master_ip
    os.environ["MASTER_PORT"]=master_port

    init_process_group(
        backend=backend,
        world_size=world_size,
        rank=local_rank
    )
    torch.cuda.set_device(local_rank)

def data_loader_preparation(
        batch_size_train,
        batch_size_evaluation    
    ):
    '''
    return the distributed dataloader for MNIST
    '''

    train_dataset = MNIST(
        root="/workspace/practice/mnist",
        transform=transforms.ToTensor()
    )
    evaluation_dataset = MNIST(
        root="/workspace/practice/mnist",
        transform=transforms.ToTensor(),
        train=False
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=False,
        sampler=DistributedSampler(dataset=train_dataset, drop_last=True)
    )

    evaluation_dataloader = DataLoader(
        dataset=evaluation_dataset,
        batch_size=batch_size_evaluation,
        shuffle=False,
        sampler=DistributedSampler(dataset=evaluation_dataset, drop_last=True)
    )

    return train_dataloader,evaluation_dataloader

def model_preparation(
        num_classes=10,
        hidden_size=1024,
        local_gpu_device=torch.device("cuda:0")
    ):
    '''
    wrap the MNIST model into distributed model
    '''
    model = Model(
        output_features=num_classes,
        hidden_size=hidden_size
    ).to(local_gpu_device)
    return DistributedDataParallel(
        module=model,
        device_ids=[local_gpu_device]
    )

def train(self):
    for num_epoch in range(self.run_epoch, self.epoch):
        if(self.local_rank==0):
            self.iterator.set_description("Training:")
        self._run_epoch()
        self.data_loader.sampler.set_epoch(num_epoch)
        if((num_epoch+1)%self.evalute_every==0 and self.local_rank==0):
            self.evaluate(num_epoch)
            self.iterator.set_description("Training:")
            self._save_checkpoint(num_epoch)
        if(self.local_rank==0):
            self.iterator.update()
    self._save_checkpoint(num_epoch)

def main(rank, world_size, master_port):
    local_rank=rank
    setattr(Trainer,'train',train)
    ddp_initialization(
        local_rank=local_rank,
        world_size=world_size,
        master_port=master_port
    )
    
    train_dataloader,evaluation_dataloader=data_loader_preparation(
        batch_size_train=3000,
        batch_size_evaluation=300
    )
    model = model_preparation(
        local_gpu_device=torch.device(f"cuda:{local_rank}"),
        hidden_size=4096,
    )
    optimizer = torch.optim.SGD(
        params=model.module.parameters(),
        lr=1e-2
    )
    loss_func=torch.nn.functional.cross_entropy
    trainer = Trainer(
        train_data_loader=train_dataloader,
        evaluate_data_loader=evaluation_dataloader,
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        epoch=100,
        gpu_device=torch.device(f"cuda:{local_rank}"),
        evalute_every=10,
        resume_path="/workspace/practice/checkpoint.bin_60",
        local_rank=local_rank
    )

    trainer.train()

if __name__ == "__main__":

    torch.multiprocessing.spawn(
        main,
        args=(torch.cuda.device_count(),"40114"),
        nprocs=torch.cuda.device_count()
    )


