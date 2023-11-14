import torch

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(
            self,
            train_data_loader,
            evaluate_data_loader,
            model,
            optimizer,
            loss_func,
            epoch,
            gpu_device,
            evalute_every,
            resume_path=None,
            local_rank=None
    ) -> None:
        self.data_loader = train_data_loader
        self.model:torch.nn.Module = model
        self.optim = optimizer
        self.loss_func = loss_func
        self.epoch = epoch
        self.gpu_device = gpu_device
        self.evaluate_data_loader = evaluate_data_loader
        self.local_rank = local_rank
        if(self.local_rank==0):
            self.iterator=tqdm(
                total=self.epoch
            )
        else:
            self.iterator = None
        self.evalute_every = evalute_every
        self.model.to(self.gpu_device)
        if(resume_path):
            if(self.local_rank==0):
                self.iterator.write(f"resume training from {resume_path}")
            self._resume_from_checkpoint(resume_path)
        else:
            if(self.local_rank==0):
                self.iterator.write("training from stratch")
            self.run_epoch=0

    def _forward_calculate(self, batch):
        '''
        forward calculation
        ----
        input: batch: a batch of training data
        output: output of the model
        '''
        return self.model(batch)
    
    def _loss_calculate(self, prediction, label):
        '''
        loss calculation
        ----
        input:
            prediction: model output
            label: ground truth
        output:
            loss of the batch of data
        '''
        return self.loss_func(prediction, label)
    
    def _backward_propagation(self, loss):
        '''
        backward propagation
        ----
        input:
            loss: loss
        output:
            None
        '''
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    
    def _learning_once(self, inputs, label):
        '''
        learning once from the batch of data
        ----
        input:
            inputs: batch of inputs data
            label: ground truth for the data
        output:
            None
        '''
        model_output = self._forward_calculate(inputs)
        loss = self._loss_calculate(model_output, label)
        if(self.local_rank==0):
            self.iterator.set_postfix({'loss':"{:.2f}".format(loss.item())})
        self._backward_propagation(loss)
    
    def _run_epoch(self):
        '''
        running for one epoch
        '''
        for batch,label in self.data_loader:
            batch = batch.to(self.gpu_device)
            label = label.to(self.gpu_device)
            self._learning_once(batch,label)
            
    def train(self):
        for num_epoch in range(self.run_epoch, self.epoch):
            self._run_epoch()
            
            if((num_epoch+1)%self.evalute_every==0):
                self.evaluate(num_epoch)
                if(self.local_rank==0):
                    self.iterator.set_description("Training:")
            
                self._save_checkpoint(num_epoch)
            if(self.local_rank==0):
                self.iterator.update()
        self._save_checkpoint(num_epoch)
                
    def evaluate(self, current_epoch):
        if(self.local_rank==0):
            self.iterator.set_description("Evaluation:")
            self.iterator.set_postfix_str("")
        for batch,label in self.evaluate_data_loader:
            batch = batch.to(self.gpu_device)
            label = label.to(self.gpu_device)  
            accs = []          
            with torch.no_grad():
                output = self._forward_calculate(batch)
                predictions = self._get_predictions(output)
                accuracy = self._get_accuracy(predictions, label)
                accs.append(accuracy)
        if(self.local_rank==0):
            self.iterator.write(f"Epoch {current_epoch}: accuracy: {sum(accs)/len(accs):.2f}")

    def _get_predictions(self,output):
        '''
        get predictions of model
        '''
        return torch.argmax(output, dim=-1)

    def _get_accuracy(self, predictions, label):
        '''
        calculate the accuracy of inference
        '''
        return (sum(predictions==label)/len(predictions)).item()*100
        
    def _save_checkpoint(self,num_epoch):
        ckp = self.model.state_dict()
        tings2save = {
            "state_dict":ckp,
            "epoch":num_epoch
        }
        if(self.local_rank==0):
            self.iterator.write(f"saving checkpoint.bin_{num_epoch}")
        torch.save(tings2save, f"checkpoint.bin_{num_epoch}")
    
    def _resume_from_checkpoint(self, resume_chekcpoint_file):

        ckpt = torch.load(resume_chekcpoint_file, map_location=self.gpu_device)
        self.run_epoch = ckpt["epoch"]+1
        self.model.load_state_dict(ckpt["state_dict"])
        if(self.local_rank==0):
            self.iterator.update(self.run_epoch)


class Data(Dataset):
    def __init__(
            self,
            random:bool=False,
            random_shape:tuple=None,
            random_label_shape:tuple=None
    ) -> None:
        '''
        Dataset
        ----
        random: whether use random data
        random_size: if use random data, the shape of your random dataset
        random_label_shape: if use random data, the shape of the label correspoding to one sample
        '''
        super().__init__()
        if(random):
            self.input_dataset = torch.rand(
                size=random_shape
            )
            self.label_set = torch.rand(
                size=random_label_shape
            )

    
    def __getitem__(self, index):
        return self.input_dataset[index],self.label_set[index]
    
    def __len__(self):
        return len(self.input_dataset)

class Model(torch.nn.Module):
    def __init__(self, 
                 input_features=224,
                 output_features=1,
                 hidden_size=1024,
                 *args, **kwargs) -> None:
        '''
        model definition
        ----
        input:
            input_features:    the last dimension of input data
            output_features:    the last dimension of output of model
            hidden_size: output size or input size in the middle of the model

        '''
        super().__init__(*args, **kwargs)
        
        self.model = torch.nn.ModuleList([
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(3,3)
            ),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=2),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3,3)
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=3872,out_features=hidden_size),
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=hidden_size,out_features=hidden_size),
            torch.nn.Dropout(),
            torch.nn.Linear(in_features=hidden_size, out_features=output_features)
        ])

    def forward(self, inputs):
        for module in self.model:
            inputs = module(inputs)
        return inputs
    
def main():
    dataset_shape=(4096, 784)
    label_shape=(4096,1)
    batch_size=6000
    learning_rate=1e-2
    epoch=200
    # dataset = Data(
    #     random=True,
    #     random_shape=dataset_shape,
    #     random_label_shape=label_shape
    # )
    train_dataset = MNIST(
        root="/workspace/practice/mnist",
        transform=transforms.Compose([
            transforms.ToTensor(),
            # torch.nn.Flatten(),
        ])
    )
    evaluation_dataset = MNIST(
        root="/workspace/practice/mnist",
        transform=transforms.Compose([
            transforms.ToTensor(),
            # torch.nn.Flatten(),
        ]),
        train=False        
    )
    train_data_loader = DataLoader(
        dataset=evaluation_dataset,
        batch_size=batch_size
    )
    evaluation_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size
    )    
    model = Model(
        input_features=784,
        output_features=10,
        hidden_size=4096
    )
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=learning_rate
    )
    loss_func=torch.nn.functional.cross_entropy
    
    trainer = Trainer(
        train_data_loader=train_data_loader,
        evaluate_data_loader=evaluation_data_loader,
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        epoch=epoch,
        gpu_device=torch.device("cuda:0"),
        evalute_every=20,
        resume_path="/workspace/practice/checkpoint.bin_160",
        local_rank=0
    )

    trainer.train()

if __name__ == "__main__":
    main()
