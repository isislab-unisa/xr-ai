import os
from typing import Dict
import argparse

import torch
from torch import nn
from torchvision import transforms, models
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm

import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

import numpy as np


def get_datasets(input_path):
    num_classes = len(os.listdir(os.path.join(input_path, "train")))
    train_ds = ray.data.read_images("local://" + os.path.join(input_path, "train"))
    test_ds = ray.data.read_images("local://" + os.path.join(input_path, "test"))
    return train_ds, test_ds, num_classes

def transform_data(train_ds, test_ds):
    train_transform = transforms.Compose(
        [
            ToTensor(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_transform = (
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    )
    
    def transform_image_train(row: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        row["transformed_image"] = train_transform(row["image"])
        return row
    
    def transform_image_test(row: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        row["transformed_image"] = test_transform(row["image"])
        return row
    
    train_ds = train_ds.map(transform_image_train)
    test_ds = test_ds.map(transform_image_test)

    return train_ds, test_ds

def get_dataloaders():
    train_data_shard = train.get_dataset_shard("train")
    test_data_shard = train.get_dataset_shard("test")
    train_dataloader = train_data_shard.iter_torch_batches(batch_size=4, dtypes=torch.float32)
    test_dataloader = test_data_shard.iter_torch_batches(batch_size=4, dtypes=torch.float32)
    return train_dataloader, test_dataloader

def train_func_per_worker(config: Dict):
    train_dataloader, test_dataloader = get_dataloaders()
    print("DATA OK")	
    model_ft = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)
    num_ftrs = model_ft.head.in_features
    model_ft.head = nn.Linear(num_ftrs, config["num_classes"])
    model = model_ft
    model = ray.train.torch.prepare_model(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print("MODEL OK")
    
    os.makedirs(config["output_dir"], exist_ok=True)
    final_model_path = config["output_dir"] + "/model.pth"

    print("TRAINING START")
    
    for epoch in range(config["epochs"]):
        if ray.train.get_context().get_world_size() > 1:
            train_dataloader.sampler.set_epoch(epoch)
            
        model.train()
        for X, y in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        test_loss, num_correct, num_total = 0, 0, 0
        with torch.no_grad():
            for X, y in tqdm(test_dataloader, desc=f"Test Epoch {epoch}"):
                pred = model(X)
                loss = loss_fn(pred, y)
                test_loss += loss.item()
                num_correct += (pred.argmax(1) == y).sum().item()
                num_total += len(y)
                
        test_loss /= len(test_dataloader)
        accuracy = num_correct / num_total
        
        ray.train.report(metrics={"loss": test_loss, "accuracy": accuracy})

def train_func(input_dir, output_dir, num_workers=2, use_gpu=False, num_epochs=25):
    global_batch_size = 32
    
    train_ds, test_ds, num_classes = get_datasets(input_dir)
    train_ds, test_ds = transform_data(train_ds, test_ds)
    
    train_config = {
        "lr" : 1e-3,
        "epochs" : num_epochs,
        "batch_size_per_worker" : global_batch_size // num_workers,
        "input_dir" : input_dir,
        "output_dir" : output_dir,
        "num_classes" : num_classes,	
    }
    
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        datasets={"train": train_ds, "test": test_ds},
    )
    
    result = trainer.fit()
    print(f"Training result: {result}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train script for image classification using PyTorch."
    )
    parser.add_argument("--input_dir", type=str, help="Path to the input directory.", required=True)
    parser.add_argument("--output_dir", type=str, help="Path to the output directory.", required=True)
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs to train the model.")
    args = parser.parse_args()
    
    ray.init()
    
    train_func(args.input_dir, args.output_dir, num_workers=1, use_gpu=True if torch.cuda.is_available() else False, num_epochs = args.num_epochs)
