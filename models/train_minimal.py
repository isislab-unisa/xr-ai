from typing import Dict

import torch
import os


import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.train import RunConfig




def get_dataloaders(input_dir, batch_size, useGDrive):
    
    from torchvision import datasets, transforms
    import torch
    import gdown
    
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    if useGDrive:
        folder_id = "https://drive.google.com/drive/folders/1wnR8bbONwZp4Hlfj3pP1ShnN4yO_NfUl?usp=drive_link"
        paths = gdown.download_folder(folder_id)
        print(paths)
        
        
    else:
        image_dataset = {
            x: datasets.ImageFolder("local://" + os.path.join(input_dir, x), data_transforms[x])
            for x in ["train", "test"]
        }
    
    num_classes = len(image_dataset["train"].classes)

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_dataset[x], batch_size=batch_size, shuffle=True
        )
        for x in ["train", "test"]
    }
    
    return dataloaders["train"], dataloaders["test"], num_classes



def train_func_per_worker(config: Dict):
    
    from torchvision import models
    from tqdm import tqdm
    from torch import nn
    import os
    import torch

    train_dataloader, test_dataloader, num_classes = get_dataloaders(config["input_dir"], config["batch_size_per_worker"], config["useGDrive"])
    train_dataloader = ray.train.torch.prepare_dataloader(train_dataloader)
    test_dataloader = ray.train.torch.prepare_dataloader(test_dataloader)
    print("DATA OK")
    
    model_ft = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT)
    num_ftrs = model_ft.head.in_features
    model_ft.head = nn.Linear(num_ftrs, num_classes)
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


def train_func(input_dir, output_dir, num_workers=2, use_gpu=False, num_epochs=25, useGDrive=False):
    global_batch_size = 16

    train_config = {
        "lr": 1e-3,
        "epochs": num_epochs,
        "batch_size_per_worker": global_batch_size // num_workers,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "useGDrive": useGDrive,
    }

    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    
    # run_config = RunConfig(storage_path=input_dir, name="train_run")

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        # run_config = run_config,
    )

    result = trainer.fit()
    print(f"Training result: {result}")

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="Train script for image classification.")
    parser.add_argument(
        "--input_dir", type=str, help="Path to the input directory.", required=True
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to the output directory.", required=True
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=25,
        help="Number of epochs to train the model.",
    )
    
    parser.add_argument(
        "--useGDrive", type=bool, default=False, help="Use Google Drive for input/output directories."
    )
    
    args = parser.parse_args()
    
    return args.input_dir, args.output_dir, args.num_epochs, args.useGDrive

if __name__ == "__main__":

    input_dir, output_dir, num_epochs, useGDrive = parse_arguments()
    
    # ray.init()

    train_func(
        input_dir,
        output_dir,
        num_workers=1,
        use_gpu=True if torch.cuda.is_available() else False,
        num_epochs=num_epochs,
        useGDrive=useGDrive,
    )
