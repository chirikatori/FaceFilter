from utils import Trainer
from utils.data import (
    LandmarkDataset,
    GetData,
    LoadData
)
from torchvision.models import resnet18, resnet34, resnet50
import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch
from torch.utils.data import DataLoader


@hydra.main(version_base="1.3",
            config_name="config",
            config_path="config")
def training_pipeline(cfg: DictConfig) -> None:
    data_folder = Path(cfg.dataset.path)

    if data_folder.is_dir():
        print("Data already existed, begin training...")
    else:
        print("Preparing data...")
        GetData(raw_data_path="data/archive.zip",
                extract_path="data/").extract_zip_data()
        print("Data was prepared, ready to train...")

    train_data = LoadData(cfg.trainset.path).load()
    test_data = LoadData(cfg.testset.path).load()

    trainset = LandmarkDataset(train_data, cfg.dataset.path, cfg.input_size)
    testset = LandmarkDataset(test_data, cfg.dataset.path, cfg.input_size)

    trainloader = DataLoader(trainset,
                             batch_size=cfg.batch_size,
                             shuffle=True,
                             num_workers=cfg.num_workers)
    testloader = DataLoader(testset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers)

    if cfg.model.name == "resnet":
        model_name = "resnet" + str(cfg.model.num_layers)
        if model_name == "resnet18":
            model = resnet18(num_classes=cfg.num_classes)
        elif model_name == "resnet34":
            model = resnet34(num_classes=cfg.num_classes)
        else:
            model = resnet50(num_classes=cfg.num_classes)
    else:
        raise ValueError("Model hasn't been constructed!")

    params = model.parameters()

    if cfg.train_type == "transfer":
        for param in model.parameters():
            param.requires_grad = False
        model.fc = torch.nn.Linear(model.fc.in_features,
                                   cfg.num_classes)
        params = model.fc.parameters()

    criterion = hydra.utils.instantiate(cfg.criterion)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=params)
    lr_scheduler = hydra.utils.instantiate(cfg.lr_scheduler,
                                           optimizer=optimizer)

    trainer = Trainer(num_epochs=cfg.num_epochs,
                      model=model,
                      criterion=criterion,
                      optimizer=optimizer,
                      trainloader=trainloader,
                      testloader=testloader,
                      lr_scheduler=[lr_scheduler],
                      lr_threshold=cfg.lr_threshold,
                      #   pretrained_model=True,
                      #   model_path="logs/24-10-25_18:09:58/model.pth",
                      #   device=torch.device("cpu")
                      )

    trainer.train()


if __name__ == "__main__":
    training_pipeline()
