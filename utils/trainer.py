import os
import torch
from tqdm import tqdm, trange
from datetime import datetime
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self,
                 num_epochs: int,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim,
                 trainloader: torch.utils.data.DataLoader = None,
                 testloader: torch.utils.data.DataLoader = None,
                 lr_scheduler: list = None,
                 lr_threshold: float = None,
                 pretrained_model: bool = False,
                 device: torch.device = None,
                 model_path: str = None,
                 start_epoch: int = 0,
                 early_stopping: bool = False,
                 patience: int = None,
                 min_delta: float = None,
                 start_epoch_early_stopping: int = None):
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader
        self.lr_scheduler = lr_scheduler
        self.lr_threshold = lr_threshold
        self.device = device
        self.model_path = model_path
        self.early_stopping = early_stopping

        self.L1_score = torch.nn.L1Loss()

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available()
                                       else "cpu")

        self.model.to(self.device)

        if pretrained_model:
            self._load_model()

        today = datetime.today()
        self.today = today.strftime("%y-%m-%d/%H:%M:%S")

        self.log_dir = os.path.join("logs", str(self.today))
        os.makedirs(self.log_dir)

        if self.early_stopping:
            self.patience = patience
            self.min_delta = min_delta
            self.start_epoch_early_stopping = start_epoch_early_stopping
            if self.patience is None:
                self.patience = 5
            if self.min_delta is None:
                self.min_delta = 0
            if self.start_epoch_early_stopping is None:
                self.start_epoch_early_stopping = 0

    def train(self):
        cost_value = []
        cost_test_value = []
        best_loss_value = float("inf")
        best_val_loss_value = float("inf")
        patience_counter = 0

        self.writer = SummaryWriter(log_dir=self.log_dir)
        for epoch in trange(self.num_epochs):
            curr_epoch = self.start_epoch + epoch
            train_loss, L1_score = self._train_step()
            test_loss = self._evaluate_step()
            if train_loss < best_loss_value:
                if self.early_stopping:
                    if curr_epoch < self.start_epoch_early_stopping:
                        self._save_model(epoch=curr_epoch,
                                         train_loss=train_loss)
                        best_loss_value = train_loss
                else:
                    self._save_model(epoch=curr_epoch,
                                     train_loss=train_loss)
                    best_loss_value = train_loss
            if test_loss < best_val_loss_value - self.min_delta:
                best_val_loss_value = test_loss
                patience_counter = 0
                if self.early_stopping:
                    if curr_epoch >= self.start_epoch_early_stopping:
                        self._save_model(epoch=curr_epoch,
                                         train_loss=train_loss)
                        best_loss_value = train_loss
            else:
                if self.early_stopping:
                    if curr_epoch >= self.start_epoch_early_stopping:
                        patience_counter += 1
                        print("Patience counter: ", patience_counter)
                        if patience_counter >= self.patience:
                            print(f"Early stopping at epoch {curr_epoch}")
                            break
            cost_value.append(train_loss)
            cost_test_value.append(test_loss)
            self._tracking_model(epoch=curr_epoch,
                                 train_loss=train_loss,
                                 test_loss=test_loss,
                                 L1_score=L1_score)
        self.writer.close()
        self._plot_loss(cost_value, cost_test_value)

    def evaluate(self):
        if self.testloader is not None:
            cost_test_value = []
            cost_test_value.append(self._evaluate_step(cost_test_value))
            self._plot_loss(cost_test_value)
        else:
            raise ValueError("Testloader is None")

    def _train_step(self):
        self.model.train()
        losses = []
        mae = []
        avg_loss = 0.0
        avg_mae_loss = 0.0

        for batch_idx, (image, label) in enumerate(tqdm(self.trainloader)):
            image = image.to(self.device)
            label = label.to(self.device)
            # with torch.autocast(device_type=self.device,
            #                     dtype=torch.bfloat16):

            prediction = self.model(image).reshape(image.shape[0],
                                                   68, 2)
            loss = self.criterion(prediction, label)
            mae_ls = self.L1_score(prediction, label)
            losses.append(loss.item())
            mae.append(mae_ls.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.lr_scheduler:
            for scheduler in self.lr_scheduler:
                print("Learning rate: ", scheduler.get_last_lr())
                if scheduler.get_last_lr()[0] <= self.lr_threshold:
                    break
                scheduler.step()
        for index in range(len(losses)):
            avg_loss += losses[index]/len(losses)
            avg_mae_loss += mae[index]/len(mae)
        print("Train loss: ", avg_loss)
        print("L1 score: ", avg_mae_loss)
        print()
        return avg_loss, avg_mae_loss

    def _evaluate_step(self):
        self.model.eval()
        losses = []
        avg_loss = 0.0
        with torch.inference_mode():
            for batch_idx, (image, label) in enumerate(tqdm(self.testloader)):
                image = image.to(self.device)
                label = label.to(self.device)
                prediction = self.model(image).reshape(image.shape[0], 68, 2)
                loss = self.criterion(prediction, label)
                losses.append(loss.item())
        for ls in losses:
            avg_loss += ls/len(losses)
        print("Test loss: ", avg_loss)
        print()
        return avg_loss

    def _save_model(self,
                    epoch: int,
                    train_loss: float):
        torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_state_dict": [lr.state_dict()
                                      for lr in self.lr_scheduler],
                    "loss": train_loss
                    }, f"{self.log_dir}/model.pth")
        print()
        print("Model saved")

    def _tracking_model(self,
                        epoch: int,
                        train_loss: float,
                        test_loss: float,
                        L1_score: float):
        self.writer.add_scalars(main_tag="Loss",
                                tag_scalar_dict={
                                    "train_loss": train_loss,
                                    "test_loss": test_loss
                                },
                                global_step=epoch)
        self.writer.add_scalar(tag="L1 loss",
                               scalar_value=L1_score,
                               global_step=epoch)

    def _plot_loss(self, cost_values, cost_test_values):
        plt.plot(cost_values, label="train_loss")
        plt.plot(cost_test_values, label="test_loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        os.makedirs(f"../assets/{self.today}")
        plt.savefig(f"../assets/{self.today}/loss.png")
        plt.show()

    def _load_model(self):
        if self.model_path is None:
            raise ValueError("Model path is None")
        checkpoint = torch.load(self.model_path, weights_only=False)
        self.start_epoch = checkpoint["epoch"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler = [lr.load_state_dict(checkpoint["lr_state_dict"][i])
                             for i, lr in zip(len(checkpoint["lr_state_dict"]),
                                              self.lr_scheduler)]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded")
