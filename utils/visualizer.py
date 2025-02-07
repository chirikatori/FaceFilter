import matplotlib.pyplot as plt
from PIL import Image
import torch
import pandas as pd
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torchvision import transforms


class Visualizer():
    def __init__(self,
                 dataset: pd.DataFrame,
                 data_path: str,
                 input_size: int,
                 model: torch.nn.Module = None,
                 transform=None):
        self.dataset = dataset
        self.data_path = data_path
        self.model = model
        self.input_size = input_size
        self.transform = transform
        if self.transform is None:
            self.transform = A.Compose([
                A.RGBShift(r_shift_limit=15,
                           g_shift_limit=15,
                           b_shift_limit=15),
                # A.HorizontalFlip(),
                # A.VerticalFlip(),
                A.RandomBrightnessContrast(),
                # A.GaussNoise(),
                A.Blur(),
                A.Resize(self.input_size, self.input_size),
                A.Normalize(),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy',
                                                remove_invisible=False))

    def visual_raw_data(self, idx):
        image = self.dataset["image"].loc[idx]
        image = f"{self.data_path}/{image}"
        image = Image.open(image)
        X_points = pd.Series(self.dataset["pointX"].loc[idx])
        Y_points = pd.Series(self.dataset["pointY"].loc[idx])
        self._print_image_with_points_(image, [X_points, Y_points])
        image.close()

    def visual_random_raw_data(self):
        idx = torch.randint(low=0,
                            high=self.dataset.shape[0],
                            size=(1,)).item()
        image = self.dataset["image"].loc[idx]
        image = f"{self.data_path}/{image}"
        image = Image.open(image)
        X_points = pd.Series(self.dataset["pointX"].loc[idx])
        Y_points = pd.Series(self.dataset["pointY"].loc[idx])
        self._print_image_with_points_(image, [X_points, Y_points])
        image.close()

    def visual_preprocessed_data(self, idx):
        image = self._process_image_(idx)
        image = image.squeeze()
        box_top = self.dataset["box_top"].loc[idx]
        box_left = self.dataset["box_left"].loc[idx]
        box_width = self.dataset["box_width"].loc[idx]
        box_height = self.dataset["box_height"].loc[idx]
        X_points = pd.Series(self.dataset["pointX"].loc[idx])
        Y_points = pd.Series(self.dataset["pointY"].loc[idx])
        X_points = (X_points-box_left)*self.input_size/box_width
        Y_points = (Y_points-box_top)*self.input_size/box_height
        self._print_image_with_points_(image, [X_points, Y_points])

    def visual_random_preprocessed_data(self):
        idx = torch.randint(low=0,
                            high=self.dataset.shape[0],
                            size=(1,)).item()
        image = self._process_image_(idx)
        image = image.squeeze()
        box_top = self.dataset["box_top"].loc[idx]
        box_left = self.dataset["box_left"].loc[idx]
        box_width = self.dataset["box_width"].loc[idx]
        box_height = self.dataset["box_height"].loc[idx]
        X_points = pd.Series(self.dataset["pointX"].loc[idx])
        Y_points = pd.Series(self.dataset["pointY"].loc[idx])
        X_points = (X_points-box_left)*self.input_size/box_width
        Y_points = (Y_points-box_top)*self.input_size/box_height
        self._print_image_with_points_(image, [X_points, Y_points])

    def visual_predicted_image(self, idx):
        input_image = self._process_image_(idx)
        pred = self._get_pred_(input_image)

        box_top = self.dataset["box_top"].loc[idx]
        box_left = self.dataset["box_left"].loc[idx]
        box_width = self.dataset["box_width"].loc[idx]
        box_height = self.dataset["box_height"].loc[idx]
        box_right = box_left + box_width
        box_bot = box_top + box_height

        image = self.dataset["image"].loc[idx]
        image = f"{self.data_path}/{image}"
        image = Image.open(image)
        image = image.crop((box_left, box_top, box_right, box_bot))
        image = image.resize((self.input_size, self.input_size))
        label_X = np.array(self.dataset["pointX"].loc[idx])
        label_X = (label_X-box_left)*self.input_size/box_width
        label_Y = np.array(self.dataset["pointY"].loc[idx])
        label_Y = (label_Y-box_top)*self.input_size/box_height
        print("Label X", label_X)
        print("Label Y", label_Y)

        pred = pred.reshape((68, 2))
        print("Pred: ", pred)
        X_points = np.array(pred[:, 0])
        X_points += 0.5
        X_points *= self.input_size
        # X_points += box_left
        print("X points", X_points)

        Y_points = np.array(pred[:, 1])
        Y_points += 0.5
        Y_points *= self.input_size
        # Y_points += box_top
        print("Y points", Y_points)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image)
        ax1.scatter(x=label_X, y=label_Y, s=1, color="red")
        ax1.set_title('True value:')
        ax2.imshow(image)
        ax2.scatter(x=X_points, y=Y_points, s=1, color="red")
        ax2.set_title("Predict:")
        fig.show()
        plt.show()

    def visual_random_predicted_image(self,
                                      num_of_pics: int):
        for _ in range(num_of_pics):
            idx = torch.randint(low=0,
                                high=self.dataset.shape[0],
                                size=(1,)).item()
            input_image = self._process_image_(idx)
            pred = self._get_pred_(input_image)

            box_top = self.dataset["box_top"].loc[idx]
            box_left = self.dataset["box_left"].loc[idx]
            box_width = self.dataset["box_width"].loc[idx]
            box_height = self.dataset["box_height"].loc[idx]

            image = self.dataset["image"].loc[idx]
            image = f"{self.data_path}/{image}"
            image = Image.open(image)
            # image = image.crop((box_left, box_top, box_right, box_bot))
            # image = image.resize((self.input_size, self.input_size))
            label_X = np.array(self.dataset["pointX"].loc[idx])
            # label_X = (label_X-box_left)*self.input_size/box_width
            label_Y = np.array(self.dataset["pointY"].loc[idx])
            # label_Y = (label_Y-box_top)*self.input_size/box_height
            print("Label X", label_X)
            print("Label Y", label_Y)

            pred = pred.reshape((68, 2))
            print("Pred: ", pred)
            X_points = np.array(pred[:, 0])
            X_points += 0.5
            # X_points /= self.input_size
            X_points *= box_width
            X_points += box_left
            print("X points", X_points)

            Y_points = np.array(pred[:, 1])
            Y_points += 0.5
            # Y_points /= self.input_size
            Y_points *= box_height
            Y_points += box_top
            print("Y points", Y_points)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(image)
            ax1.scatter(x=label_X, y=label_Y, s=1, color="red")
            ax1.set_title('True value:')
            ax2.imshow(image)
            ax2.scatter(x=X_points, y=Y_points, s=1, color="red")
            ax2.set_title("Predict:")
            fig.show()
            plt.show()

    def visual_augmented_image(self, idx):
        image = self.dataset["image"].loc[idx]
        image = f"{self.data_path}/{image}"
        image = Image.open(image)
        image = image.convert("RGB")
        label = []
        label.extend(self.dataset["pointX"].loc[idx])
        label.extend(self.dataset["pointY"].loc[idx])
        self._print_augmented_image(image, label)

    def visual_random_augmented_image(self):
        idx = torch.randint(low=0,
                            high=self.dataset.shape[0],
                            size=(1,)).item()
        image = self.dataset["image"].loc[idx]
        image = f"{self.data_path}/{image}"
        image = Image.open(image)
        image = image.convert("RGB")
        label = []
        label.extend(self.dataset["pointX"].loc[idx])
        label.extend(self.dataset["pointY"].loc[idx])
        self._print_augmented_image(image, label)

    def _print_augmented_image(self, image, label):
        image = np.array(image)
        keypoints = []
        for x, y in zip(label[:68], label[68:]):
            keypoints.append((x, y))
        augmentation = self.transform(image=image, keypoints=keypoints)
        augmented_image = augmentation["image"]
        augmented_label = augmentation["keypoints"]

        x_list, y_list = [], []
        for (x, y) in augmented_label:
            x_list.append(x)
            y_list.append(y)

        T = transforms.ToPILImage()
        augmented_image = T(augmented_image)

        plt.imshow(augmented_image)
        plt.scatter(x_list, y_list, s=1, color="red")
        plt.show()

    def _process_image_(self, idx):
        image = self.dataset["image"].loc[idx]
        image = f"{self.data_path}/{image}"
        image = Image.open(image)
        image = image.convert("RGB")

        box_top = self.dataset["box_top"].loc[idx]
        box_left = self.dataset["box_left"].loc[idx]
        box_width = self.dataset["box_width"].loc[idx]
        box_height = self.dataset["box_height"].loc[idx]

        box_right = box_left + box_width
        box_bot = box_top + box_height

        image = image.crop((box_left, box_top, box_right, box_bot))
        # image = image.resize((self.input_size, self.input_size))
        # T = transforms.ToTensor()
        # image = T(image)
        # image = image.float()
        # image = image.unsqueeze(dim=0)
        # print(image.shape)
        image = np.array(image)
        transform_img = A.Compose([
            A.Resize(self.input_size, self.input_size),
            A.Normalize(),
            ToTensorV2()
        ])
        image = transform_img(image=image)["image"]
        image = image.unsqueeze(dim=0)
        return image

    def _print_image_with_points_(self, image,
                                  points: tuple[pd.Series, pd.Series]):
        image = image.permute(1, 2, 0)
        plt.imshow(image)
        plt.scatter(x=points[0], y=points[1], s=1, color="red")
        plt.show()

    def _get_pred_(self, input_image):
        self.model.eval()
        with torch.inference_mode():
            prediction = self.model(input_image)
        return torch.clone(prediction)
