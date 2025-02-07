from ultralytics import YOLO
import torch
from typing import Literal


def get_device(device: Literal["cpu", "cuda"]):
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Cuda is not available, switch to 'cpu'")
            return torch.device("cpu")
    return torch.device("cpu")


# def yolov8_large(device: Literal["cpu", "cuda"]):
#     device = get_device(device)
#     model = YOLO("utils/models/yolov8l_100e.pt")
#     model.to(device)
#     return model


def yolov8_medium(device: Literal["cpu", "cuda"]):
    device = get_device(device)
    model = YOLO("utils/models/yolov8m_200e.pt")
    model.to(device)
    return model


# def yolov8_nano(device: Literal["cpu", "cuda"]):
#     device = get_device(device)
#     model = YOLO("utils/models/yolov8n_100e.pt")
#     model.to(device)
#     return model
