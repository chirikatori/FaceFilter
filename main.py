from utils.models import yolov8_medium, resnet50
from utils.filters import Filter

import torch
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_name="config", config_path="config")
def main(cfg: DictConfig) -> None:
    detector = yolov8_medium("cuda")
    landmark_model = resnet50(num_classes=cfg.num_classes).to("cuda")

    # Load model weights
    model_weights = torch.load("utils/models/resnet_landmark.pth",
                               weights_only=False, map_location="cuda")
    landmark_model.load_state_dict(model_weights["model_state_dict"])

    filter_vid = Filter(detector=detector,
                        landmark_model=landmark_model)
    filter_vid.apply_filter()


if __name__ == "__main__":
    main()
