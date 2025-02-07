from utils import Visualizer # ,detect_faces # noqa
from utils.models import resnet50 #,yolov8_medium # noqa
from utils.data import LoadData

import torch
import hydra
from omegaconf import DictConfig
from PIL import Image # noqa
import cv2 # noqa
import albumentations as A # noqa
from albumentations.pytorch import ToTensorV2 # noqa
import numpy as np # noqa


@hydra.main(version_base="1.3",
            config_name="config",
            config_path="config")
def main(cfg: DictConfig) -> None:
    checkpoint = torch.load("logs/24-11-05/14:28:19/model.pth",
                            weights_only=False,
                            map_location=torch.device("cpu"))
    model = resnet50(num_classes=cfg.num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_data = LoadData(cfg.testset.path).load()
    # train_data = LoadData(cfg.trainset.path).load()

    visualizer_test = Visualizer(dataset=test_data,
                                 data_path=cfg.dataset.path,
                                 input_size=cfg.input_size,
                                 model=model)
    # visualizer_train = Visualizer(dataset=train_data,
    #                               data_path=cfg.dataset.path,
    #                               input_size=cfg.input_size,
    #                               model=model)
    # visualizer_train.visual_random_predicted_image(5)
    visualizer_test.visual_random_predicted_image(5)

    # image = Image.open("assets/experiment.jpg")
    # image = cv2.imread("assets/experiment.jpg")
    # detector = yolov8_medium("cuda")
    # # image_array = np.array(image)
    # crops, boxes, scores, cls = detect_faces(detector,
    #                                          [Image.fromarray(image)],
    #                                          box_format='xywh', th=0.4)
    # print(boxes)
    # left, top, width, height = boxes[0][0]
    # cropped_image = image[top: top + height, left: left + width]
    # cropped_image = np.array(cropped_image)
    # resize_func = A.Compose([
    #     A.Normalize(),
    #     A.Resize(cfg.input_size, cfg.input_size),
    #     ToTensorV2()
    # ])
    # cropped_image = resize_func(image=cropped_image)["image"]
    # cropped_image = cropped_image.unsqueeze(0)
    # keypoints = model(cropped_image)
    # # print(keypoints)
    # keypoints = keypoints.detach().numpy()
    # print(keypoints)
    # keypoints = keypoints.reshape((68, 2))
    # keypoints[:, 0] = (keypoints[:, 0] + 0.5)*width + left
    # keypoints[:, 1] = (keypoints[:, 1] + 0.5)*height + top

    # # cv2.imshow(image)
    # # plt.scatter(x=keypoints[:, 0], y=keypoints[:, 1], s=1, color="red")
    # for (x, y) in zip(keypoints[:, 0], keypoints[:, 1]):
    #     cv2.circle(image, (int(x), int(y)), 1, (0, 0, 225))
    # cv2.imshow("image", image)

    # # plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
