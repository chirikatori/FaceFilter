import cv2
import numpy as np
import utils.filters.faceBlendCommon as fbc
import csv
import math
import utils.filters.filter_config as config
from utils import detect_faces
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


class Filter():
    def __init__(self,
                 detector,
                 landmark_model,
                 filter_name: str = None,  # multiple choices
                 input_size: int = 224,
                 count: int = 0,
                 isFirstFrame: bool = True,
                 sigma: int = 50,
                 visualize_face_points: bool = False):
        self.detector = detector
        self.landmark_model = landmark_model
        self.filter_name = filter_name
        self.count = count
        self.isFirstFrame = isFirstFrame
        self.sigma = sigma
        self.visualize_face_points = visualize_face_points
        self.transform_func = A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(),
            ToTensorV2()
        ])

    def apply_filter(self, source=0, output_path=None):
        cap = cv2.VideoCapture(source)
        # cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if source != 0 and output_path is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path / 'annotated_video.mp4',
                                  fourcc, fps, frame_size)
        if self.filter_name is None:
            iter_filter_keys = iter(config.filters_config.keys())
            self.filter_name = next(iter_filter_keys)
        points2Prev, img2GrayPrev = None, None

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret is False:
                break
            if source == 0:
                frame = cv2.flip(frame, 1)
            frame, points2Prev, img2GrayPrev = self._filter_frame_(frame,
                                                                   points2Prev,
                                                                   img2GrayPrev) # noqa
            if source == 0:
                cv2.imshow("Video", frame)
                keypressed = cv2.waitKey(1) & 0xFF
                if keypressed == 27:
                    break
                elif keypressed == ord('f'):
                    try:
                        self.filter_name = next(iter_filter_keys)
                    except: # noqa
                        iter_filter_keys = iter(config.filters_config.keys())
                        self.filter_name = next(iter_filter_keys)
            else:
                if output_path is not None:
                    out.write(frame)
        cap.release()
        if source != 0 and output_path is not None:
            out.release()
        cv2.destroyAllWindows()

    def _filter_frame_(self, frame, points2Prev, img2GrayPrev):
        filters, multi_filter_runtime = self._load_filter_()

        multiple_face_points = self._get_landmark_(frame)
        for points2 in multiple_face_points:
            img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if points2Prev is None and img2GrayPrev is None:
                points2Prev = np.array(points2, np.float32)
                img2GrayPrev = np.copy(img2Gray)
            lk_params = dict(winSize=(101, 101), maxLevel=15,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001)) # noqa
            points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev,
                                                            img2Gray,
                                                            points2Prev,
                                                            np.array(points2, np.float32), # noqa
                                                            **lk_params)
            # Final landmark points are a weighted average of detected
            # landmarks and tracked landmarks
            for k in range(0, len(points2)):
                d = cv2.norm(np.array(points2[k]) - points2Next[k])
                alpha = math.exp(-d * d / 50)
                points2[k] = (1 - alpha) * np.array(points2[k]) + alpha * points2Next[k] # noqa
                points2[k] = fbc.constrainPoint(points2[k], frame.shape[1],
                                                frame.shape[0])
                points2[k] = (int(points2[k][0]), int(points2[k][1]))

            # Update variables for next pass
            points2Prev = np.array(points2, np.float32)
            img2GrayPrev = img2Gray

            # applying filter
            for idx, filter in enumerate(filters):
                filter_runtime = multi_filter_runtime[idx]
                img1 = filter_runtime['img']
                points1 = filter_runtime['points']
                img1_alpha = filter_runtime['img_a']
                if filter["morph"]:
                    hull1 = filter_runtime['hull']
                    hullIndex = filter_runtime['hullIndex']
                    dt = filter_runtime['dt']
                    # create copy of frame
                    warped_img = np.copy(frame)
                    # Find convex hull
                    hull2 = []
                    for i in range(0, len(hullIndex)):
                        hull2.append(points2[hullIndex[i][0]])
                    mask1 = np.zeros((warped_img.shape[0],
                                      warped_img.shape[1]),
                                     dtype=np.float32)
                    mask1 = cv2.merge((mask1, mask1, mask1))
                    img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha,
                                                 img1_alpha))
                    # Warp the delaunay triangles
                    for i in range(0, len(dt)):
                        t1 = []
                        t2 = []
                        for j in range(0, 3):
                            t1.append(hull1[dt[i][j]])
                            t2.append(hull2[dt[i][j]])
                        fbc.warpTriangle(img1, warped_img, t1, t2)
                        fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)
                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
                    mask2 = (255.0, 255.0, 255.0) - mask1
                    # Perform alpha blending of the two images
                    temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2
                else:
                    dst_points = [points2[int(list(points1.keys())[0])],
                                  points2[int(list(points1.keys())[1])]]
                    tform = fbc.similarityTransform(list(points1.values()),
                                                    dst_points)
                    # Apply similarity transform to input image
                    trans_img = cv2.warpAffine(img1, tform, (frame.shape[1],
                                                             frame.shape[0]))
                    trans_alpha = cv2.warpAffine(img1_alpha, tform,
                                                 (frame.shape[1],
                                                  frame.shape[0]))
                    mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))
                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
                    mask2 = (255.0, 255.0, 255.0) - mask1
                    # Perform alpha blending of the two images
                    temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2

                frame = output = np.uint8(output)

        return frame, points2Prev, img2GrayPrev

    def _load_filter_(self):
        filters = config.filters_config[self.filter_name]
        multi_filter_runtime = []
        for f in filters:
            temp_dict = {}
            img, img_alpha = self._load_filter_image_(f["path"],
                                                      f["has_alpha"])
            temp_dict["img"] = img
            temp_dict["img_a"] = img_alpha

            points = self._load_landmarks_(f["anno_path"])
            temp_dict["points"] = points
            if f["morph"]:
                hull, hullIndex = self._find_convex_hull_(points)

                sizeImg = img.shape
                rect = (0, 0, sizeImg[1], sizeImg[0])
                dt = fbc.calculateDelaunayTriangles(rect, hull)

                temp_dict["hull"] = hull
                temp_dict["hullIndex"] = hullIndex
                temp_dict["dt"] = dt

                if len(dt) == 0:
                    continue

            if f["animated"]:
                filter_cap = cv2.VideoCapture(f["path"])
                temp_dict["cap"] = filter_cap
            multi_filter_runtime.append(temp_dict)
        return filters, multi_filter_runtime

    def _find_convex_hull_(self, points):
        hull = []
        hullIndex = cv2.convexHull(np.array(list(points.values())),
                                   clockwise=False,
                                   returnPoints=False)
        hullIndex = hullIndex[:15]
        addPoints = [
            [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], # Outer lips # noqa
            [60], [61], [62], [63], [64], [65], [66], [67], # Inner lips # noqa
            [27], [28], [29], [30], [31], [32], [33], [34], [35], # Nose # noqa
            [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], # Eyes # noqa
            [17], [18], [19], [20], [21], [22], [23], [24], [25], [26] # EyeBrows # noqa
        ]
        hullIndex = np.concatenate((hullIndex, addPoints))
        for i in range(0, len(hullIndex)):
            hull.append(points[str(hullIndex[i][0])])

        return hull, hullIndex

    def _load_filter_image_(self, img_path, has_alpha):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        alpha = None
        if has_alpha:
            b, g, r, alpha = cv2.split(img)
            img = cv2.merge((b, g, r))
        return img, alpha

    def _load_landmarks_(self, annotation_file):
        with open(annotation_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            points = {}
            for i, row in enumerate(csv_reader):
                try:
                    x, y = int(row[1]), int(row[2])
                    points[row[0]] = (x, y)
                except ValueError:
                    continue
            return points

    def _get_landmark_(self, frame):
        crop, boxes, scores, cls = detect_faces(self.detector,
                                                [Image.fromarray(frame)],
                                                box_format="xywh", th=0.4)
        multiple_face_keypoints = []
        for box in boxes[0]:
            left, top, width, height = box
            cropped_frame = frame[top: top + height, left: left + width]
            cropped_frame = self.transform_func(image=cropped_frame)["image"]
            cropped_frame = cropped_frame.permute(1, 2, 0)

            input_img = cropped_frame.permute(2, 0, 1).unsqueeze(0).cuda()
            with torch.no_grad():
                self.landmark_model.eval()
                predicted_keypoints = self.landmark_model(input_img)
            keypoints = predicted_keypoints.view(-1, 68, 2)
            keypoints = keypoints.detach().cpu()
            keypoints = keypoints.squeeze().numpy()
            keypoints = keypoints.reshape((68, 2))
            keypoints[:, 0] = (keypoints[:, 0] + 0.5)*width + left
            keypoints[:, 1] = (keypoints[:, 1] + 0.5)*height + top
            multiple_face_keypoints.append(keypoints)
        return multiple_face_keypoints
