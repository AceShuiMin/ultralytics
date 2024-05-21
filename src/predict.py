"""
   推理功能上线版
"""
import argparse
import json
from pathlib import Path

import requests
from shapely.geometry import Polygon
import torch.cuda
import sys

sys.path.append("/usr/src/ultralytics")
sys.path.append("/usr/src/ultralytics/ultralytics")
sys.path.append("/usr/src/ultralytics/ultralytics/data")

from ultralytics.data.build import load_inference_source, check_source
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)
import requests
import yaml

WORK_DIR = "D:/PycharmProjects/ultralytics/"
# WORK_DIR = "/usr/src/ultralytics/"
RESULT_SAVE_DIR = "src/results/"
IMG_TEMP_DIR = "src/temp/"
REGION_CONFIG = WORK_DIR + "src/config/regions.yaml"
DEVICE_CONFIG = WORK_DIR + "src/config/device.yaml"


class CommonPredictor:

    def __init__(self, opt):
        self.source = opt.source
        self.weights = opt.weights
        self.classes = opt.classes
        # 推理模式，0--离线视频，1--在线视频流
        self.mode = opt.mode
        self.model = YOLO(f"{self.weights}")
        self.model.to("cuda") if torch.cuda.is_available() else self.model.to("cpu")
        self.dataset = load_inference_source(self.source)
        self.line_thickness = opt.line_thickness
        self.event = opt.event  # bool类型

    def predict(self):
        # Check source path
        if self.mode == 0:  # 离线视频检查路径是否存在
            if not Path(self.source).exists():
                raise FileNotFoundError(f"Source path '{self.source}' does not exist.")

        # 支持单个视频流推理
        if self.mode == 1:
            pass

        # 通过cap属性确认数据集是否是视频还是图像
        if self.dataset.cap is not None:
            if self.event:
                # 触发事件
                self.detect_videos_with_events()
            else:
                # 不触发事件
                self.detect_videos()
        else:
            pass
        # 不触发事件

    def main(self):
        self.predict()

    def detect_videos_with_events(self):
        counting_regions = self.get_regions(REGION_CONFIG)
        device_channel = self.get_device(DEVICE_CONFIG)
        send_ids = []
        # Extract classes names
        names = self.model.model.names
        vid_frame_count = 0
        results_videos = {}
        # video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))
        video_writer_map = {}
        for paths, imgs, info in self.dataset:
            source = paths[0]
            if source not in video_writer_map.keys():
                videocapture = cv2.VideoCapture(source)
                frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
                fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
                video_name = str(WORK_DIR + IMG_TEMP_DIR + f"{Path(source).stem}.mp4")
                results_videos[video_name] = str(WORK_DIR + RESULT_SAVE_DIR + f"{Path(source).stem}.mp4")
                video_writer = cv2.VideoWriter(video_name, fourcc, fps,
                                               (frame_width, frame_height))
                video_writer_map[source] = video_writer

            video_writer = video_writer_map[source]
            # results save
            vid_frame_count += 1
            data = []
            files = []
            frame = imgs[0]
            frame_copy = frame.copy()
            # Extract the results
            results = self.model.track(frame, persist=True, classes=self.classes)
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()

                annotator = Annotator(frame, line_width=self.line_thickness, example=str(names))

                for box, track_id, cls in zip(boxes, track_ids, clss):
                    is_save = False
                    annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                    bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                    track = track_history[track_id]  # Tracking Lines plot
                    track.append((float(bbox_center[0]), float(bbox_center[1])))
                    if len(track) > 30:
                        track.pop(0)
                    # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    # cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                    # Check if detection inside region
                    if counting_regions is not None:
                        for region in counting_regions:
                            if track_id not in send_ids:
                                if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                                    is_save = True
                    else:
                        if track_id not in send_ids:
                            is_save = True
                    if is_save:
                        send_ids.append(track_id)
                        data.append({
                            "objId": int(cls),
                            "location": box.tolist(),
                            "deviceId": device_channel["deviceId"],
                            "channelId": device_channel["channelId"],
                            "taskId": device_channel["taskId"]
                        })

            if counting_regions is not None:
                # Draw regions (Polygons/Rectangles)
                for region in counting_regions:
                    region_label = str(region["counts"])
                    region_color = region["region_color"]
                    region_text_color = region["text_color"]

                    polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
                    centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

                    text_size, _ = cv2.getTextSize(
                        region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=self.line_thickness
                    )
                    text_x = centroid_x - text_size[0] // 2
                    text_y = centroid_y + text_size[1] // 2
                    cv2.rectangle(
                        frame,
                        (text_x - 5, text_y - text_size[1] - 5),
                        (text_x + text_size[0] + 5, text_y + 5),
                        region_color,
                        -1,
                    )
                    cv2.putText(
                        frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color,
                        self.line_thickness
                    )
                    cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color,
                                  thickness=self.line_thickness)

            video_writer.write(frame)

            if len(data):
                img_name = WORK_DIR + RESULT_SAVE_DIR + f"event-{vid_frame_count}.jpg"
                files.append(img_name)
                cv2.imwrite(img_name, frame)
                img_copy = WORK_DIR + RESULT_SAVE_DIR + f"snapshot-{vid_frame_count}.jpg"
                files.append(img_copy)
                cv2.imwrite(img_copy, frame_copy)
                self.send_results(data, files)

            # for region in counting_regions:  # Reinitialize count for each region
            #     region["counts"] = 0

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        del vid_frame_count
        for writer in video_writer_map.values():
            writer.release()

        cv2.destroyAllWindows()

    def detect_videos(self):
        counting_regions = self.get_regions(REGION_CONFIG)
        device_channel = self.get_device(DEVICE_CONFIG)
        send_ids = []
        # Extract classes names
        names = self.model.model.names
        vid_frame_count = 0
        results_videos = {}
        # video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))
        video_writer_map = {}
        for paths, imgs, info in self.dataset:
            source = paths[0]
            if source not in video_writer_map.keys():
                videocapture = cv2.VideoCapture(source)
                frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
                fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
                video_name = str(WORK_DIR + IMG_TEMP_DIR + f"{Path(source).stem}.mp4")
                results_videos[video_name] = str(WORK_DIR + RESULT_SAVE_DIR + f"{Path(source).stem}.mp4")
                video_writer = cv2.VideoWriter(video_name, fourcc, fps,
                                               (frame_width, frame_height))
                video_writer_map[source] = video_writer

            video_writer = video_writer_map[source]
            # results save
            vid_frame_count += 1
            detect_info = []
            data = []
            files = []
            frame = imgs[0]
            frame_copy = frame.copy()
            # Extract the results
            results = self.model.track(frame, persist=True, classes=self.classes)
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                confs = results[0].probs

                annotator = Annotator(frame, line_width=self.line_thickness, example=str(names))

                for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
                    is_save = False
                    annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                    bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                    track = track_history[track_id]  # Tracking Lines plot
                    track.append((float(bbox_center[0]), float(bbox_center[1])))
                    if len(track) > 30:
                        track.pop(0)
                    # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    # cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                    # Check if detection inside region
                    if counting_regions is not None:
                        for region in counting_regions:
                            if track_id not in send_ids:
                                if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                                    is_save = True
                    else:
                        if track_id not in send_ids:
                            is_save = True
                    if is_save:
                        send_ids.append(track_id)
                        detect_info.append({
                            "objId": int(cls),
                            "location": box.tolist(),
                            "conf": conf
                        })

            video_writer.write(frame)
            if len(detect_info):
                img_name = WORK_DIR + RESULT_SAVE_DIR + f"event-{vid_frame_count}.jpg"
                img_copy = WORK_DIR + RESULT_SAVE_DIR + f"snapshot-{vid_frame_count}.jpg"
                cv2.imwrite(img_copy, frame_copy)
                cv2.imwrite(img_name, frame)
                data.append({
                    "taskId": device_channel["taskId"],
                    "detectInfo": detect_info,
                    "label": img_name,
                    "raw": img_copy
                })
                self.send_results(data)

            # for region in counting_regions:  # Reinitialize count for each region
            #     region["counts"] = 0

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        del vid_frame_count
        for writer in video_writer_map.values():
            writer.release()

        cv2.destroyAllWindows()

    # 获取区域检测配置区域
    def get_regions(self, path):
        if path is None or not Path(path).exists():
            return None
        with open(path, 'r') as file:
            yaml_str = file.read()
            file.close()
            data = yaml.load(yaml_str, Loader=yaml.Loader)

        regions = []
        for key in data:
            region = {}
            area = []
            for s in data[key]:
                area.append((int(s[0]), int(s[1])))
            region["name"] = key
            region["counts"] = 0
            region["region_color"] = (37, 255, 225)
            region["text_color"] = (255, 255, 255)
            region["polygon"] = Polygon(area)
            regions.append(region)

        return regions

    def get_device(self, path):
        with open(path, 'r') as file:
            yaml_str = file.read()
            file.close()
            data = yaml.load(yaml_str, Loader=yaml.Loader)

        return data

    def send_results(self, data):
        url = "http://192.168.100.113:8060/ai/result"
        json_data = {"data": json.dumps(data)}
        # datas = json.dumps(json_data)
        print(json_data)
        response = requests.post(url, data=json_data)
        if response.status_code != 200:
            print(f'error : {response.status_code}')
        else:
            print('预警信息发送成功！')


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--mode", type=int, default=0, help="predict mode")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")
    parser.add_argument("--event", action="store_true", help="trigger events")

    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    predictor = CommonPredictor(opt)
    predictor.main()
