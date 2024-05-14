import argparse

import torch.cuda
import sys

sys.path.append("/usr/src/ultralytics")
sys.path.append("/usr/src/ultralytics/ultralytics")
sys.path.append("/usr/src/ultralytics/ultralytics/data")

from ultralytics.data.build import load_inference_source, check_source
import json
from collections import defaultdict
from pathlib import Path
from get_config import get_device, get_regions

import cv2
import numpy as np
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)
import os
import requests
import logging

LOGGER = logging.getLogger("xhdt-yolo-detect")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_region = None

work_dir = r"/usr/src/ultralytics/"
# work_dir = r"D:/PycharmProjects/ultralytics/"
img_save_dir = "main/results/"
img_tmp_dir = "main/temp/"
region_config = work_dir + "main/config/regions.yaml"
device_config = work_dir + "main/config/device.yaml"


def detect_videos(dataset, model, classes, line_thickness, track_thickness, region_thickness):
    counting_regions = get_regions(region_config)
    # counting_regions = get_regions(None)
    device_channel = get_device(device_config)
    send_ids = []
    # Extract classes names
    names = model.model.names
    vid_frame_count = 0
    results_videos = {}
    # video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))
    video_writer_map = {}
    for paths, imgs, info in dataset:
        source = paths[0]

        if source not in video_writer_map.keys():
            videocapture = cv2.VideoCapture(source)
            frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
            fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")
            video_name = str(work_dir + img_tmp_dir + f"{Path(source).stem}.mp4")
            results_videos[video_name] = str(work_dir + img_save_dir + f"{Path(source).stem}.mp4")
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
        print(info)
        # Extract the results
        results = model.track(frame, persist=True, classes=classes)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                is_save = False
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                track = track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

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
                        "id": track_id,
                        "objId": int(cls),
                        "type": str(names[cls]),
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
                    region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
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
                    line_thickness
                )
                cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

        video_writer.write(frame)

        if len(data):
            img_name = work_dir + img_save_dir + f"event-{vid_frame_count}.jpg"
            files.append(img_name)
            cv2.imwrite(img_name, frame)
            img_copy = work_dir + img_save_dir + f"snapshot-{vid_frame_count}.jpg"
            files.append(img_copy)
            cv2.imwrite(img_copy, frame_copy)
            send_results(data, files)

        # for region in counting_regions:  # Reinitialize count for each region
        #     region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    for writer in video_writer_map.values():
        writer.release()

    # for k in results_videos.keys():
    #     v = results_videos[k]
    #     cmd = f"ffmpeg -i {k} -c:v libx264 {v}"
    #     print(cmd)
    #     os.system(cmd)
    # video_writer.release()
    cv2.destroyAllWindows()


def detect_images(dataset, model, classes, line_thickness):
    # counting_regions = get_regions(region_config)
    # counting_regions = get_regions(None)
    device_channel = get_device(device_config)
    names = model.model.names
    id = 0
    for paths, imgs, info in dataset:
        frame = imgs[0]
        frame_copy = imgs[0].copy()
        results = model.predict(frame, classes=classes)
        data = []
        files = []
        for r in results:
            # r.show()
            img_name = work_dir + img_save_dir + Path(paths[0]).name
            r.save(filename=img_name)
            files.append(img_name)
            snapshot = work_dir + img_save_dir + f"snapshot-{id}.jpg"
            cv2.imwrite(snapshot, frame_copy)
            files.append(snapshot)
            boxes = r.boxes
            clss = r.boxes.cls.cpu().tolist()
            for box, cls in zip(boxes, clss):
                data.append({
                    "id": id,
                    "objId": int(cls),
                    "type": str(names[cls]),
                    "location": box.xyxy.cpu().tolist()[0],
                    "deviceId": device_channel["deviceId"],
                    "channelId": device_channel["channelId"],
                    "taskId": device_channel["taskId"]
                })

        if len(data):
            send_results(data, files)

        id += 1
        print(data)


def run(source, weights,
        view_img=False,
        save_img=False,
        classes=2,
        line_thickness=1,
        track_thickness=1,
        region_thickness=1, ):
    dataset = load_inference_source(source)

    # Setup Model
    model = YOLO(f"{weights}")
    model.to("cuda") if torch.cuda.is_available() else model.to("cpu")

    if dataset.cap is not None:
        # 视频数据
        detect_videos(dataset, model, classes, line_thickness, track_thickness, region_thickness)
    else:
        detect_images(dataset, model, classes, line_thickness)


def send_results(data, files):
    img_path, snapshot = files[0], files[1]
    url = "http://192.168.100.113:8060/ai/python/result/test"
    file = {'labeled': (img_path, open(img_path, 'rb')), 'raw': (snapshot, open(snapshot, 'rb'))}
    json_data = {"data": json.dumps(data)}
    # datas = json.dumps(json_data)
    print(json_data)
    response = requests.post(url, data=json_data, files=file)
    if response.status_code != 200:
        LOGGER.info(f'error : {response.status_code}')
    else:
        LOGGER.info('预警信息发送成功！')


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    parser.add_argument("--track-thickness", type=int, default=2, help="Tracking line thickness")
    parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")

    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == '__main__':
    # run(r"D:\PycharmProjects\ultralytics\resources\exp",
    #     weights=r"D:\PycharmProjects\ultralytics\weights\safety_hat\best.pt")
    opt = parse_opt()
    main(opt)
