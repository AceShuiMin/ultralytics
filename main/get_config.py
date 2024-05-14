from pathlib import Path

import yaml
from shapely.geometry import Polygon


def get_regions(path):
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


def get_device(path):

    with open(path, 'r') as file:
        yaml_str = file.read()
        file.close()
        data = yaml.load(yaml_str, Loader=yaml.Loader)

    return data


if __name__ == '__main__':
    work_dir = r"D:/PycharmProjects/ultralytics/"
    a = get_device(work_dir + "main/config/device.yaml")
    counting_regions = [
        {
            "name": "YOLOv8 Polygon Region",
            "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  # Polygon points
            "counts": 0,
            "dragging": False,
            "region_color": (255, 42, 4),  # BGR Value
            "text_color": (255, 255, 255),  # Region Text Color
        },
        {
            "name": "YOLOv8 Rectangle Region",
            "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),  # Polygon points
            "counts": 0,
            "dragging": False,
            "region_color": (37, 255, 225),  # BGR Value
            "text_color": (0, 0, 0),  # Region Text Color
        },
    ]
    print(a)
