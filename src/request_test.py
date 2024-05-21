import json

import requests

url = "http://192.168.100.113:8060/ai/result"

data = {
    "taskId": 1,
    "detectInfo": [
        {
            "objId": 0,
            "location": {
                "top": 100,
                "left": 100,
                "width": 100,
                "height": 100
            },
            "conf": 0.5
        },
        {
            "objId": 1,
            "location": {
                "top": 20,
                "left": 200,
                "width": 200,
                "height": 200
            },
            "conf": 0.6
        }
    ],
    "label": "/ai/datasets/test.jpg",
    "raw": "/ai/datasets/test1.jpg",
    "dataset": "test"
}
json_data = {"data": json.dumps(data)}
print(json_data)
response = requests.post(url, data=json_data)
print(response.text)
