from ultralytics import YOLOWorld
import cv2

# model = YOLOWorld("yolov8s-worldv2.pt")
#
# results = model.predict('results/snapshot.jpg')
#
# # Show results
# results[0].show()

videocapture = cv2.VideoCapture(r"D:\PycharmProjects\ultralytics\resources\smoking_test.mp4")
frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"avc1")

# Output setup
video_writer = cv2.VideoWriter("smoking.mp4", fourcc, fps, (frame_width, frame_height))
count = 0
while videocapture.isOpened():
    count += 1
    success, frame = videocapture.read()
    if not success:
        break

    video_writer.write(frame)
    if count > 150:
        video_writer.release()
        videocapture.release()
