from detectron import *
import cv2
"""
read any bgr image with cv2 otherwise convert rgb-to-bgr
"""
imgPath = 'images/city.jpg'
# imgPath = 'scene_data/test/malignant_256.jpg'
img = cv2.imread(imgPath)


detector = Detector(model_type='IS')
# detector.seg_img(img)
# detector.seg_video(video='vid.mp4')
# detector.object_detection(img)
# detector.video_detection(path='vid.mp4')
# detector.key_point_detection(img)
# detector.key_point_video(path='vid.mp4')

