'''import tensorflow as tf
import keras

print(tf.__version__)
print(keras.__version__)
'''
from yolo import YOLO


images_path = 'a.tif'

face_det_obj = YOLO()

face_det_obj.detect_image(img_path=images_path, save_path='images/' + images_path[0] + '.tif')