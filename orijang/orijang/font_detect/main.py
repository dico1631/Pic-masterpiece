import base64
import tensorflow as tf
import IPython.display as display
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import PIL.Image
import time
import functools
import base64
import PIL.Image as Image
import io
import os
from io import BytesIO
from .train_mirchine import train_mirchine, StyleContentModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 텐서를 이미지로 바꾸는 함수
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

# 전달받은 base64를 load
def load_base64(img):
  max_dim = 512
  img = tf.io.decode_base64(img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# loss를 계산하는 함수
def style_content_loss(outputs, TM):
    style_weight=1e-2
    content_weight=1e4
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-TM.style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / TM.num_style_layers
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-TM.content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / TM.num_content_layers
    loss = style_loss + content_loss
    return loss

# 학습 함수
def train_step(image, TM):
  with tf.GradientTape() as tape:
    outputs = TM.extractor(image)
    loss = style_content_loss(outputs, TM)
    loss += TM.total_variation_weight*tf.image.total_variation(image)
  grad = tape.gradient(loss, image)
  TM.opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

# 학습 실행 함수
def detect(content_image, style_img):
  file_path = "C:/Users/USER/.keras/datasets"
  file_name = "style.jpg"
  style_path = tf.keras.utils.get_file(file_name, style_img)
  imagedata = content_image
  content_image = load_base64(imagedata)
  TM = train_mirchine()
  TM.Setup(style_path, content_image)
  image = tf.Variable(content_image)
  
  epochs = 1
  steps_per_epoch = 200
  step = 0
  for n in range(epochs):
    for m in range(steps_per_epoch):
      step += 1
      train_step(image, TM)
  buffered = BytesIO()
  convert_image = tensor_to_image(image)
  convert_image.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue())
  img_str = str(img_str)[2:-1]
  os.remove(os.path.join(file_path, file_name))
  return img_str