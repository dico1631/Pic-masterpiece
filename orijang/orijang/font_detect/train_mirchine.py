import base64
import tensorflow as tf
import numpy as np
import PIL.Image
import functools
import base64
import PIL.Image as Image
import io
import os
from io import BytesIO

# 학습에 필요한 데이터를 가지고 있는 함수
class train_mirchine():
    def __init__(self, style_image = None, content_image = None, style_extractor = None, 
    style_outputs = None, extractor = None, results = None, style_targets = None, content_targets = None):
        self.content_layers = ['block5_conv2'] 
        self.style_layers = ['block1_conv1',
                            'block2_conv1',
                            'block3_conv1', 
                            'block4_conv1', 
                            'block5_conv1'] 
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.style_image = style_image
        self.content_image = content_image
        self.style_extractor = style_extractor
        self.style_outputs = style_outputs
        self.extractor = extractor
        self.results = results
        self.style_targets = style_targets
        self.content_targets = content_targets
        self.total_variation_weight=30
        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    
    # 이미지를 학습에 사용할 수 있게 형변환하는 함수
    def load_img(self, path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img
    
    # 모델 구성에 필요한 레이어 생성
    def vgg_layers(self, layer_name):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_name]
        model = tf.keras.Model([vgg.input], outputs)
        return model
    
    # 클래스 초기화
    def Setup(self, path_to_img, content_image):
        self.style_image = self.load_img(path_to_img)
        self.content_image = content_image
        self.style_extractor = self.vgg_layers(self.style_layers)
        self.style_outputs = self.style_extractor(self.style_image*255)
        self.extractor = StyleContentModel(self.style_layers, self.content_layers, self)
        self.results = self.extractor(tf.constant(self.content_image))
        self.style_targets = self.extractor(self.style_image)['style']
        self.content_targets = self.extractor(self.content_image)['content']
        self.total_variation_weight=30
        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1) 
        
# 학습 모델 클래스
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers, train_mirchine):
        super(StyleContentModel, self).__init__()
        self.TM = train_mirchine
        self.vgg = self.TM.vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)
    def call(self, inputs):
        # "Expects float input in [0,1]"
        print(type(inputs))
        print(inputs.shape)
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                        outputs[self.num_style_layers:])
        style_outputs = [self.gram_matrix(style_output)
                        for style_output in style_outputs]
        content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}
        style_dict = {style_name:value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}
        return {'content':content_dict, 'style':style_dict}