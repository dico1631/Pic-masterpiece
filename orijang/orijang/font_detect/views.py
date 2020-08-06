from django.shortcuts import render
from django.http import HttpResponse
from .forms import ImageForm
from .main import detect
import base64
import tensorflow as tf
import numpy

def crosscolor(request):
    form = ImageForm()
    return render(request, 'font_detect/site.html', {"form" : form})
 
def showchange(request):
    if request.method == "POST":
        # 웹에서 이미지를 받아서
        style_img = request.POST.get('style_img')
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            content_image = form.cleaned_data['image']
            # 바이너리로 읽어서
            with content_image.open("rb") as image_file:
                # base64로 인코드해서
                encoded_string = tf.io.encode_base64(image_file.read(), pad = False, name=None)
                result_img = encoded_string
            # 학습한다
            result = detect(result_img, style_img)
        else:
            raise ValueForm('invalid form')

    return render(request, 'font_detect/showpicture.html', {"result_img" : result})

