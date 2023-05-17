from rest_framework import generics
import pandas
import numpy
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import requests
import torch
from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import serializers
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework import serializers
import base64
import geocoder


# Create your views here.


def test(image_data):

    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='best.pt')

    img = Image.open(BytesIO(image_data.read()))

    results = model(img)

    resultslist = (results.pandas().xyxy[0]).to_dict(orient="list")

    res = set(resultslist['name'])
    print(res)

    xminl, yminl, xmaxl, ymaxl, classl, labell = resultslist['xmin'], resultslist[
        'ymin'], resultslist['xmax'], resultslist['ymax'], resultslist['class'], resultslist['name']

    color = {2: 'red', 3: 'blue', 4: 'green', 6: "yellow",
             8: 'orange', 10: 'purple', 11: 'violet'}

    for i in range(0, len(xminl)):
        xmin, xmax, ymin, ymax = xminl[i], xmaxl[i], yminl[i], ymaxl[i]
        draw = ImageDraw.Draw(img)

        # Draw bounding box
        draw.rectangle([(xmin, ymin), (xmax, ymax)],
                       outline=color[classl[i]], width=15)

        # Calculate text and background size
        label = labell[i]
        font_size = int((ymax - ymin) / 12)
        font = ImageFont.truetype("arial.ttf", font_size)
        text_width, text_height = draw.textsize(label, font)
        bg_width = text_width + font_size
        bg_height = text_height + font_size

        # Draw label background and text
        bg_left = xmin
        bg_top = ymin - bg_height
        bg_right = xmin + bg_width
        bg_bottom = ymin
        text_left = bg_left + font_size / 5
        text_top = bg_top + font_size / 5
        draw.rectangle([(bg_left, bg_top), (bg_right, bg_bottom)],
                       fill=color[classl[i]])
        draw.text((text_left, text_top), label, fill='white', font=font)

    # img_byte_arr = BytesIO()
    img.save('image.png', format='PNG')
    # img_byte_arr = img_byte_arr.getvalue()

    # Set up the API endpoint and parameters

    # Send a POST request to the API endpoint

    # Print the URL of the uploaded image
    # print(response.json()['data']['url'])
    return res


@api_view(['GET'])
def getData(request):
    data = {'name': 'Siddhesh', 'age': 19}
    return Response(data)


class ImageSerializer(serializers.Serializer):
    image = serializers.ImageField(required=False)


@api_view(['POST'])
def putData(request):
    data = request.data
    serializer = ImageSerializer(data=data)

    if serializer.is_valid():
        # Save image
        oimage_data = serializer.validated_data
        image_data = oimage_data.pop('image')
        img = image_data.file.read()

        filename = default_storage.save(
            image_data.name, ContentFile(image_data.read()))
        # image_url = default_storage.url(filename)

        # image_binary = BytesIO(serializer)

        # response_data = {'image_url': image_url}
        # params = {"key": "9fb3c2f5dd4b3ce023abd18e2ccfa5e4",
        #           "image": image_data.file.read()}
        # response = requests.post("https://api.imgbb.com/1/upload",
        #                          params=params)

        return Response("response")
    # return Response("ok")


# def send_image_to_api(image_data):
#     # Replace with the URL of the target API endpoint
#     url = 'https://api.imgbb.com/1/upload?key=9fb3c2f5dd4b3ce023abd18e2ccfa5e4'
#     # Replace with the appropriate content type for your image format
#     headers = {'Content-Type': 'image/jpeg'}
#     params = {"key": "9fb3c2f5dd4b3ce023abd18e2ccfa5e4",
#               "image": image_data}
#     data = {"image": image_data}
#     response = requests.post(url, headers=headers,
#                              data=data)

#     print(response.json())

#     if response.ok:
#         # Process the response data here
#         return response.json()
#     else:
#         # Handle errors here
#         print("a0")
#         return None


class ImageView(generics.CreateAPIView):
    serializer_class = ImageSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        image_data = serializer.validated_data.get('image')
        res = test(image_data)
        detect = "True"
        if(res):
            detect = "True"
        else:
            detect = "False"
        # Process the image data here
        g = geocoder.ip('me')
        latitude, longitude = g.latlng
        return Response({'message': 'Image received', 'detection': detect, "latitude": latitude, "longitude": longitude,  "res": res})
