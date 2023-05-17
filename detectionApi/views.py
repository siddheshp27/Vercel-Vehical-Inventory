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
import face_recognition
from scipy.spatial import distance as dist
import playsound
import numpy as np
from threading import Thread
import time
import collections
import overpy
import requests
from ultralytics import YOLO
import cv2
from PIL import Image


# Create your views here.

# WINDOW_SIZE = 10  # Number of frames to consider for analysis
# THRESHOLD = 0.7  # Minimum ratio of drowsy frames within the window to consider as drowsiness

# # Create a deque to store the status of previous frames
# status_buffer = collections.deque(maxlen=WINDOW_SIZE)


def test(image_data):

    # model = torch.hub.load('ultralytics/yolov5', 'custom',
    #                        path='bestnew.pt', force_reload=True)

    model = YOLO('bestnew.pt')

    img = Image.open(BytesIO(image_data.read()))

    results = model(img)
    names = model.names
    class_set = set([])
    boxes = results[0].boxes
    for r in results:
        for c in r.boxes.cls:
            class_set.add(names[int(c)])
    print(class_set)
    res_plotted = results[0].plot()
    image = Image.fromarray(res_plotted.astype(np.uint8))

    present_vehicles = []

    if 'car' in class_set:
        present_vehicles.append('car')

    if 'bicycle' in class_set:
        present_vehicles.append('bicycle')

    if 'bus' in class_set:
        present_vehicles.append('bus')

    if 'motorbike' in class_set:
        present_vehicles.append('motorbike')

# Save the image
    image.save("output_image.png")
# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     print(boxes)
#     masks = result.masks  # Masks object for segmentation masks outputs
#     print(masks)
#     probs = result.probs  #
#     print(probs)

# model.predict(img, save=True, imgsz=320, conf=0.5)

# resultslist = (results.pandas().xyxy[0]).to_dict(orient="list")

# res = set(resultslist['name'])
# print(res)

# xminl, yminl, xmaxl, ymaxl, classl, labell = resultslist['xmin'], resultslist[
#     'ymin'], resultslist['xmax'], resultslist['ymax'], resultslist['class'], resultslist['name']

# color = {2: 'red', 3: 'blue', 4: 'green', 6: "yellow",
#          8: 'orange', 10: 'purple', 11: 'violet'}

# for i in range(0, len(xminl)):
#     xmin, xmax, ymin, ymax = xminl[i], xmaxl[i], yminl[i], ymaxl[i]
#     draw = ImageDraw.Draw(img)

#     # Draw bounding box
#     draw.rectangle([(xmin, ymin), (xmax, ymax)],
#                    outline=color[classl[i]], width=15)

#     # Calculate text and background size
#     label = labell[i]
#     font_size = int((ymax - ymin) / 12)
#     font = ImageFont.truetype("arial.ttf", font_size)
#     text_width, text_height = draw.textsize(label, font)
#     bg_width = text_width + font_size
#     bg_height = text_height + font_size

#     # Draw label background and text
#     bg_left = xmin
#     bg_top = ymin - bg_height
#     bg_right = xmin + bg_width
#     bg_bottom = ymin
#     text_left = bg_left + font_size / 5
#     text_top = bg_top + font_size / 5
#     draw.rectangle([(bg_left, bg_top), (bg_right, bg_bottom)],
#                    fill=color[classl[i]])
#     draw.text((text_left, text_top), label, fill='white', font=font)

# img.save('image.png', format='PNG')

    return present_vehicles


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

        return Response("response")


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


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear


def detectDrowsiness(image_data):

    img = Image.open(BytesIO(image_data.read()))
    image = img.convert("RGB")

    # Convert PIL image to numpy array
    frame = np.array(image)

    # Minimum eye aspect ratio to consider as drowsiness

    MIN_EAR = 0.3

    # Maping the fetched image with face recognition and extracting facial landmarks
    face_landmarks_list = face_recognition.face_landmarks(frame)

    status = "NULL"
    # from the face landmarks list fetching the required area of eye in form of 6 key coordinates
    for face_landmark in face_landmarks_list:

        leftEye = face_landmark["left_eye"]
        rightEye = face_landmark["right_eye"]

        # to determine the eye aspect ratio of each eye individually
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2

        if ear < MIN_EAR:
            status = "Drowsy"
        else:
            status = "Awake"

    img.save('image.png', format='PNG')
    res = status
    return res


# Create a deque to store the status of previous frames
status_buffer = collections.deque(maxlen=10)


class FaceView(generics.CreateAPIView):
    serializer_class = ImageSerializer

    def post(self, request, *args, **kwargs):
        global status_buffer

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        image_data = serializer.validated_data.get('image')

        res = detectDrowsiness(image_data)

        status_buffer.append(res)

        if len(status_buffer) == 10:
            drowsy_ratio = status_buffer.count("Drowsy") / float(10)
            if drowsy_ratio >= 0.7:
                final_status = "Drowsy"
                return Response({'message': 'Image received',  "res": final_status})
                # Take action or trigger an alert for drowsy driver
            else:
                final_status = "Awake"
                print("Final Status:", final_status)
                return Response({'message': 'Image received',  "res": final_status})
        return Response({'message': 'Image received',  "res": "null"})


@api_view(['POST'])
def hospital(request):
    data = request.data

    print(data.get('test'))

    # Define the latitude and longitude
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    # Query the Overpass API for the nearest hospital
    api = overpy.Overpass()

    query = f"""
    node(around:1000, {latitude}, {longitude})["amenity"="hospital"];
    out;
    """
    result = api.query(query)

    # Extract the nearest hospital from the result
    nearest_hospital = result.nodes[0]

    # Print the details of the nearest hospital
    print("Nearest Hospital:")
    print("Name:", nearest_hospital.tags.get("name", "N/A"))
    print("Latitude:", nearest_hospital.lat)
    print("Longitude:", nearest_hospital.lon)

    # Define the latitude and longitude of the start and end points
    start_latitude = latitude
    start_longitude = longitude
    end_latitude = nearest_hospital.lat
    end_longitude = nearest_hospital.lon

    # Define your GraphHopper API key
    graphhopper_api_key = "2cb62f6f-f6b7-4cdf-9b23-2134fa66d3d5"

    # Construct the GraphHopper API request URL
    api_url = f"https://graphhopper.com/api/1/route?point={start_latitude},{start_longitude}&point={end_latitude},{end_longitude}&vehicle=car&locale=en-US&key={graphhopper_api_key}"

    # Send a GET request to the GraphHopper API
    response = requests.get(api_url)

    # Parse the JSON response
    data = response.json()

    # Check if a route exists
    if "paths" in data and len(data["paths"]) > 0:
        # Extract the road distance from the response
        distance = data["paths"][0]["distance"]
    # Print the road distance
        print("Road Distance (in meters):", distance)
    else:
        print("No route found.")

    google_maps_url = f"https://www.google.com/maps/dir/{start_latitude},{start_longitude}/{end_latitude},{end_longitude}/"

    print("Google Maps Direction Link:")
    print(google_maps_url)

    return Response("response")
