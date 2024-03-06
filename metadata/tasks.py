from celery import shared_task
import cv2
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from django.http import JsonResponse, HttpResponse
import time
import torch
import os
import uuid
import sys
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
from pymilvus import Milvus, CollectionSchema, FieldSchema, DataType, Collection, connections, utility
import base64
from .models import Metadata
from .serializers import MetadataSerializer
from dotenv import load_dotenv

load_dotenv()

MILVUS_HOST = os.environ.get('MILVUS_HOST')
MILVUS_PORT = os.environ.get('MILVUS_PORT')


def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
        return base64_encoded


def get_image_metadata(image_path):
    with Image.open(image_path) as img:
        metadata = {
            'format': img.format,
            'mode': img.mode,
            'size': img.size,
            'info': img.info
        }
    return metadata


def get_image_embedding(image_path):
    # Load pre-trained ResNet model
    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # Remove the last fully connected layer
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    # Set model to evaluation mode
    resnet.eval()
    # Load image using OpenCV for face detection
    image = cv2.imread(image_path)
    # Convert image to RGB (OpenCV uses BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load pre-trained face detection model from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Perform face detection
    faces = face_cascade.detectMultiScale(image_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected in the image.")
        return None

    # Get the largest face (assuming only one face per image)
    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])

    # Crop the image to the detected face
    face_image = image_rgb[y:y + h, x:x + w]

    # Define preprocessing transformations
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess face image
    input_tensor = preprocess(face_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        output = resnet(input_batch)

    # Flatten the output
    embedding = output.squeeze().numpy()

    return embedding


@shared_task
def process_image(image_path, collection_name):
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )

    collection = Collection(collection_name)
    embedding = get_image_embedding(image_path)
    base64_string = image_to_base64(image_path)
    metadata = get_image_metadata(image_path)

    # Generate UUID for the embedding
    embedding_id = str(uuid.uuid4())
    full_name = metadata['info']['FIO']

    name_components = full_name.split()
    first_name = None
    surname = None
    patronymic = None
    # Check if all three components are present
    if len(name_components) == 3:
        surname, first_name, patronymic = name_components
    elif len(name_components) == 2:
        surname, first_name = name_components
        patronymic = ""  # If patronymic is missing, assign an empty string
    else:
        print("Invalid full name format")

    Metadata.objects.create(
        vector_id=embedding_id,
        firstName=first_name,
        surname=surname,
        patronymic=patronymic,
        photo=base64_string
    )

    data = [
        [embedding_id],
        [embedding]
    ]

    collection.insert(data)
    print(f"Done: {image_path}")
