from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from django.http import JsonResponse, HttpResponse
import time
from .tasks import process_image
import cv2
import torch
import os
import uuid
import sys
import numpy as np
from dotenv import load_dotenv

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
from pymilvus import Milvus, CollectionSchema, FieldSchema, DataType, Collection, connections, utility
import base64
from .models import Metadata
from .serializers import MetadataSerializer
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
        print(f"No faces detected in the image: {image_path}")

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


def import_embeddings(image_directory, collection_name):
    milvus = Milvus(host=MILVUS_HOST, port=MILVUS_PORT)
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    client = Milvus(MILVUS_HOST, MILVUS_PORT)

    # Check if collection exists
    if collection_name in milvus.list_collections():
        collection = Collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    else:
        # Create collection with embedding and id fields
        vector_id = FieldSchema(
            name="vector_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=36,
            auto_id=False
        )
        vector = FieldSchema(
            name="face_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=2048
        )
        schema = CollectionSchema(
            fields=[vector_id, vector],
            description="Collection of face embeddings",
            enable_dynamic_field=True
        )
        collection = Collection(
            name=collection_name,
            schema=schema,
            using='default'
        )

    start = 1
    # Import embeddings into Milvus

    for index, file_name in enumerate(os.listdir(image_directory), start=start):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            image_path = os.path.join(image_directory, file_name)
            process_image.delay(image_path, collection_name)

    client.flush([collection_name])
    stats = client.get_collection_stats(collection_name)
    print(stats)

    # Create index
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    if collection:
        collection.create_index(
            field_name="face_vector",
            index_params=index_params
        )

        utility.index_building_progress(collection_name)


class MetadataViewSet(viewsets.ModelViewSet):
    queryset = Metadata.objects.all()
    serializer_class = MetadataSerializer
    permission_classes = (IsAuthenticated,)

    @action(detail=False, methods=['get'])
    def commit(self, request, *args, **kwargs):
        start_time = time.time()
        image_directory = "C:/Users/User4/PycharmProjects/eTanuReincarnationAPI/metadata/data/test"

        collection_name = 'face_embeddings'

        import_embeddings(image_directory, collection_name)

        end_time = time.time()
        wasted_time = end_time - start_time

        print("Request wasted time:", wasted_time)

        return JsonResponse({'status': 'Success'})
