import base64
from google.cloud import storage
from google.cloud import aiplatform
from google.protobuf import struct_pb2
import sys
import time
import typing
from google.auth import default

# 参考 : https://colab.research.google.com/github/tankbattle/hello-world/blob/master/Build_Cloud_CoCa_Image_Embedding_Dataset_%26_Search.ipynb#scrollTo=x2BrVlM-phGN

credentials, project_id = default()
client = aiplatform.gapic.PredictionServiceClient(credentials=credentials)


PROJECT_ID = 'suzu-develop-stg' # @param {type: "string"}

# Inspired from https://stackoverflow.com/questions/34269772/type-hints-in-namedtuple.
class EmbeddingResponse(typing.NamedTuple):
  text_embedding: typing.Sequence[float]
  image_embedding: typing.Sequence[float]

class EmbeddingPredictionClient:
  """Wrapper around Prediction Service Client."""
  def __init__(self, project : str,
    location : str = "asia-northeast1",
    api_regional_endpoint: str = "asia-northeast1-aiplatform.googleapis.com"):
    client_options = {"api_endpoint": api_regional_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    self.client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    self.location = location
    self.project = project

  def get_embedding(self, text : str = None, image_bytes : bytes = None):
    if not text and not image_bytes:
      raise ValueError('At least one of text or image_bytes must be specified.')

    instance = struct_pb2.Struct()
    if text:
      instance.fields['text'].string_value = text

    if image_bytes:
      encoded_content = base64.b64encode(image_bytes).decode("utf-8")
      image_struct = instance.fields['image'].struct_value
      image_struct.fields['bytesBase64Encoded'].string_value = encoded_content

    instances = [instance]
    endpoint = (f"projects/{self.project}/locations/{self.location}"
      "/publishers/google/models/multimodalembedding@001")
    response = self.client.predict(endpoint=endpoint, instances=instances)

    text_embedding = None
    if text:
      text_emb_value = response.predictions[0]['textEmbedding']
      text_embedding = [v for v in text_emb_value]

    image_embedding = None
    if image_bytes:
      image_emb_value = response.predictions[0]['imageEmbedding']
      image_embedding = [v for v in image_emb_value]

    return EmbeddingResponse(
      text_embedding=text_embedding,
      image_embedding=image_embedding)

client = EmbeddingPredictionClient(project=PROJECT_ID)

# Extract image embedding
def getImageEmbeddingFromImageContent(content):
  response = client.get_embedding(text=None, image_bytes=content)
  return response.image_embedding

def getImageEmbeddingFromGcsObject(gcsBucket, gcsObject):
  client = storage.Client()
  bucket = client.bucket(gcsBucket)
  blob = bucket.blob(gcsObject)

  with blob.open("rb") as f:
    return getImageEmbeddingFromImageContent(f.read())

def getImageEmbeddingFromFile(filePath):
  with open(filePath, "rb") as f:
    return getImageEmbeddingFromImageContent(f.read())

def getImageEmbeddingFromBytes(bytes_data):
    return getImageEmbeddingFromImageContent(bytes_data)
  
# Extract text embedding
def getTextEmbedding(text):
  response = client.get_embedding(text=text, image_bytes=None)
  return response.text_embedding



# cloud storage の image を embedding
# IMAGE_SET_BUCKET_NAME = "suzu-ec-coordinate-images"

# gcsBucket = storage.Client()
# blob = gcsBucket.bucket(IMAGE_SET_BUCKET_NAME).blob("072b27bce603b7199048af823e5ce368_l.jpg")


# print(blob.name)

# embedding = getImageEmbeddingFromGcsObject(IMAGE_SET_BUCKET_NAME, blob.name)

# print(embedding)


# ローカル の image を embedding
# filePath = "./imgs/146f4817ad415b7e28171ff33688c54a_l.jpg"

# embedding = getImageEmbeddingFromFile(filePath)

# print(embedding)