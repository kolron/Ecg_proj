# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "https://ecg4u-api.herokuapp.com/predict"
IMAGE_PATH = "D:/ECG/ECG_DATABASE/Images/Not Afib/MI(101).jpg"

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()
print(r)

