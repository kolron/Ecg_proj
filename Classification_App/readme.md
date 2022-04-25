# Classification App
---
NOTE: This directory is designed purely for the backend api to allow communication with our server ,which is deployed on Herkou.
Meaning: This is NOT the actual (Android) app - that will be in a different repository.

---
To communicate with our server, which hosts all our models, send a POST request with an ECG image as 'files' payload to https://ecg4u-api.herokuapp.com/predict endpoint.

backend api is built using flask, flask_restful, gunicorn.

## How does it work?
1. Sending a post request to the aforementioned endpoint - will be routed by the route specified in [app.py](https://github.com/kolron/Ecg_proj/blob/main/Classification_App/app.py) file
2. This will call [Classify.py](https://github.com/kolron/Ecg_proj/blob/main/Classification_App/Classify.py) to run the image on each one of the models specified in that file.
3. Construct result and send it as a response.

Models are construced using the [Model.py](https://github.com/kolron/Ecg_proj/blob/main/Classification_App/Model.py) file, and have the indentical architecture as the ones used in training.

Checkpoints (Weights,Bias) are loaded from the corresponding file in [here](https://github.com/kolron/Ecg_proj/tree/main/Classification_App/checkpoints)
additionaly, checkpoints are loaded only once on server startup, in order to reduce latency in real-time applications.
 
