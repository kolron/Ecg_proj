from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import Classify
from PIL import Image
import io
from flask_cors import CORS


app = Flask(__name__)
api = Api(app)
CORS(app)

@app.before_first_request
def before_request_load():
    print("loading models...")
    Classify.load_models()

class Status(Resource):
    def get(self):
        try:
            return {'data': 'Api running'}
        except error:
            return {'data': error}


class Prediction(Resource):
    def post(self):
        if request.files.get("image"):
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            result = Classify.predict(image)
            return jsonify(result)
        else:
            return {'data':"bad res"}


api.add_resource(Status, '/')
api.add_resource(Prediction,'/predict')

if __name__ == '__main__':

    app.run(debug=False)

