import tensorflow
from flask import Flask, jsonify, request
import numpy as np
from data import get_prediction
from keras.models import load_model
from keras.preprocessing import image


app = Flask(__name__)

model = load_model('config/cnn.hdf5')
graph = tensorflow.get_default_graph()


@app.route('/', methods=['GET'])
def main():
	global graph
	with graph.as_default():
		# img = np.array(request.get_json().get('data'))
		new_img = image.load_img('dataset/test/TomatoHealthy1.JPG', target_size=(224, 224))
		img = image.img_to_array(new_img)
		return jsonify({"disease" : get_prediction(model, img)})


if __name__ == '__main__':
    app.run(threaded=True)
