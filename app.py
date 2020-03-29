from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, send_file, url_for
import os
from PIL import Image, ImageOps
import numpy as np
import math

app = Flask(__name__)
app.static_folder = 'static'
dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_PATH = os.path.join(os.getcwd(), 'keras_model.h5')

sess = tf.Session()

graph = tf.compat.v1.get_default_graph()
set_session(sess)
model = load_model(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        np.set_printoptions(suppress=True)
        
        if request.method == 'POST':
            print("POST to /")
            file = request.files['query_img']

            img = Image.open(file.stream)  # PIL image
            # uploaded_img_path = "static\\uploads\\" + file.filename
            uploaded_img_path = os.path.join('static', 'uploads', file.filename)
            # uploaded_img_path = os.path.join(os.getcwd(), 'uploads', file.filename)
            img.save(uploaded_img_path)
            
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            # Replace this with the path to your image
            # uploaded_img_path = os.path.join(os.getcwd(), 'uploads', 'IM-0033-0001-0001.jpg')

            print("Image path {}".format(uploaded_img_path))
            image = Image.open(uploaded_img_path)

            #resize the image to a 224x224 with the same strategy as in TM2:
            #resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            image = image.convert('RGB')
            # image.show()
            #turn the image into a numpy array
            image_array = np.asarray(image)

            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # run the inference
            prediction = model.predict(data)

            # print("Prediction....")
            # print(prediction)

            print("Positive : {}".format(float(prediction[0][0])))
            # positive_score = float(prediction[0][0])
            print("Negative : {}".format(float(prediction[0][1])))
            # negative_score = float(prediction[0][1])

            predicted_scores = [{
                "positive": round(float(math.floor(prediction[0][0]*100)), 2),
                "negative": round(float(math.floor(prediction[0][1]*100)), 2)
            }]
            return render_template('index.html', query_path=uploaded_img_path, predicted_scores=predicted_scores)
            
        else:
            return render_template('index.html')

@app.route('/questions', methods=['GET'])
def questions():
    return render_template('infermedica.html')

@app.route('/detect', methods=['POST'])
def post_example():
    print(request)
    global sess
    global graph
    with graph.as_default():
        
        # perform the prediction
        set_session(sess)
        np.set_printoptions(suppress=True)

        if not request.headers.get('Content-type') is None:
            if(request.headers.get('Content-type').split(';')[0] == 'multipart/form-data'):
                if 'image' in request.files.keys():
                    print("inside get image statement")
                    file = request.files['image']
                    img = Image.open(file.stream)  # PIL image
                    uploaded_img_path = os.path.join(os.getcwd(), 'static', 'uploads', file.filename)
                    print("Upload Path : {}".format(uploaded_img_path))

                    # uploaded_img_path = "uploads\\" +  str(int(round(time.time() * 1000))) + "_" + file.filename
                    img.save(uploaded_img_path)

                    # prediction = classify_covid19.classify(uploaded_img_path)

                    # print("Complete....")
                    # print("Classifying....")
                    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                    # Replace this with the path to your image
                    # uploaded_img_path = os.path.join(os.getcwd(), 'uploads', 'IM-0033-0001-0001.jpg')

                    print("Image path {}".format(uploaded_img_path))
                    image = Image.open(uploaded_img_path)

                    #resize the image to a 224x224 with the same strategy as in TM2:
                    #resizing the image to be at least 224x224 and then cropping from the center
                    size = (224, 224)
                    image = ImageOps.fit(image, size, Image.ANTIALIAS)
                    image = image.convert('RGB')
                    # image.show()
                    #turn the image into a numpy array
                    image_array = np.asarray(image)

                    # Normalize the image
                    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

                    # Load the image into the array
                    data[0] = normalized_image_array

                    # run the inference
                    prediction = model.predict(data)

                    # print("Prediction....")
                    # print(prediction)

                    print("Positive : {}".format(float(prediction[0][0])))
                    print("Negative : {}".format(float(prediction[0][1])))

                    result = {
                        "positive": float(prediction[0][0]),
                        "negative": float(prediction[0][1])
                    }

                    return jsonify(result), 200
                   
                
                else:
                    return jsonify(get_status_code("Invalid body", "Please provide valid format for Image 2")), 415

            elif(request.headers.get('Content-type') == 'application/json'):
                if(request.data == b''):
                    return jsonify(get_status_code("Invalid body", "Please provide valid format for Image")), 415
                else:
                    body = request.get_json()
                    if "image_string" in body.keys():
                        str_image = body['image_string']
                        # str_image = img_string.split(',')[1]
                        imgdata = base64.b64decode(str_image)
                        img = "uploads\\" +  str(int(round(time.time() * 1000))) + "image_file.jpg"
                        with open(img, 'wb') as f:
                            f.write(imgdata)

                        # image=Image.open(img)
                        result = classify_covid19.classify(uploaded_img_path)
                        #print (img)
                        payload = {
                            "results":{
                                "positive": result[0][0],
                                "negative": result[0][1]
                            }
                        }
                        return jsonify(payload)

            else:
                return jsonify(get_status_code("Invalid header", "Please provide correct header with correct data")), 415

        else:
            return jsonify(get_status_code("Invalid Header", "Please provide valid header")), 401


def get_status_code(argument, message):
    res = {
        "error": {
            "code": argument,
            "message": message
        }
    }
    return res

if __name__=="__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
