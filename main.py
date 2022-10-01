from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


IMAGE_SIZE = 224
NUM_CLASSES = 10


model = tf.lite.Interpreter("static/model.tflite")
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

display_names = [
    'Chicken Curry', 'Chicken Wing', 'Fried Rice', 'Grilled Salmon',
    'Hamburger', 'Ice Cream', 'Pizza', 'Ramen', 'Steak', 'Sushi']

def model_predict(imgs_arr):
  predictions = [0] * len(imgs_arr)

  for i, val in enumerate(predictions):
    model.set_tensor(input_details[0]['index'], imgs_arr[i].reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.invoke()
    predictions[i] = model.get_tensor(output_details[0]['index']).reshape((NUM_CLASSES,))

  prediction_probabilities = np.array(predictions)
  argmaxs = np.argmax(prediction_probabilities, axis=1)

  return argmaxs


app = FastAPI()
app.mount("/static", StaticFiles(directory='static'), name="static")


def resize(image):
    return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

@app.post("/uploadfiles/", response_class=HTMLResponse)
async def create_upload_files(files: List[UploadFile] = File(...)):
    images = []
    for file in files:
        f = await file.read()
        images.append(f)

    images = [np.frombuffer(img, np.uint8) for img in images]
    images = [cv2.imdecode(img, cv2.IMREAD_COLOR) for img in images]
    images_resized = [resize(img) for img in images]
    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_resized]

    names = [file.filename for file in files]

    for image, name in zip(images_rgb, names):
        pillow_image = Image.fromarray(image)
        pillow_image.save("static/" + name)

    image_paths = ['static/' + name for name in names]

    images_arr = np.array(images_rgb, dtype=np.float32)

    class_indexes = model_predict(images_arr)

    class_predictions = [display_names[x] for x in class_indexes]

    column_labels = ["Image", "Prediction"]

    table_html = get_html_table(image_paths, class_predictions, column_labels)

    content = head_html + """
    <marquee width='525' behaviour='alternate'>
        <h1 style="color:red;font-family:Arial">
            Here is Our Predictions!
        </h1>
    </marquee>
    """ + str(table_html) + '''
    <br>
    <form method="post" action="/">
        <button type="submit">Home</button>
    </form>
    '''

    return content

@app.post("/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def main():
    content = head_html + """
    <marquee width='525' behaviour='alternate'>
        <h1 style="color:red;font-family:Arial">
            Please Upload Your Food Images!
        </h1>
    </marquee>
    <h3 style="font-family:Arial">
        We'll Try to Predict Which of These Categories They Are:
    </h3>
    <br>
    """
    original_paths = [
        'chicken_curry.jpg', 'chicken_wing.jpg', 'fried_rice.jpg',
        'grilled_salmon.jpg', 'hamburger.jpg', 'ice_cream.jpg',
        'pizza.jpg', 'ramen.jpg', 'steak.jpg', 'sushi.jpg']

    full_original_paths = ['static/original/' + x for x in original_paths]

    column_labels = []

    content = content + get_html_table(full_original_paths, display_names, column_labels)

    content = content + """
    <br/>
    <br/>
    <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
    <input name="files", type="file" multiple>
    <input type="submit">
    </form>
    </body>
    """

    return content

head_html = """
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
</head>
<body style="background-color:powderblue;">
<center>
"""

def get_html_table(image_paths, names, column_labels):
    s = '<table align="center">'
    if column_labels:
        s += '<tr><th><h4 style="font-family:Arial">' \
             + column_labels[0] + '</h4></th><th><h4 style="font-family:Arial">'\
             + column_labels[1] + '</h4></th></tr>'
        for name, image_path, in zip(names, image_paths):
            s += '<tr><td><img height="80" src="/' + image_path + '" ></td>'
            s += '<td style="text-align:center">' + name + '</td></tr>'
    else:
        for i in range(5):
            s += '<tr><td><img height="80" src="/' + image_paths[i] + '" ></td>'
            s += '<td style="text-align:center">' + names[i] + '</td>'
            s += '<td><img height="80" src="/' + image_paths[i+5] + '" ></td>'
            s += '<td style="text-align:center">' + names[i+5] + '</td></tr>'

    s += '</table>'

    return s
