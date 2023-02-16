import tensorflow as tf
import dash
import dash_core_components as dcc
import dash_html_components as html
import io
import base64
import numpy as np
from PIL import Image

app = dash.Dash(__name__)

server = app.server
model = tf.keras.models.load_model("model_final.h5")
label_map = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


def predict_image(image):
    img = Image.open(io.BytesIO(image.getvalue()))
    img = img.convert('RGB')
    img = img.resize((48, 48))
    img = np.array(img)
    img = img.reshape((48, 48, 3))
    img = img.astype('float32')
    img /= 255
    prediction = model.predict(img[np.newaxis, ...])
    return prediction[0]
    
    return predicted_class


def parse_image(contents, filename, date):
    contents = str(contents)
    iimage = contents.split(",")[1]
    image = base64.b64decode(iimage)
    image = io.BytesIO(image)
    prediction = predict_image(image)
    image_b64 = base64.b64encode(image.getvalue()).decode('utf-8')
    return prediction, "data:image/png;base64," + image_b64


app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Image')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-image')
])


@app.callback(
    dash.dependencies.Output('output-image', 'children'),
    [dash.dependencies.Input('upload-image', 'contents')],
    [dash.dependencies.State('upload-image', 'filename'),
     dash.dependencies.State('upload-image', 'last_modified')]
)
def update_output(contents, filename, date):
    if contents is None:
        return html.Div([
            html.H3("No image uploaded")
        ])

    prediction, image = parse_image(contents, filename, date)

    prediction_str = label_map[np.argmax(prediction)]
    return html.Div([
        html.H3("Prediction"),
        html.Div(prediction_str),
        html.Img(src=image, style={'width': '50%'})
    ])


if __name__ == '__main__':
    app.run_server(debug=False,port=8086)

