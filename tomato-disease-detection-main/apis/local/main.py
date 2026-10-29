# importing packages
from io import BytesIO
import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import requests
import setuptools as _setuptools
import uvicorn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tomato_model_utils import CLASS_NAMES, estimate_affected_area, estimate_severity, gradcam_explanation

app = FastAPI()  # creating FastAPI instance

# allowing requests from port 3000
origins = [
    'http://localhost',
    'http://localhost:3000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# The project requirement is to use the trained six-class MobileNetV3Small model.
# TensorFlow Serving is intentionally ignored here because stale serving models
# can override predictions and use mismatched preprocessing/class ordering.
MODEL_PATH = ROOT / 'models' / 'tomato-mobilenetv3.keras'
_LOCAL_MODEL = None


def _load_local_model():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        import tensorflow as tf
        _LOCAL_MODEL = tf.keras.models.load_model(str(MODEL_PATH))
    return _LOCAL_MODEL


# testing connection
@app.get('/')
async def root():
    return {
        'status': 'ok',
        'message': 'Tomato disease detection API is running.',
        'health': '/ping',
        'docs': '/docs',
    }


@app.get('/ping')
async def ping():  # asynchronous and non-blocking
    return 'Ready!'


# predicting image
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()  # preventing blocking
    img = Image.open(BytesIO(file_bytes)).convert('RGB')
    img_array = np.array(img)

    # Match the MobileNetV3Small training preprocessing exactly:
    # RGB -> resize to 224x224 -> float32 -> MobileNetV3 preprocess_input.
    img_resized = np.array(
        Image.fromarray(img_array).resize((224, 224), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    img_batch = np.expand_dims(img_resized, axis=0)

    if not MODEL_PATH.exists():
        raise RuntimeError('Local MobileNetV3 model is not available.')

    import tensorflow as tf
    img_batch = tf.keras.applications.mobilenet_v3.preprocess_input(img_batch)

    model = _load_local_model()
    pred = model.predict(img_batch, verbose=0)[0]

    pred_class = CLASS_NAMES[int(np.argmax(pred))]
    pred_conf = float(np.max(pred))
    severity = estimate_severity(pred_class, pred_conf, img_array)
    affected_area = estimate_affected_area(img_array)
    explanation = gradcam_explanation(pred_class, severity, affected_area)

    return {
        'pred_class': pred_class,
        'pred_conf': pred_conf,
        'disease': pred_class,
        'confidence': pred_conf,
        'severity': severity,
        'affected_area': affected_area,
        'affected_area_percent': affected_area,
        'explanation': explanation,
        'gradcam_explanation': explanation,
    }


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
