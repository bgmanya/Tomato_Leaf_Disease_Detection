# Tomato Disease Detection

<a href="https://github.com/georgemuriithi/tomato-disease-detection/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/georgemuriithi/tomato-disease-detection.svg?color=blue&cachedrop">
</a>

This project follows the PRD for tomato leaf disease detection using a lightweight MobileNetV3-based classifier, a web dashboard, and explainable Grad-CAM output. The model compares the public PlantVillage-style dataset with the Kaggle tomato leaf disease dataset and shows disease, confidence, severity, and affected area in the UI.

## Web Page
The browser-based interface is the target deployment. It shows the predicted disease, confidence score, disease severity, percentage of affected leaf area, and a Grad-CAM-style explanation for each diagnosis.

## Problem Description
Farmers face economic losses for failing to identify diseases in their tomato plants or giving them wrong treatments after making incorrect assumptions. It is also an additional cost to hire experts to inspect tomato plants and identify diseases.

## Solution Approach
The system uses a MobileNetV3-Small or MobileNetV3-Large backbone instead of the legacy CNN architecture, and it is optimized for a web-based tomato leaf classifier.

- Data Collection: Public PlantVillage-style dataset and the tomato leaf disease Kaggle dataset
- Model: TensorFlow, MobileNetV3-Small/Large, Grad-CAM explainability
- MLOps: TensorFlow Serving, Docker
- Backend: FastAPI
- Frontend: React JS
- Deployment: GCP (Google Cloud Platform)

### Model
<a href="https://colab.research.google.com/drive/1-4BZ6qLznewBHl65NsbmWRswJijglM5w?usp=sharing">
  <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
</a>

<p align="center">
  <img src="https://user-images.githubusercontent.com/21691211/217465060-e69f8ffa-4027-4ad8-a5ae-f314599f193c.png">
</p>

### Dataset Comparison
Generate the class-count table and SVG graph with:

```bash
.venv312/bin/python generate_dataset_comparison.py
```

The outputs are written to `reports/dataset_comparison.csv` and
`reports/dataset_comparison.svg`. In this workspace, the GitHub project's
prepared dataset contains duplicated/augmented copies of the Kaggle filenames,
so the comparison describes dataset preparation and class balance rather than
independent dataset quality.

## Running Locally
### Project Python Environment

The project uses the Python 3.12 environment in `.venv312` at the repository root. From the project directory, install the dependencies with:

```bash
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -r requirements.txt
```

TensorFlow 2.16.2 still imports the `distutils` compatibility namespace during
startup. The requirements pin the compatible `setuptools` release, and the
Python entry points initialize that compatibility shim before importing
TensorFlow. Always install from `requirements.txt` rather than installing
TensorFlow separately.

In VS Code, open this repository folder (the folder containing this README). The included `.vscode/settings.json` selects `.venv312` automatically.

### Setting up Local API
- ```cd /path/to/tomato-disease-detection-main```
- ```.venv312/bin/python -m pip install -r requirements.txt```
- <a href="https://docs.docker.com/get-docker/">Install Docker</a>
- Get docker image for <a href="https://www.tensorflow.org/tfx/serving/docker">TensorFlow Serving</a>
- Serve a model by running the TF serving container. Change the folder path accordingly and add the appropriate config file at the end of the command. To serve the latest model, add ```all-models.config```. To serve a target model, add ```target-models.config```.
```
docker run -t --rm -p 8605:8605 -v C:\Users\User\tomato-disease-detection:/tomato-disease-detection tensorflow/serving --rest_api_port=8605 --allow_version_labels_for_unavailable_models --model_config_file=/tomato-disease-detection/config-files/
```
- Change the endpoint in ```apis/local/main.py``` accordingly, then run the file in an IDE or using the following command:
```
.venv312/bin/python -m uvicorn apis.local.main:app --reload --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`, with interactive documentation at
`http://127.0.0.1:8000/docs`. The VS Code **Start Tomato API** task and **Tomato API**
debug configuration use the same project-local interpreter.

If your terminal is already inside `apis/local` and shows the nested `.venv312`
environment, install its dependencies once and start the API with:

```bash
cd /path/to/tomato-disease-detection-main/apis/local
.venv312/bin/python -m pip install -r requirements.txt
.venv312/bin/python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Running Web Page
```
cd web-page
npm install
npm start
```

## Running on GCP
### Setting up GCP
- Create <a href="https://console.cloud.google.com/">GCP account</a>
- Create <a href="https://cloud.google.com/appengine/docs/standard/nodejs/building-app/creating-project">GCP project</a>
- Create <a href="https://cloud.google.com/storage/docs/creating-buckets">GCP bucket</a>
- Create ```models``` folder in the GCP bucket
- Upload ```tomato-disease-detection-model.h5``` model from ```models``` folder in the repo to ```models``` folder in the GCP bucket

### Deploying GCF (Google Cloud Functions)
- Rename GCP bucket name in ```apis/gcf/main.py``` accordingly
- <a href="https://cloud.google.com/sdk/docs/install-sdk">Install Google Cloud CLI</a>
- <a href="https://cloud.google.com/sdk/docs/initializing">Initialize Google Cloud CLI</a>
- Deploy ```predict``` GCF using Google Cloud SDK Shell by running the following commands. Change the Python version and Project ID accordingly.
```
cd apis/gcf
gcloud functions deploy predict --runtime python310 --trigger-http --memory 512 --project project-id
```
- Enable Google Cloud Build API if necessary
- Get ```httpsTrigger url``` after deployment

### Running Web Page
- Change the url in the ```.env``` files to ```httpsTrigger url```
- Run accordingly

## Credits
@codebasics
