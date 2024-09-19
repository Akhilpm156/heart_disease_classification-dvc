# heart_disease_classification-using-DVC

<h2>Project Structure</h2>

<pre>
heart_disease/
├── data/
│   ├── archive.zip
│   ├── train.csv
│   ├── test.csv
│   └── unzipped_data/
├── models/
│   └── model.pkl
├── notebooks/
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
├── app.py
├── dvc.yaml
├── config.yaml
├── requirements.txt
└── README.md
</pre>





## Fast API

uvicorn src.app:app --host 0.0.0.0 --port 8000
 
#### input features

'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg','thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'


## Training Pipline DVC

dvc repro

