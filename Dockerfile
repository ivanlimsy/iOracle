FROM python:3.8

COPY TaxiFareModel /TaxiFareModel
COPY api /api
COPY requirements.txt /requirements.txt
COPY model.joblib /model.joblib
COPY /home/derrick/code/darkwing2025/gcp/le-wagon-bootcamp-334008-40185761a7cd.json /credentials.json

# RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT