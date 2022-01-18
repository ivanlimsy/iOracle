FROM python:3.8

COPY iOracle /iOracle
COPY api /api
COPY raw_data /raw_data
COPY requirements.txt /requirements.txt
COPY service-account-file.json /service-account-file.json
COPY predict.py /predict.py

# RUN pip install --upgrade pip
RUN pip install -r requirements.txt
    
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT