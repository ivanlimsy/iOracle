from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predict_multi import main

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {'greeting': 'Hello world'}

@app.get("/predict/")
def predict(ticker_name):
    prediction = main(ticker_name)
    return prediction

if __name__ == '__main__':
    print(predict('aapl'))