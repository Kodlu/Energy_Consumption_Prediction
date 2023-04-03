from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from random import randint
import uuid
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import pickle

pickled_model1 = pickle.load(open('model_regressor.pkl', 'rb'))
pickled_model2 = pickle.load(open('model_regressor_gpu.pkl', 'rb'))


from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict_cpu")
async def predict_cpu(info : Request):

    print(await info.body())

    req_info = await info.json()

    req_info = dict(req_info)

    a = req_info['CPU_Usage']
    b = req_info['CPU_SOC']

    X = np.array([[a, b]])

    predictions = pickled_model1.predict(X)
    predictions = list(predictions)
    predictions = predictions[0]

    print(predictions)

    return {"Core_SoC" : round(predictions,2)}





@app.post("/predict_gpu")
async def predict_cpu(info : Request):

    print(await info.body())

    req_info = await info.json()

    req_info = dict(req_info)

    a = req_info['GPU_Memory']
    b = req_info['GPU_Temp']

    X = np.array([[a, b]])

    predictions = pickled_model2.predict(X)
    predictions = list(predictions)
    predictions = predictions[0]

    print(predictions)

    return {"GPU_Power" : round(predictions,2)}




