import numpy as np
from tensorflow import keras
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

model.load_state_dict(torch.load('model.pth'))

model.eval()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.get_json()
    x_input = data['input1']
    y_input = data['input2']

    x_tensor = torch.tensor([x_input,y_input],dtype = torch.float32)

    prediction = model(x_tensor).round()

    json_response = jsonify({"Prediction" : prediction.item()})

    return json_response

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port)