import requests
import json

endpoint = 'http://localhost:8000/predict'

data = {"gender": "Male", "payment_method": "Credit Card", "age": 20, "download": 10, "charge": 50 }

res = requests.post(endpoint, json=data)

print(json.loads(res.content.decode("utf-8")))