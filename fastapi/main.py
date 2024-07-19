import pandas as pd
import uvicorn
from fastapi import FastAPI
from utils import preprocessing
from pathlib import Path
import pickle
from pydantic import BaseModel, Field

# # # Initialization
app = FastAPI()
artifact_dir = Path("artifacts")

# # # Load artifacts
def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

artifacts = load_pickle(artifact_dir / "artifacts.pkl")
clf = load_pickle(artifact_dir / "model.pkl")

# # # Define input class
class Input(BaseModel):
    gender: str 
    payment_method: str
    age: int 
    download: int 
    charge: int 

# # # route definitions
@app.get("/")
async def root():
    return {"message": "Hello"}

# # # predict route
@app.post("/predict")
async def predict(data: Input):
    df = pd.DataFrame([dict(data)])
    print(data)
    X, _ = preprocessing(df, artifacts)
    y_pred = clf.predict(X)
    return {
        "predict": int(y_pred)
        }

if __name__ == '__main__':
    uvicorn.run(app, reload=True)