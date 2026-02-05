import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import apply_label, process_data
from ml.model import inference, load_model
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

path = "model/encoder.pkl"
encoder = load_model(path)

path = "model/model.pkl" 
model = load_model(path)

app = FastAPI()

API_KEY = "secret-key-123"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )


@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <html>
      <body>
        <h2>Income Prediction API</h2>
        <ul>
          <li><a href="/ui">Open the Front End (UI)</a></li>
          <li><a href="/docs">Open Swagger Docs</a></li>
        </ul>
      </body>
    </html>
    """


@app.post("/data/")
async def post_inference(
    data: Data,
    api_key: str = Depends(verify_api_key),
):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()

    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    data_processed, _, _, _ = process_data(
        X=data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=None,
    )

    _inference = inference(model, data_processed)
    return {"result": apply_label(_inference)}


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return """
    <html>
        <head>
            <title>Income Prediction API</title>
        </head>
        <body>
            <h2>Income Prediction</h2>
            <p>Click the button to run inference.</p>

            <button onclick="runInference()">Run Prediction</button>

            <pre id="output"></pre>

            <script>
                async function runInference() {
                    const data = {
                    age: 37,
                    workclass: "Private",
                    fnlgt: 178356,
                    education: "HS-grad",
                    "education-num": 10,
                    "marital-status": "Married-civ-spouse",
                    occupation: "Prof-specialty",
                    relationship: "Husband",
                    race: "White",
                    sex: "Male",
                    "capital-gain": 0,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States"
                    };

                    const response = await fetch("/data/", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-API-Key": "secret-key-123"
                    },
                    body: JSON.stringify(data)
                    });

                    const resultText = await response.text();
                    document.getElementById("output").textContent =
                    `HTTP ${response.status}\n\n${resultText}`;
                }
                </script>

        </body>
    </html>
    """
