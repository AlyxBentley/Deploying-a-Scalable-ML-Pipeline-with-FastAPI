import requests

URL = "http://127.0.0.1:8000"

# ---------- GET ----------
r = requests.get(URL)
print("Status Code:", r.status_code)
print("Result:", r.json())


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# ---------- POST ----------
r = requests.post(f"{URL}/data/", json=data)
print("Status Code:", r.status_code)
print("Result:", r.json())
