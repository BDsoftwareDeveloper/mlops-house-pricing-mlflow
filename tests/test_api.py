import requests
import time

def test_predict_request():
    # Wait for API to boot
    time.sleep(5)
    data = {"features": [0.038, 0.02, 5.0, 1.0, 0.02, 3.0, 37.0, -122.0]}
    response = requests.post("http://localhost:8000/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
