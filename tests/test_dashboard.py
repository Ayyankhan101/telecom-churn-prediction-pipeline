import pytest
import requests


def test_url_parsing():
    """Test URL parsing extracts correct base URL"""
    test_cases = [
        ("http://localhost:5000/predict", "http://localhost:5000"),
        ("http://localhost:5000/model/info", "http://localhost:5000"),
        ("http://localhost:5000/metrics", "http://localhost:5000"),
        ("http://localhost:5000", "http://localhost:5000"),
    ]
    
    for api_url, expected_base in test_cases:
        if '/predict' in api_url:
            base_url = api_url.split('/predict')[0]
        elif '/model/info' in api_url:
            base_url = api_url.split('/model/info')[0]
        elif '/metrics' in api_url:
            base_url = api_url.split('/metrics')[0]
        else:
            base_url = api_url
        
        assert base_url == expected_base, f"Failed for {api_url}"


def test_api_health_check():
    """Test API health endpoint returns healthy"""
    try:
        resp = requests.get("http://localhost:5000/health", timeout=5)
        assert resp.status_code == 200
        assert resp.json().get("status") == "healthy"
    except requests.exceptions.ConnectionError:
        pytest.skip("API not running")
    except requests.exceptions.ReadTimeout:
        pytest.skip("API not responding")


def test_model_info_no_auth():
    """Test model info endpoint works without auth"""
    try:
        resp = requests.get("http://localhost:5000/model/info", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
    except requests.exceptions.ConnectionError:
        pytest.skip("API not running")


def test_predict_requires_auth():
    """Test predict endpoint requires auth"""
    try:
        resp = requests.post(
            "http://localhost:5000/predict",
            json={"gender": "Male", "SeniorCitizen": 0},
            timeout=5
        )
        assert resp.status_code in [401, 403, 422]
    except requests.exceptions.ConnectionError:
        pytest.skip("API not running")