import requests
import time

BASE_URL = "http://localhost:8000"
API_KEY = None


def test_health():
    """Test health check"""
    print("\n" + "=" * 60)
    print("Testing Health Check")
    print("=" * 60)
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    response = requests.get(f"{BASE_URL}/api/v1/models/health", headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200


def test_register():
    """Test user registration"""
    global API_KEY
    
    print("\n" + "=" * 60)
    print("Testing User Registration")
    print("=" * 60)
    
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/register",
        json={
            "username": f"test_user_{int(time.time())}",
            "email": f"test_{int(time.time())}@example.com",
            "password": "test_password123"
        }
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"User: {data['username']}")
    print(f"API Key: {data['api_key']}")
    
    API_KEY = data['api_key']
    assert response.status_code == 200


def test_login():
    """Test admin login"""
    print("\n" + "=" * 60)
    print("Testing Admin Login")
    print("=" * 60)
    
    response = requests.post(
        f"{BASE_URL}/api/v1/auth/login",
        json={
            "username": "admin",
            "password": "admin123"
        }
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Token received: {data['access_token'][:50]}...")
    assert response.status_code == 200


def test_news_detection():
    """Test news detection"""
    print("\n" + "=" * 60)
    print("Testing News Detection")
    print("=" * 60)
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    response = requests.post(
        f"{BASE_URL}/api/v1/detect/news",
        headers=headers,
        json={
            "content": "Scientists have discovered a revolutionary cure that eliminates all known diseases overnight!",
            "threshold": 0.7
        }
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Verdict: {data['verdict']}")
    print(f"Confidence: {data['confidence']:.2%}")
    print(f"Processing Time: {data['processing_time_ms']:.2f}ms")
    assert response.status_code == 200


def test_sms_detection():
    """Test SMS detection"""
    print("\n" + "=" * 60)
    print("Testing SMS Detection")
    print("=" * 60)
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    response = requests.post(
        f"{BASE_URL}/api/v1/detect/sms",
        headers=headers,
        json={
            "content": "URGENT! Click here to claim your $1000 prize now: http://fake-link.com",
            "threshold": 0.7
        }
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Verdict: {data['verdict']}")
    print(f"Confidence: {data['confidence']:.2%}")
    assert response.status_code == 200


def test_auto_detection():
    """Test auto-routing detection"""
    print("\n" + "=" * 60)
    print("Testing Auto Detection")
    print("=" * 60)
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    response = requests.post(
        f"{BASE_URL}/api/v1/detect",
        headers=headers,
        json={
            "content": "This is a test message to see which model gets selected",
        }
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Model Used: {data['model']}")
    print(f"Verdict: {data['verdict']}")
    assert response.status_code == 200


def test_history():
    """Test history retrieval"""
    print("\n" + "=" * 60)
    print("Testing History")
    print("=" * 60)
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    response = requests.get(
        f"{BASE_URL}/api/v1/history?page=1&page_size=5",
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total Requests: {data['total']}")
    print(f"Items on Page: {len(data['items'])}")
    assert response.status_code == 200


def test_user_stats():
    """Test user statistics"""
    print("\n" + "=" * 60)
    print("Testing User Stats")
    print("=" * 60)
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    response = requests.get(
        f"{BASE_URL}/api/v1/stats",
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total Requests: {data['total_requests']}")
    print(f"Quota Remaining: {data['quota_remaining']}/{data['quota_limit']}")
    assert response.status_code == 200


def test_models_list():
    """Test models list"""
    print("\n" + "=" * 60)
    print("Testing Models List")
    print("=" * 60)
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    response = requests.get(
        f"{BASE_URL}/api/v1/models",
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    data = response.json()
    for model in data['models']:
        print(f"  - {model['name']}: {model['type']} (loaded: {model['loaded']})")
    assert response.status_code == 200


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("TruthShield API Test Suite")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    
    try:
        # Authenticate first before running any tests
        test_register()  # Creates user and gets API key
        test_login()     # Test JWT login with admin
        
        # Now run tests that require authentication
        test_health()
        test_news_detection()
        test_sms_detection()
        test_auto_detection()
        test_history()
        test_user_stats()
        test_models_list()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
    
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {str(e)}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()
