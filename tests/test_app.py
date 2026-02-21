import unittest
import os
import sys
from io import BytesIO

# Add parent dir to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

class SkinCareAppTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_predict_no_file(self):
        response = self.app.post('/predict')
        self.assertEqual(response.status_code, 400)

    def test_predict_with_file(self):
        # Create a valid simple JPEG image in memory
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = BytesIO(buffer.tobytes())
        
        data = {
            'file': (img_bytes, 'test.jpg')
        }
        response = self.app.post('/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        
        json_data = response.get_json()
        self.assertIn('condition', json_data)
        self.assertIn('confidence', json_data)
        self.assertIn('recommendations', json_data)
        self.assertIsInstance(json_data['recommendations'], list)

if __name__ == '__main__':
    unittest.main()
