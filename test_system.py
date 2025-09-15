
"""
Unit tests for the Crop Yield Prediction System
"""

import unittest
import numpy as np
from complete_crop_prediction_system import CropYieldPredictor, CropOptimizationApp

class TestCropPrediction(unittest.TestCase):

    def setUp(self):
        self.predictor = CropYieldPredictor()
        self.predictor.train_model()

    def test_model_training(self):
        """Test if model trains successfully"""
        self.assertTrue(self.predictor.is_trained)
        self.assertIsNotNone(self.predictor.model)

    def test_yield_prediction(self):
        """Test yield prediction functionality"""
        soil_params = {
            'nitrogen': 80,
            'phosphorus': 40,
            'potassium': 60,
            'ph': 6.2,
            'moisture': 65
        }

        weather_params = {
            'temperature': 28,
            'humidity': 70,
            'rainfall': 180
        }

        result = self.predictor.predict_yield(soil_params, weather_params, area=2.0)

        self.assertIsInstance(result, dict)
        self.assertIn('predicted_yield_tons_per_hectare', result)
        self.assertGreater(result['predicted_yield_tons_per_hectare'], 0)

    def test_crop_recommendation(self):
        """Test crop recommendation functionality"""
        soil_params = {
            'nitrogen': 80,
            'phosphorus': 40,
            'potassium': 60,
            'ph': 6.2,
            'moisture': 65
        }

        weather_params = {
            'temperature': 28,
            'humidity': 70,
            'rainfall': 180
        }

        recommendations = self.predictor.recommend_best_crop(soil_params, weather_params)

        self.assertIsInstance(recommendations, list)
        self.assertEqual(len(recommendations), 10)  # All crop types
        self.assertIn('crop', recommendations[0])

    def test_optimization_recommendations(self):
        """Test optimization recommendations"""
        soil_params = {
            'nitrogen': 30,  # Low nitrogen
            'phosphorus': 20,  # Low phosphorus
            'potassium': 30,   # Low potassium
            'ph': 5.0,         # Acidic soil
            'moisture': 30     # Low moisture
        }

        recommendations = self.predictor.generate_optimization_recommendations(soil_params)

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

class TestCropOptimizationApp(unittest.TestCase):

    def setUp(self):
        self.app = CropOptimizationApp()
        self.app.initialize_system()

    def test_system_initialization(self):
        """Test system initialization"""
        self.assertTrue(self.app.predictor.is_trained)

    def test_real_time_prediction(self):
        """Test real-time prediction functionality"""
        prediction = self.app.get_real_time_prediction(area=1.5)

        self.assertIsInstance(prediction, dict)
        self.assertIn('soil_conditions', prediction)
        self.assertIn('weather_conditions', prediction)
        self.assertIn('crop_recommendations', prediction)
        self.assertIn('optimization_recommendations', prediction)

if __name__ == '__main__':
    unittest.main()
