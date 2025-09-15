
"""
AI-Powered Crop Yield Prediction and Optimization System
SIH 2025 - Problem Statement 25044
Government of Odisha - Electronics & IT Department

This system provides:
1. Crop yield prediction using ML models
2. Optimal crop recommendation based on soil and weather conditions
3. Agricultural optimization suggestions
4. Real-time predictions using IoT sensor data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

class CropYieldPredictor:
    """
    Complete Crop Yield Prediction and Optimization System
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Moisture', 
                            'Temperature', 'Humidity', 'Rainfall', 'Area_hectares', 'Crop_Encoded']
        self.crop_names = ['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Maize', 'Groundnut', 
                          'Sesamum', 'Niger', 'Mustard', 'Arhar']
        self.is_trained = False

    def create_dataset(self, n_samples=2000):
        """
        Create synthetic agricultural dataset for training
        """
        np.random.seed(42)

        # Soil parameters
        nitrogen = np.random.uniform(10, 140, n_samples)
        phosphorus = np.random.uniform(5, 145, n_samples)
        potassium = np.random.uniform(5, 205, n_samples)
        ph = np.random.uniform(3.5, 9.0, n_samples)
        moisture = np.random.uniform(14, 100, n_samples)

        # Weather parameters
        temperature = np.random.uniform(8, 43, n_samples)
        humidity = np.random.uniform(14, 99, n_samples)
        rainfall = np.random.uniform(20.2, 298.6, n_samples)

        # Regional parameters
        area_hectares = np.random.uniform(0.5, 10.0, n_samples)
        crop_encoded = np.random.choice(range(len(self.crop_names)), n_samples)

        # Calculate yield based on agricultural knowledge
        yield_base = (
            0.3 * (nitrogen / 140) + 
            0.2 * (phosphorus / 145) + 
            0.2 * (potassium / 205) + 
            0.1 * (1 - abs(ph - 6.5) / 3.0) +
            0.1 * (moisture / 100) +
            0.05 * (1 - abs(temperature - 25) / 20) +
            0.03 * (humidity / 100) +
            0.02 * (rainfall / 300)
        )

        # Crop-specific multipliers (tons per hectare)
        crop_multipliers = {
            0: 4.5,   # Rice
            1: 3.2,   # Wheat
            2: 65.0,  # Sugarcane
            3: 2.8,   # Cotton
            4: 5.5,   # Maize
            5: 1.8,   # Groundnut
            6: 0.8,   # Sesamum
            7: 0.6,   # Niger
            8: 1.2,   # Mustard
            9: 1.5    # Arhar
        }

        yield_prediction = []
        for i in range(n_samples):
            base_yield = yield_base[i] * crop_multipliers[crop_encoded[i]]
            noise = np.random.normal(0, 0.1)
            area_effect = np.log(area_hectares[i] + 1) * 0.2
            final_yield = max(0, base_yield + noise + area_effect)
            yield_prediction.append(final_yield)

        data = pd.DataFrame({
            'Nitrogen': nitrogen,
            'Phosphorus': phosphorus,
            'Potassium': potassium,
            'pH': ph,
            'Moisture': moisture,
            'Temperature': temperature,
            'Humidity': humidity,
            'Rainfall': rainfall,
            'Area_hectares': area_hectares,
            'Crop_Type': [self.crop_names[i] for i in crop_encoded],
            'Crop_Encoded': crop_encoded,
            'Yield_tons_per_hectare': yield_prediction
        })

        return data

    def train_model(self, data=None):
        """
        Train the crop yield prediction model
        """
        if data is None:
            data = self.create_dataset()

        # Prepare features and target
        X = data[self.feature_names]
        y = data['Yield_tons_per_hectare']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train multiple models and select the best
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6),
            'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10)
        }

        best_score = -float('inf')
        best_model_name = None

        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

            if score > best_score:
                best_score = score
                best_model_name = name
                self.model = model

        self.is_trained = True

        # Evaluate final model
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        return {
            'best_model': best_model_name,
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'training_samples': len(data)
        }

    def predict_yield(self, soil_params, weather_params, area=1.0, crop_type='Rice'):
        """
        Predict crop yield for given conditions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")

        # Get crop encoding
        try:
            crop_encoded = self.crop_names.index(crop_type)
        except ValueError:
            crop_encoded = 0  # Default to Rice

        # Prepare input
        input_data = np.array([[
            soil_params['nitrogen'],
            soil_params['phosphorus'],
            soil_params['potassium'],
            soil_params['ph'],
            soil_params['moisture'],
            weather_params['temperature'],
            weather_params['humidity'],
            weather_params['rainfall'],
            area,
            crop_encoded
        ]])

        predicted_yield = self.model.predict(input_data)[0]

        return {
            'crop': crop_type,
            'predicted_yield_tons_per_hectare': predicted_yield,
            'total_yield_tons': predicted_yield * area,
            'area_hectares': area
        }

    def recommend_best_crop(self, soil_params, weather_params, area=1.0):
        """
        Recommend the best crop for given conditions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_model() first.")

        crop_predictions = []

        for crop_name in self.crop_names:
            prediction = self.predict_yield(soil_params, weather_params, area, crop_name)
            crop_predictions.append(prediction)

        # Sort by predicted yield
        crop_predictions.sort(key=lambda x: x['predicted_yield_tons_per_hectare'], reverse=True)

        return crop_predictions

    def generate_optimization_recommendations(self, soil_params):
        """
        Generate agricultural optimization recommendations
        """
        recommendations = []

        # Nutrient recommendations
        if soil_params['nitrogen'] < 50:
            recommendations.append({
                'type': 'fertilizer',
                'message': 'Increase nitrogen fertilizer application',
                'target': '80-120 kg/ha',
                'priority': 'high'
            })
        elif soil_params['nitrogen'] > 120:
            recommendations.append({
                'type': 'fertilizer',
                'message': 'Reduce nitrogen to prevent over-fertilization',
                'target': '80-120 kg/ha',
                'priority': 'medium'
            })

        if soil_params['phosphorus'] < 30:
            recommendations.append({
                'type': 'fertilizer',
                'message': 'Apply phosphorus-rich fertilizer',
                'target': '40-80 kg/ha',
                'priority': 'high'
            })

        if soil_params['potassium'] < 40:
            recommendations.append({
                'type': 'fertilizer',
                'message': 'Increase potassium fertilizer',
                'target': '60-100 kg/ha',
                'priority': 'high'
            })

        # pH recommendations
        if soil_params['ph'] < 5.5:
            recommendations.append({
                'type': 'soil_treatment',
                'message': 'Apply lime to increase soil pH',
                'target': '6.0-7.0',
                'priority': 'high'
            })
        elif soil_params['ph'] > 7.5:
            recommendations.append({
                'type': 'soil_treatment',
                'message': 'Apply organic matter to reduce soil pH',
                'target': '6.0-7.0',
                'priority': 'medium'
            })

        # Moisture recommendations
        if soil_params['moisture'] < 40:
            recommendations.append({
                'type': 'irrigation',
                'message': 'Improve irrigation frequency',
                'target': '60-80%',
                'priority': 'high'
            })
        elif soil_params['moisture'] > 90:
            recommendations.append({
                'type': 'drainage',
                'message': 'Improve drainage to prevent waterlogging',
                'target': '60-80%',
                'priority': 'high'
            })

        return recommendations

    def save_model(self, filepath='crop_yield_model.pkl'):
        """
        Save the trained model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save.")

        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'crop_names': self.crop_names
        }, filepath)

    def load_model(self, filepath='crop_yield_model.pkl'):
        """
        Load a trained model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.crop_names = model_data['crop_names']
        self.is_trained = True

# API Integration Class for Real-time Data
class WeatherAPIIntegration:
    """
    Integration with weather APIs for real-time data
    """

    def __init__(self, api_key=None):
        self.api_key = api_key

    def get_weather_data(self, latitude, longitude):
        """
        Get current weather data (placeholder - integrate with actual API)
        """
        # This would integrate with actual weather APIs like OpenWeatherMap
        # For demonstration, returning sample data
        return {
            'temperature': 28.5,
            'humidity': 72,
            'rainfall': 180
        }

class IoTSensorInterface:
    """
    Interface for IoT soil sensors
    """

    def __init__(self, sensor_config=None):
        self.sensor_config = sensor_config or {}

    def read_soil_parameters(self):
        """
        Read soil parameters from IoT sensors (placeholder)
        """
        # This would integrate with actual IoT sensors
        # For demonstration, returning sample data
        return {
            'nitrogen': 85,
            'phosphorus': 45,
            'potassium': 70,
            'ph': 6.3,
            'moisture': 68
        }

# Main application class
class CropOptimizationApp:
    """
    Complete application for crop yield prediction and optimization
    """

    def __init__(self):
        self.predictor = CropYieldPredictor()
        self.weather_api = WeatherAPIIntegration()
        self.iot_interface = IoTSensorInterface()

    def initialize_system(self):
        """
        Initialize the system by training the model
        """
        print("Initializing Crop Optimization System...")
        training_results = self.predictor.train_model()

        print(f"✅ Model trained successfully!")
        print(f"📊 Best Model: {training_results['best_model']}")
        print(f"🎯 R² Score: {training_results['r2_score']:.4f}")
        print(f"📈 RMSE: {training_results['rmse']:.4f}")

        return training_results

    def get_real_time_prediction(self, latitude=None, longitude=None, area=1.0):
        """
        Get real-time crop prediction using IoT sensors and weather API
        """
        # Get soil data from IoT sensors
        soil_data = self.iot_interface.read_soil_parameters()

        # Get weather data from API
        weather_data = self.weather_api.get_weather_data(latitude, longitude)

        # Get crop recommendations
        recommendations = self.predictor.recommend_best_crop(soil_data, weather_data, area)

        # Get optimization suggestions
        optimizations = self.predictor.generate_optimization_recommendations(soil_data)

        return {
            'soil_conditions': soil_data,
            'weather_conditions': weather_data,
            'crop_recommendations': recommendations[:5],  # Top 5
            'optimization_recommendations': optimizations,
            'timestamp': pd.Timestamp.now().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize the application
    app = CropOptimizationApp()

    # Train the system
    results = app.initialize_system()

    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE REAL-TIME PREDICTION")
    print("="*60)

    prediction = app.get_real_time_prediction(area=2.5)

    print(f"\n🌱 Soil Conditions:")
    for param, value in prediction['soil_conditions'].items():
        print(f"   {param.title()}: {value}")

    print(f"\n🌤️ Weather Conditions:")
    for param, value in prediction['weather_conditions'].items():
        print(f"   {param.title()}: {value}")

    print(f"\n🏆 Top Crop Recommendations:")
    for i, rec in enumerate(prediction['crop_recommendations'], 1):
        print(f"   {i}. {rec['crop']}: {rec['predicted_yield_tons_per_hectare']:.2f} tons/ha")

    print(f"\n💡 Optimization Recommendations:")
    for i, opt in enumerate(prediction['optimization_recommendations'], 1):
        print(f"   {i}. {opt['message']} (Target: {opt['target']})")

    print(f"\n✅ System ready for deployment!")
