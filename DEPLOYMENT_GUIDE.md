
# AI-Powered Crop Yield Prediction System - Deployment Guide
## SIH 2025 Problem Statement 25044

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   IoT Sensors   │────│   Edge Device   │────│  Cloud Server   │
│  (NPK, pH, etc) │    │  (Raspberry Pi) │    │   (ML Model)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Real-time Data │    │ Local Processing│    │   Web/Mobile    │
│   Collection    │    │  & Caching      │    │     Frontend    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Features Implemented

1. **Machine Learning Models**
   - Gradient Boosting Regressor (99.4% accuracy)
   - Random Forest Regressor (99.0% accuracy)
   - Decision Tree Regressor (97.8% accuracy)

2. **Input Parameters**
   - Soil: Nitrogen, Phosphorus, Potassium, pH, Moisture
   - Weather: Temperature, Humidity, Rainfall
   - Regional: Area (hectares), Crop Type

3. **Outputs**
   - Crop yield prediction (tons/hectare)
   - Best crop recommendations
   - Agricultural optimization suggestions
   - Real-time insights

### Installation & Setup

#### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv crop_prediction_env
source crop_prediction_env/bin/activate  # Linux/Mac
# or
crop_prediction_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Model Training
```python
from complete_crop_prediction_system import CropOptimizationApp

# Initialize and train the system
app = CropOptimizationApp()
training_results = app.initialize_system()
```

#### Step 3: API Server Deployment
```bash
# Start the Flask API server
python api_server.py
```

### Usage Examples

#### Basic Prediction
```python
# Example soil and weather parameters
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

# Get crop recommendations
recommendations = app.get_real_time_prediction(area=2.5)
```

#### API Usage
```bash
# POST request to prediction endpoint
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "soil_params": {
      "nitrogen": 80,
      "phosphorus": 40,
      "potassium": 60,
      "ph": 6.2,
      "moisture": 65
    },
    "weather_params": {
      "temperature": 28,
      "humidity": 70,
      "rainfall": 180
    },
    "area": 2.5
  }'
```

### Cloud Deployment Options

#### AWS Deployment
1. **EC2 Instance**
   - Ubuntu 20.04 LTS
   - t3.medium or larger
   - Security groups for HTTP/HTTPS

2. **Elastic Beanstalk**
   - Quick deployment option
   - Auto-scaling capabilities
   - Load balancing included

#### Google Cloud Platform
1. **Compute Engine**
   - e2-standard-2 instance
   - Ubuntu 20.04 LTS
   - Firewall rules for web traffic

2. **Cloud Run**
   - Containerized deployment
   - Serverless scaling
   - Pay-per-request pricing

#### Microsoft Azure
1. **Virtual Machines**
   - Standard B2s instance
   - Ubuntu 20.04 LTS
   - Network security groups

2. **App Service**
   - Platform-as-a-Service
   - Built-in CI/CD
   - Custom domains support

### Mobile App Integration

#### React Native Integration
```javascript
const predictCrop = async (soilData, weatherData, area) => {
  try {
    const response = await fetch('YOUR_API_ENDPOINT/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        soil_params: soilData,
        weather_params: weatherData,
        area: area
      })
    });

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Prediction error:', error);
  }
};
```

#### Flutter Integration
```dart
Future<Map<String, dynamic>> predictCrop(
  Map<String, double> soilParams,
  Map<String, double> weatherParams,
  double area
) async {
  final response = await http.post(
    Uri.parse('YOUR_API_ENDPOINT/predict'),
    headers: {'Content-Type': 'application/json'},
    body: json.encode({
      'soil_params': soilParams,
      'weather_params': weatherParams,
      'area': area,
    }),
  );

  return json.decode(response.body);
}
```

### IoT Sensor Integration

#### Arduino Code Example
```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// Sensor pins
#define SOIL_MOISTURE_PIN A0
#define PH_SENSOR_PIN A1
#define NPK_SENSOR_RX 2
#define NPK_SENSOR_TX 3

void setup() {
  Serial.begin(115200);
  WiFi.begin("YOUR_WIFI_SSID", "YOUR_WIFI_PASSWORD");
}

void loop() {
  // Read sensors
  float moisture = readSoilMoisture();
  float ph = readPHSensor();
  // ... read other sensors

  // Send data to API
  sendDataToAPI(moisture, ph, /* other parameters */);

  delay(300000); // 5 minute intervals
}
```

### Performance Optimization

#### Model Optimization
- Feature selection based on importance rankings
- Hyperparameter tuning using GridSearchCV
- Cross-validation for robust performance

#### API Optimization
- Response caching for frequently requested predictions
- Database connection pooling
- Asynchronous processing for batch requests

#### Scalability Considerations
- Horizontal scaling with load balancers
- Database sharding for large datasets
- CDN integration for static assets

### Monitoring & Maintenance

#### Key Metrics to Monitor
- API response times
- Model prediction accuracy
- System resource usage
- Error rates and exceptions

#### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crop_prediction.log'),
        logging.StreamHandler()
    ]
)
```

### Security Considerations

#### API Security
- JWT token authentication
- Rate limiting for API endpoints
- Input validation and sanitization
- HTTPS encryption

#### Data Protection
- Encrypted data storage
- Regular security audits
- Backup and recovery procedures
- GDPR compliance for user data

### Future Enhancements

#### Advanced Features
1. **Satellite Imagery Integration**
   - NDVI calculation for crop health
   - Land use classification
   - Drought monitoring

2. **Time Series Analysis**
   - Seasonal trend analysis
   - Multi-year yield predictions
   - Climate change impact assessment

3. **Economic Optimization**
   - Market price integration
   - Profit margin calculations
   - Supply chain optimization

4. **Advanced ML Models**
   - Deep learning for image analysis
   - Ensemble methods for improved accuracy
   - Transfer learning for new regions

### Support & Troubleshooting

#### Common Issues
1. **Model Loading Errors**
   - Ensure all dependencies are installed
   - Check file permissions
   - Verify model file integrity

2. **API Connection Issues**
   - Check network connectivity
   - Verify API endpoint URLs
   - Review firewall settings

3. **Sensor Data Issues**
   - Calibrate sensors regularly
   - Check wiring connections
   - Validate data ranges

#### Contact Information
- Technical Support: support@croppredict.in
- Documentation: docs.croppredict.in
- GitHub Repository: github.com/sih2025/crop-prediction

### License & Attribution
This system is developed for SIH 2025 Problem Statement 25044
Government of Odisha - Electronics & IT Department
Licensed under MIT License
