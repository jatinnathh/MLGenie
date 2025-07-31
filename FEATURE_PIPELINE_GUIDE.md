# MLGenie Feature Engineering Pipeline Guide

## Overview

The MLGenie system now includes a sophisticated feature engineering pipeline that automatically tracks and applies transformations consistently during training and prediction. This solves the problem of having to manually handle feature engineering for new predictions.

## Key Features

✅ **Automatic Pipeline Tracking**: All feature engineering operations are automatically saved
✅ **Raw Input Prediction**: Users provide raw data, system handles all preprocessing
✅ **Consistent Transformations**: Same transformations applied during training and prediction
✅ **Export Ready Models**: Models include preprocessing pipeline for production use

## How It Works

### 1. Feature Engineering Phase
- User uploads raw dataset
- Applies feature engineering operations (scaling, encoding, etc.)
- System automatically tracks all transformations in a pipeline
- Original dataset structure is preserved for reference

### 2. Model Training Phase
- Model trains on feature-engineered data
- Pipeline information is linked to trained models
- Original feature names and target column are saved

### 3. Prediction Phase
- User provides raw input data (original format)
- System automatically applies all feature engineering transformations
- Model makes prediction on properly preprocessed data

## User Workflow

### Step 1: Upload Data
```
Raw Dataset → Feature Engineering Page
```

### Step 2: Apply Feature Engineering
- Select transformations (scaling, encoding, etc.)
- System automatically tracks each operation
- Pipeline summary shows all applied transformations

### Step 3: Train Models
- Navigate to ML Training page
- Select models and train on engineered data
- Models are linked with preprocessing pipeline

### Step 4: Make Predictions
- Provide raw input data (same format as original upload)
- System automatically preprocesses using saved pipeline
- Get predictions without manual preprocessing

### Step 5: Export Models
- Download complete model package
- Includes model + preprocessing pipeline + usage instructions
- Ready for production deployment

## Example Usage

### Original Data Input
```json
{
  "age": 25,
  "sex": "male",
  "education": "high_school",
  "income": 50000
}
```

### What Happens Internally
1. One-hot encoding: `sex` → `sex_male`, `sex_female`
2. Label encoding: `education` → `education_encoded`
3. Scaling: `age`, `income` → standardized values
4. Model prediction on transformed features

### User Only Needs to Provide
```json
{
  "age": 25,
  "sex": "male", 
  "education": "high_school",
  "income": 50000
}
```

## Benefits

1. **Simplified Prediction**: Users never need to know about feature engineering details
2. **Consistent Results**: Same preprocessing always applied
3. **Production Ready**: Exported models work with raw data
4. **Error Prevention**: No manual preprocessing means no mistakes

## Technical Implementation

### FeatureEngineeringPipeline Class
- Tracks all transformations with parameters
- Applies transformations consistently
- Handles missing values, scaling, encoding, etc.
- Saves/loads pipeline state

### Integration Points
- **Feature_eng.py**: Records transformations during engineering
- **ML_training.py**: Uses pipeline during training and prediction
- **Export**: Includes pipeline in model packages

## Pipeline Operations Supported

1. **Missing Value Handling**: Mean/median/mode imputation
2. **Feature Scaling**: Standard/MinMax scaling
3. **One-Hot Encoding**: Categorical to binary features
4. **Binning**: Numeric to categorical conversion
5. **Target Encoding**: Category to numeric mapping
6. **Frequency Encoding**: Category frequency mapping

## Production Deployment

### Using Exported Models
```python
import pickle
import pandas as pd

# Load complete model package
with open('model_package.pkl', 'rb') as f:
    package = pickle.load(f)

model = package['model']
pipeline = package['feature_pipeline']

# Raw input (what user provides)
raw_input = {"feature1": "value1", "feature2": 123}

# Apply preprocessing pipeline
input_df = pd.DataFrame([raw_input])
processed_features = pipeline.transform(input_df)

# Make prediction
prediction = model.predict(processed_features)
```

## Troubleshooting

### Common Issues
1. **Missing Features**: Ensure all original features are provided
2. **Wrong Data Types**: Match original data types (string/numeric)
3. **Category Values**: Use values that existed in training data

### Debug Information
- Pipeline summary shows all applied transformations
- Prediction interface shows expected raw features
- Error messages guide correct input format

## Migration from Old System

If you have existing models without pipelines:
1. Re-run feature engineering with new system
2. Retrain models to link with pipeline
3. Export new model packages for production

This ensures all future predictions work with raw input data.
