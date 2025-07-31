"""
MLGenie Model Usage Example
===========================

This script demonstrates how to use exported models from MLGenie
with the feature engineering pipeline for production predictions.
"""

import pickle
import pandas as pd
import json

def load_model_package(filename):
    """Load a complete MLGenie model package"""
    try:
        with open(filename, 'rb') as f:
            package = pickle.load(f)
        
        print(f"✅ Loaded model: {package['model_name']}")
        print(f"📊 Original features: {len(package['original_columns'])}")
        print(f"🔧 Transformations: {len(package['feature_pipeline'].transformations)}")
        
        return package
    except FileNotFoundError:
        print(f"❌ Model file '{filename}' not found")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None

def make_prediction(model_package, raw_input):
    """Make a prediction using raw input data"""
    try:
        # Extract components
        model = model_package['model']
        feature_pipeline = model_package['feature_pipeline']
        
        print(f"\n🔄 Processing input with {len(feature_pipeline.transformations)} transformations...")
        
        # Convert input to DataFrame
        if isinstance(raw_input, dict):
            input_df = pd.DataFrame([raw_input])
        else:
            input_df = raw_input
        
        print(f"📥 Raw input: {input_df.iloc[0].to_dict()}")
        
        # Apply feature engineering pipeline
        processed_features = feature_pipeline.transform(input_df)
        
        print(f"🔧 Processed features: {processed_features.shape[1]} columns")
        
        # Make prediction
        prediction = model.predict(processed_features.values)
        
        # Handle classification vs regression
        training_config = model_package.get('training_config', {})
        problem_type = training_config.get('problem_type', 'Unknown')
        
        if problem_type == 'Classification' and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_features.values)
            confidence = max(probabilities[0])
            result = {
                'prediction': prediction[0],
                'confidence': round(confidence, 3),
                'problem_type': problem_type
            }
        else:
            result = {
                'prediction': prediction[0],
                'problem_type': problem_type
            }
        
        print(f"✅ Prediction: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return None

def validate_input(model_package, raw_input):
    """Validate that input contains all required features"""
    try:
        feature_pipeline = model_package['feature_pipeline']
        expected_features = feature_pipeline.original_columns.copy()
        
        # Remove target column if present
        if feature_pipeline.target_column in expected_features:
            expected_features.remove(feature_pipeline.target_column)
        
        provided_features = set(raw_input.keys())
        required_features = set(expected_features)
        
        missing = required_features - provided_features
        extra = provided_features - required_features
        
        if missing:
            print(f"⚠️  Missing features: {list(missing)}")
        if extra:
            print(f"ℹ️  Extra features (ignored): {list(extra)}")
        
        return len(missing) == 0
        
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
        return False

# Example usage
if __name__ == "__main__":
    # Example 1: Load and use a model
    print("MLGenie Model Usage Example")
    print("=" * 40)
    
    # Load model package (replace with your actual model file)
    model_file = "random_forest_complete_model.pkl"
    package = load_model_package(model_file)
    
    if package:
        # Example raw input (modify according to your model's features)
        raw_input = {
            "feature1": 25.0,
            "feature2": "category_a",
            "feature3": 100.5,
            "feature4": "value_x"
        }
        
        print(f"\n📋 Expected features: {package['original_columns']}")
        
        # Validate input
        if validate_input(package, raw_input):
            # Make prediction
            result = make_prediction(package, raw_input)
            
            if result:
                print(f"\n🎯 Final Result:")
                print(f"   Prediction: {result['prediction']}")
                if 'confidence' in result:
                    print(f"   Confidence: {result['confidence']}")
                print(f"   Problem Type: {result['problem_type']}")
        else:
            print("❌ Input validation failed")
    
    print("\n" + "=" * 40)
    print("Example completed!")

# Batch prediction example
def batch_predict(model_package, input_file):
    """Make predictions on a batch of inputs from CSV/JSON file"""
    try:
        if input_file.endswith('.csv'):
            data = pd.read_csv(input_file)
        elif input_file.endswith('.json'):
            with open(input_file, 'r') as f:
                data = pd.DataFrame(json.load(f))
        else:
            print(f"❌ Unsupported file format: {input_file}")
            return None
        
        print(f"📊 Processing {len(data)} samples...")
        
        results = []
        for idx, row in data.iterrows():
            raw_input = row.to_dict()
            result = make_prediction(model_package, raw_input)
            if result:
                results.append({
                    'sample_id': idx,
                    'prediction': result['prediction'],
                    'confidence': result.get('confidence', None)
                })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"❌ Batch prediction error: {str(e)}")
        return None
