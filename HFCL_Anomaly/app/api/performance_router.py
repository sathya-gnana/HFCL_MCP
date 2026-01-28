# app/api/performance_router.py
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os
import joblib # For saving/loading models
import pandas as pd
import numpy as np
import time
import warnings # Added to suppress warnings

# New imports for the LSTM Autoencoder model
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed # type inore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # Added for PCA visualization
import matplotlib.pyplot as plt # Added for plotting


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.ensemble._iforest")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")
warnings.filterwarnings("ignore", category=FutureWarning)


# Import the centralized file resolver
from app.utils.file_resolver import resolve_input_file_path

router = APIRouter()

# --- Model Persistence Configuration ---
MODEL_DIR = "./models"
# We will save the entire Keras model directory, not a joblib file
PERFORMANCE_ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_autoencoder_performance.keras") # <--- MODIFIED
# We still use joblib to save the scaler and features
PERFORMANCE_ANOMALY_SCALER_PATH = os.path.join(MODEL_DIR, "lstm_autoencoder_scaler.joblib")

# Ensure model directory exists (can also be done in main.py)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Metric to File Mapping for Performance Router ---
PERFORMANCE_METRIC_FILE_MAP = {
    "performance anomalies": {"type": "dynamic", "prefix": "ap_clients_5_min_daily"},
}

# --- Request Models ---
class PerformanceAnomalyRequest(BaseModel):
    date: str = Field(
        ...,
        description="Date related to the dataset (e.g., 'May 15', '2025-05-15'). "
                    "For 'performance anomalies' metric, this date is for context only, "
                    "as the associated file is static in name. For dynamic metrics, it is used for file resolution."
    )
    metric: str = Field(
        "performance anomalies",
        description="Metric name to identify the dataset (e.g., 'performance anomalies', 'daily performance')."
    )
    output_directory: str = Field(
        "./reports/performance_anomalies/",
        description="Directory to save anomaly reports (CSV) and plots (PNG)."
    )
    generate_plots: bool = Field(
        True,
        description="If true, generates PCA anomaly plot and anomaly score timeline plot."
    )
    anomaly_score_threshold_multiplier: float = Field(
        3.0, ge=0.0,
        description="Multiplier for standard deviation to set the anomaly threshold."
    )
    
    # Training parameters (new for LSTM Autoencoder)
    sequence_length: int = Field(
        20, ge=1,
        description="The number of time steps to look back (sequence length) for the LSTM model."
    )
    epochs: Optional[int] = Field(
        3, ge=1,
        description="Number of epochs for training (used if model needs training)."
    )
    batch_size: Optional[int] = Field(
        32, ge=1,
        description="Batch size for training (used if model needs training)."
    )
    validation_split: Optional[float] = Field(
        0.1, ge=0.0, le=0.5,
        description="Validation split for training (used if model needs training)."
    )

# --- New Models for Anomaly Events and Contributing Metrics ---
class ContributingMetric(BaseModel):
    metric_name: str
    anomaly_contribution_score: float
    original_value: Optional[float] = None
    reconstructed_value: Optional[float] = None

class AnomalyEvent(BaseModel):
    client_mac: str
    ap_mac: str
    ap_name: str
    timestamp: datetime
    anomaly_score: float
    contributing_metrics: List[ContributingMetric]

# --- Response Models ---
class PerformanceAnomalyResponse(BaseModel):
    status: str = "success"
    message: str = "Anomaly detection completed."
    total_anomalies_found: int
    dataset_shape: List[int]
    features_analyzed: List[str]
    model_trained_now: bool = Field(
        False,
        description="True if the model was trained during this API call, False if a pre-existing model was loaded."
    )
    training_duration_seconds: Optional[float] = Field(
        None,
        description="Time taken to train the model in seconds, if trained during this call."
    )
    anomalous_points: List[AnomalyEvent] = Field(
        [],
        description="A list of each detected anomalous point with details about the contributing metrics."
    )


# --- Helper Function for Data Preprocessing (Updated for Time Series) ---
def _preprocess_data(df: pd.DataFrame, features: List[str], is_training: bool = False) -> Dict[str, Any]:
    """
    Internal helper function to preprocess the data for anomaly detection.
    This version is adapted for time series, with a focus on preparing data for
    LSTM models by sorting and handling time-based features.
    """
    df_processed = df.copy()

    # Convert Timestamp to datetime and sort
    if 'created_at' in df_processed.columns:
        df_processed['created_at'] = pd.to_datetime(df_processed['created_at'], errors='coerce')
        df_processed = df_processed.sort_values('created_at').reset_index(drop=True)
    else:
        raise ValueError("The 'created_at' column is required for time-series anomaly detection.")

    # Convert Uptime to seconds
    if 'up_time' in df_processed.columns:
        def uptime_to_seconds(uptime_str):
            if pd.isna(uptime_str) or not isinstance(uptime_str, str):
                return np.nan
            try:
                parts = list(map(int, uptime_str.strip().split(':')))
                if len(parts) == 3: # HH:MM:SS
                    return parts[0] * 3600 + parts[1] * 60 + parts[2]
                elif len(parts) == 2: # MM:SS
                    return parts[0] * 60 + parts[1]
                else:
                    return np.nan
            except ValueError:
                return np.nan
        df_processed['up_time_seconds'] = df_processed['up_time'].apply(uptime_to_seconds)
    
    # Select numerical features for processing based on 'features' parameter
    # We will use the 'up_time_seconds' feature if it exists and 'up_time' is not in features
    actual_numerical_features = [f for f in features if f in df_processed.columns]
    
    # Handle missing values
    for col in actual_numerical_features:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        if df_processed[col].isna().any():
            mean_val = df_processed[col].mean()
            if pd.isna(mean_val) and is_training:
                raise ValueError(f"Column '{col}' is entirely NaN. Cannot impute with mean during training.")
            df_processed[col].fillna(mean_val, inplace=True)
            
    # Check for zero-variance columns (only relevant for training)
    if is_training:
        initial_features = list(actual_numerical_features)
        for col in initial_features:
            if df_processed[col].nunique() <= 1 or df_processed[col].var() < 1e-4:
                print(f"⚠️ Dropping constant or near-constant column during training: {col}")
                actual_numerical_features.remove(col)
        
        if not actual_numerical_features:
            raise ValueError("No variable numerical features remain after filtering during training.")

    # Keep identifier columns, including created_at
    identifier_columns = [col for col in ['client_mac', 'ap_mac', 'ap_name', 'created_at'] if col in df_processed.columns]
    df_final = df_processed[actual_numerical_features + identifier_columns].copy()

    return {
        "df_processed": df_final,
        "numerical_features_used": actual_numerical_features,
        "identifier_columns_used": identifier_columns
    }


# --- Helper Function to create time series sequences ---
def _create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i+sequence_length])
    return np.array(sequences)


# --- Helper Function for Training the LSTM Autoencoder ---
async def _train_and_save_model(
    data_file_path: str,
    sequence_length: int,
    epochs: int,
    batch_size: int,
    validation_split: float,
    model_save_path: str,
    scaler_save_path: str
) -> Dict[str, Any]:
    """
    Internal helper function to train and save the LSTM Autoencoder model.
    """
    start_time = time.time()
    try:
        df_train_raw = pd.read_csv(data_file_path)

        # Define initial features to be used for training.
        # This list can be customized based on your data.
        initial_features_for_model = [
            'rx_bytes', 'tx_bytes', 'rx_packets', 'tx_packets',
            'rssi', 'snr','data_rate_from','data_rate_to'
        ]
        
        # Preprocess the training data
        processed_data_result = _preprocess_data(df_train_raw, initial_features_for_model, is_training=True)
        df_train_cleaned = processed_data_result['df_processed']
        features_for_training = processed_data_result['numerical_features_used']

        if df_train_cleaned[features_for_training].empty or len(df_train_cleaned) < sequence_length:
            raise ValueError(f"Cleaned training data is empty or too short. Need at least {sequence_length} data points.")

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_train_cleaned[features_for_training])
        
        # Create time series sequences for the LSTM
        X_train = _create_sequences(scaled_data, sequence_length)

        # --- Build the LSTM Autoencoder Model ---
        n_features = len(features_for_training)
        model = Sequential()
        # Encoder
        model.add(LSTM(64, activation='relu', input_shape=(sequence_length, n_features), return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(RepeatVector(sequence_length))
        # Decoder
        model.add(LSTM(32, activation='relu', return_sequences=True))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(n_features)))
        
        model.compile(optimizer='adam', loss='mae')
        
        print("Starting LSTM Autoencoder training...")
        history = model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min')]
        )
        print("Training complete.")

        # Save model and the scaler and features used
        model.save(model_save_path)
        joblib.dump({
            'scaler': scaler,
            'features': features_for_training,
            'sequence_length': sequence_length
        }, scaler_save_path)
        
        end_time = time.time()
        
        return {
            "status": "success",
            "message": "Model trained and saved successfully.",
            "duration": (end_time - start_time),
            "features_used": features_for_training
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during model training: {str(e)}")

# --- New Helper Function to analyze contributing metrics for each anomalous point ---
def _analyze_contributing_metrics(
    original_point_scaled: np.ndarray, 
    reconstructed_point_scaled: np.ndarray, 
    features: List[str], 
    scaler: StandardScaler,
    top_n: int = 3
) -> List[ContributingMetric]:
    """
    Calculates the reconstruction error for each feature and identifies the top N
    metrics that contributed most to the anomaly, including their original values.
    """
    # Calculate the absolute difference (error) for each feature
    feature_errors_scaled = np.abs(original_point_scaled - reconstructed_point_scaled)
    
    # Create a dictionary of feature errors
    error_dict = dict(zip(features, feature_errors_scaled))
    
    # Sort the metrics by their error in descending order
    sorted_errors = sorted(error_dict.items(), key=lambda item: item[1], reverse=True)
    
    # Inverse transform the original and reconstructed points to get actual values
    original_point_unscaled = scaler.inverse_transform(original_point_scaled.reshape(1, -1))[0]
    reconstructed_point_unscaled = scaler.inverse_transform(reconstructed_point_scaled.reshape(1, -1))[0]
    
    # Create the list of Pydantic models with the new fields
    contributing_metrics_list = []
    for i in range(top_n):
        if i < len(sorted_errors):
            metric_name, score = sorted_errors[i]
            metric_index = features.index(metric_name)
            original_val = original_point_unscaled[metric_index]
            reconstructed_val = reconstructed_point_unscaled[metric_index]
            contributing_metrics_list.append(
                ContributingMetric(
                    metric_name=metric_name,
                    anomaly_contribution_score=score,
                    original_value=original_val,
                    reconstructed_value=reconstructed_val
                )
            )
            
    return contributing_metrics_list


# --- API Endpoint ---
@router.post("/performance/detect_anomalies", response_model=PerformanceAnomalyResponse)
async def detect_performance_anomalies(request: PerformanceAnomalyRequest):
    """
    Detects network performance anomalies using a pre-trained or newly trained
    LSTM Autoencoder model on time series data.
    """
    model_trained_now = False
    training_duration = None
    model = None
    scaler = None
    features_used_for_model = []
    metric_const = 'performance anomalies'
    sequence_length = request.sequence_length

    try:
        input_file_path = resolve_input_file_path(request.date, metric_const, PERFORMANCE_METRIC_FILE_MAP)
    except HTTPException as e:
        raise e

    # Check if a trained model and scaler exist
    if not (os.path.exists(PERFORMANCE_ANOMALY_MODEL_PATH) and os.path.exists(PERFORMANCE_ANOMALY_SCALER_PATH)):
        print(f"No trained model found. Initiating training...")
        model_trained_now = True
        
        train_result = await _train_and_save_model(
            data_file_path=input_file_path,
            sequence_length=sequence_length,
            epochs=request.epochs,
            batch_size=request.batch_size,
            validation_split=request.validation_split,
            model_save_path=PERFORMANCE_ANOMALY_MODEL_PATH,
            scaler_save_path=PERFORMANCE_ANOMALY_SCALER_PATH
        )
        training_duration = train_result.get("duration")
        features_used_for_model = train_result.get("features_used", [])

        # Load the newly trained model
        model = load_model(PERFORMANCE_ANOMALY_MODEL_PATH)
        model_data = joblib.load(PERFORMANCE_ANOMALY_SCALER_PATH)
        scaler = model_data['scaler']
        features_used_for_model = model_data['features']
        
    else:
        print(f"Trained model found. Loading model and scaler...")
        try:
            model = load_model(PERFORMANCE_ANOMALY_MODEL_PATH)
            model_data = joblib.load(PERFORMANCE_ANOMALY_SCALER_PATH)
            scaler = model_data['scaler']
            features_used_for_model = model_data['features']
            sequence_length = model_data['sequence_length'] # Use the original sequence length
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error loading pre-trained model: {str(e)}. Consider deleting '{PERFORMANCE_ANOMALY_MODEL_PATH}' and retraining.")

    # Proceed with anomaly detection
    try:
        df_original = pd.read_csv(input_file_path)
        
        # Preprocess input data for prediction
        processed_input_result = _preprocess_data(df_original, features_used_for_model, is_training=False)
        df_input_for_prediction = processed_input_result['df_processed']
        numerical_features_actual = processed_input_result['numerical_features_used']

        # Ensure input data has the features the model was trained on
        missing_features = [f for f in features_used_for_model if f not in numerical_features_actual]
        if missing_features:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Input data is missing features required by the trained model: {', '.join(missing_features)}.")
            
        if len(df_input_for_prediction) < sequence_length:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Input data is too short for the trained model. Need at least {sequence_length} data points.")

        # Scale and create sequences for prediction
        scaled_data_detect = scaler.transform(df_input_for_prediction[features_used_for_model])
        X_test = _create_sequences(scaled_data_detect, sequence_length)
        
        # Get predictions (reconstructions) from the model
        X_pred = model.predict(X_test, verbose=0)
        
        # Calculate reconstruction error (MAE) for the entire sequence
        reconstruction_errors_per_sequence = np.mean(np.abs(X_pred - X_test), axis=(1, 2))
        
        # We need to align the errors with the original dataframe
        error_df = pd.DataFrame(
            index=df_input_for_prediction.index[sequence_length - 1:], 
            data={'Anomaly_Score': reconstruction_errors_per_sequence}
        )
        df_original['Anomaly_Score'] = pd.Series(error_df['Anomaly_Score'])
        df_original['Anomaly_Score'] = df_original['Anomaly_Score'].fillna(np.nan) # NaN for the first `sequence_length-1` rows


        # Calculate threshold on anomaly scores (MAE)
        valid_scores = df_original['Anomaly_Score'].dropna()
        if valid_scores.empty:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No valid anomaly scores generated for thresholding.")

        mean_score = valid_scores.mean()
        std_dev_score = valid_scores.std()
        anomaly_threshold_final = mean_score + request.anomaly_score_threshold_multiplier * std_dev_score
        
        df_original['Is_Anomaly'] = (df_original['Anomaly_Score'] > anomaly_threshold_final).fillna(False)

        # --- Generate the API response with contributing metrics ---
        anomalous_points_list = []
        if not df_original[df_original['Is_Anomaly']].empty:
            anomalous_indices = df_original[df_original['Is_Anomaly']].index.tolist()
            
            for idx in anomalous_indices:
                error_idx = idx - (sequence_length - 1)
                
                original_point_scaled = X_test[error_idx][-1]
                reconstructed_point_scaled = X_pred[error_idx][-1]
                
                contributing_metrics = _analyze_contributing_metrics(
                    original_point_scaled, 
                    reconstructed_point_scaled, 
                    features_used_for_model,
                    scaler
                )
                
                original_row = df_original.loc[idx]
                
                anomalous_points_list.append(
                    AnomalyEvent(
                        client_mac=original_row['client_mac'],
                        ap_mac=original_row['ap_mac'],
                        ap_name=original_row['ap_name'],
                        timestamp=original_row['created_at'].to_pydatetime(),
                        anomaly_score=original_row['Anomaly_Score'],
                        contributing_metrics=contributing_metrics
                    )
                )

        # --- Generate the anomaly report CSV with all metrics and analysis ---
        os.makedirs(request.output_directory, exist_ok=True)
        report_csv_path = os.path.join(request.output_directory, "lstm_anomalies_report_with_analysis.csv")
        
        # Prepare a list of dictionaries to build the final DataFrame
        report_data = []
        
        if anomalous_points_list:
            # Iterate through the anomaly list and the original indices
            for i, idx in enumerate(anomalous_indices):
                # Access the original row directly by its index (guaranteed to exist)
                original_row_dict = df_original.loc[idx].to_dict()
                
                # Get the corresponding AnomalyEvent from the list
                anomaly_event = anomalous_points_list[i]
                
                # Add the root cause analysis metrics to the dictionary
                for j, metric in enumerate(anomaly_event.contributing_metrics):
                    original_row_dict[f'contributing_metric_{j+1}_name'] = metric.metric_name
                    original_row_dict[f'contributing_metric_{j+1}_score'] = metric.anomaly_contribution_score
                    original_row_dict[f'contributing_metric_{j+1}_original_value'] = metric.original_value
                    original_row_dict[f'contributing_metric_{j+1}_reconstructed_value'] = metric.reconstructed_value
                
                report_data.append(original_row_dict)
        
        # Create the new DataFrame and save to CSV
        if report_data:
            analysis_df = pd.DataFrame(report_data)
            # Remove the 'Is_Anomaly' column as it's redundant
            if 'Is_Anomaly' in analysis_df.columns:
                analysis_df = analysis_df.drop(columns=['Is_Anomaly'])
            analysis_df.to_csv(report_csv_path, index=False)
            print(f"Anomalies report with full metrics and analysis saved to: {report_csv_path}")
        else:
            print("No anomalies found, no analysis report created.")
        
        # --- Anomaly timeline plot (Updated for MAE scores) ---
        timeline_plot_path = None
        if request.generate_plots:
            try:
                df_plot = df_original.dropna(subset=['created_at', 'Anomaly_Score']).sort_values('created_at')

                plt.figure(figsize=(15, 7))
                plt.plot(df_plot['created_at'], df_plot['Anomaly_Score'], label='Reconstruction Error (MAE)', color='blue', alpha=0.7)
                plt.scatter(df_plot[df_plot['Is_Anomaly']]['created_at'],
                            df_plot[df_plot['Is_Anomaly']]['Anomaly_Score'],
                            color='red', s=50, label='Anomaly')
                plt.axhline(y=anomaly_threshold_final, color='green', linestyle='--', label=f'Threshold ({anomaly_threshold_final:.2f})')
                plt.title('Reconstruction Error Over Time (LSTM Autoencoder)')
                plt.xlabel('Timestamp')
                plt.ylabel('Reconstruction Error (MAE)')
                plt.legend()
                plt.tight_layout()
                timeline_plot_path = os.path.join(request.output_directory, 'anomaly_timeline_lstm.png')
                plt.savefig(timeline_plot_path)
                plt.close()
                print(f"Anomaly timeline plot saved to: {timeline_plot_path}")
            except Exception as e:
                print(f"Warning: Could not generate timeline plot due to error: {e}")
                timeline_plot_path = None
        
        return PerformanceAnomalyResponse(
            status="success",
            message=f"Anomaly detection completed. Model {'trained' if model_trained_now else 'loaded'} and used.",
            total_anomalies_found=len(anomalous_points_list),
            dataset_shape=list(df_original.shape),
            features_analyzed=features_used_for_model,
            model_trained_now=model_trained_now,
            training_duration_seconds=training_duration,
            # anomalous_points=anomalous_points_list
        )
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Input data file not found.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during anomaly detection: {str(e)}")