# app/api/interference_router.py
import pandas as pd
from datetime import datetime
import io
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sklearn.preprocessing import MinMaxScaler
import random
import time
import os

# Import the centralized file resolver
from app.utils.file_resolver import resolve_input_file_path

# --- Router and Model Definitions ---
router = APIRouter()

# --- Metric to File Mapping for Interference Router ---
INTERFERENCE_METRIC_FILE_MAP = {
    "interference": {"type": "dynamic", "prefix": "ap_clients_5_min_daily"},
}

class InterferenceAnalysisRequest(BaseModel):
    date: str = Field(
        ...,
        description="Date of the dataset (e.g., '08-08-2025'). Used for time-based filtering and file resolution."
    )
    start_time: Optional[str] = Field(
        None,
        description="Start time of the analysis period (e.g., '10:00:00')."
    )
    end_time: Optional[str] = Field(
        None,
        description="End time of the analysis period (e.g., '12:00:00')."
    )
    limit: int = Field(
        10, ge=1, le=20,
        description="The maximum number of top interference events to return. Defaults to 10, with a maximum limit of 20."
    )
    channel_utilization_threshold: int = Field(
        75, ge=0, le=100,
        description="The utilization percentage above which a channel is recommended for a change."
    )

class RecommendedChannel(BaseModel):
    recommended_alternate_channel: int
    reason: str
    
class ChannelRecommendation(BaseModel):
    ap_name: str
    ap_mac: str
    current_channel: int
    number_of_high_utilization_events: int
    recommended_channel_survey: Optional[RecommendedChannel] = None
    
class InterferenceEvent(BaseModel):
    ap_name: str
    client_mac: str
    ap_mac: str
    timestamp: str
    interference_score: float
    rssi: float
    snr: float
    client_retry_count: int
    total_chutilization: float

class InterferenceAnalysisResponse(BaseModel):
    status: str = "success"
    message: str = "Interference analysis and channel recommendations completed."
    top_interference_events: List[InterferenceEvent]
    channel_change_recommendations: List[ChannelRecommendation]

# --- Helper Functions ---
def _load_and_filter_data(file_path: str, date: str, start_time: Optional[str], end_time: Optional[str]) -> pd.DataFrame:
    """
    Loads the network data from the resolved file path and filters it by a date and optional time range.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Specific error for missing file
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"The file '{file_path}' for the specified date was not found."
        )
    except Exception as e:
        # Specific error for parsing issues
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV from '{file_path}': {e}"
        )
    
    if df.empty:
        return df

    # Ensure 'created_at' column exists before proceeding
    if 'created_at' not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The provided CSV content is missing the 'created_at' column."
        )

    # Coerce to datetime, if all fail, it will be NaT and we'll handle it below
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    if start_time and end_time:
        try:
            start_datetime = pd.to_datetime(f"{date} {start_time}", errors='raise')
            end_datetime = pd.to_datetime(f"{date} {end_time}", errors='raise')
            # Check for invalid date/time combinations after parsing
            if pd.isna(start_datetime) or pd.isna(end_datetime):
                 raise ValueError("Could not convert date or time to valid datetime object.")
        except Exception as e:
            # Specific error for invalid date/time format
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid date or time format. Please check the date ('YYYY-MM-DD' or 'DD-MM-YYYY') and time ('HH:MM:SS'). Error: {e}"
            )
        
        # Filter the DataFrame based on the time range
        df = df[(df['created_at'] >= start_datetime) & (df['created_at'] <= end_datetime)]
    
    return df

def _calculate_interference_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a weighted interference score based on key metrics.
    A higher score indicates a more severe interference event.
    """
    df = df.copy()
    key_metrics = ['rssi', 'snr', 'client_retry_count', 'total_chutilization', 'channel', 'ap_name', 'ap_mac', 'client_mac', 'created_at']
    
    # Check if all required columns exist
    missing_cols = [col for col in key_metrics if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input data is missing required columns: {', '.join(missing_cols)}."
        )

    # Convert numeric columns, handling errors gracefully
    for col in ['rssi', 'snr', 'client_retry_count', 'total_chutilization', 'channel']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Impute missing values
    for col in ['rssi', 'snr', 'client_retry_count', 'total_chutilization']:
        if df[col].isnull().all():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The column '{col}' contains no valid numeric data to process."
            )
        df[col] = df[col].fillna(df[col].mean())
    
    scaler = MinMaxScaler()
    
    # Scale metrics where a higher value is worse
    df['scaled_client_retry_count'] = scaler.fit_transform(df[['client_retry_count']])
    df['scaled_total_chutilization'] = scaler.fit_transform(df[['total_chutilization']])
    
    # Invert scaling for metrics where a lower value is worse (e.g., RSSI, SNR)
    # Check for zero-variance before scaling
    if df['rssi'].max() == df['rssi'].min():
        df['scaled_rssi'] = 0.5 # Default to middle value if no variance
    else:
        df['scaled_rssi'] = 1 - (df['rssi'] - df['rssi'].min()) / (df['rssi'].max() - df['rssi'].min())
    
    if df['snr'].max() == df['snr'].min():
        df['scaled_snr'] = 0.5
    else:
        df['scaled_snr'] = 1 - (df['snr'] - df['snr'].min()) / (df['snr'].max() - df['snr'].min())
    
    df['interference_score'] = (
        0.3 * df['scaled_client_retry_count'] +
        0.3 * df['scaled_snr'] +
        0.2 * df['scaled_rssi'] +
        0.2 * df['scaled_total_chutilization']
    )
    
    return df.sort_values(by='interference_score', ascending=False)

def _get_channel_band(channel: int) -> str:
    """
    Determines if a channel is 2.4 GHz or 5 GHz.
    """
    if 1 <= channel <= 14:
        return '2.4GHz'
    elif 36 <= channel <= 165:
        return '5GHz'
    else:
        return 'unknown'
        
def _simulate_channel_survey(current_channel: int, ap_mac: str) -> RecommendedChannel:
    """
    Simulates a channel survey to recommend an alternate channel.
    NOTE: This is a conceptual simulation, as real survey data is not available.
    """
    band = _get_channel_band(current_channel)
    
    if band == '2.4GHz':
        non_overlapping_channels = [1, 6, 11]
        available_channels = [ch for ch in non_overlapping_channels if ch != current_channel]
        if available_channels:
            alt_channel = random.choice(available_channels)
            return RecommendedChannel(
                recommended_alternate_channel=alt_channel,
                reason=f"Current channel {current_channel} is a poor choice in a congested 2.4GHz band. "
                       f"Channel {alt_channel} is a non-overlapping alternative that should reduce interference."
            )
    
    elif band == '5GHz':
        non_dfs_channels_low = [36, 40, 44, 48]
        non_dfs_channels_high = [149, 153, 157, 161]
        
        if current_channel in non_dfs_channels_low:
            available_channels = [ch for ch in non_dfs_channels_low if ch != current_channel]
            alt_channel = random.choice(available_channels) if available_channels else current_channel
            reason = (f"Channel {current_channel} is in a non-DFS group. "
                      f"Consider switching to {alt_channel} to balance load and avoid co-channel interference from nearby APs.")
        elif current_channel in non_dfs_channels_high:
            available_channels = [ch for ch in non_dfs_channels_high if ch != current_channel]
            alt_channel = random.choice(available_channels) if available_channels else current_channel
            reason = (f"Channel {current_channel} is in a non-DFS group. "
                      f"Consider switching to {alt_channel} to balance load and avoid co-channel interference from nearby APs.")
        else:
            alt_channel = random.choice(non_dfs_channels_high)
            reason = (f"Current channel {current_channel} is a DFS channel. "
                      f"Switching to a non-DFS channel like {alt_channel} is recommended to avoid radar interference.")
                      
        return RecommendedChannel(
            recommended_alternate_channel=alt_channel,
            reason=reason
        )

    return RecommendedChannel(
        recommended_alternate_channel=-1,
        reason="Could not determine a suitable alternate channel from a channel survey."
    )

def _get_channel_recommendations(df: pd.DataFrame, threshold: int) -> List[ChannelRecommendation]:
    """
    Identifies APs with high channel utilization and provides channel change recommendations.
    """
    if 'total_chutilization' not in df.columns or 'channel' not in df.columns:
        return []
    
    # Filter for high utilization events
    high_utilization_df = df[df['total_chutilization'] > threshold]
    
    if high_utilization_df.empty:
        return []
    
    # Group by AP to count the number of high utilization events
    recommendations_df = high_utilization_df.groupby(['ap_name', 'ap_mac', 'channel']).size().reset_index(name='number_of_high_utilization_events')
    
    recommendations_df = recommendations_df.sort_values(by='number_of_high_utilization_events', ascending=False)
    
    recommendations_list = []
    for _, row in recommendations_df.iterrows():
        # Perform a simulated channel survey for each recommended AP
        survey_result = _simulate_channel_survey(row['channel'], row['ap_mac'])
        recommendations_list.append(
            ChannelRecommendation(
                ap_name=row['ap_name'],
                ap_mac=row['ap_mac'],
                current_channel=row['channel'],
                number_of_high_utilization_events=row['number_of_high_utilization_events'],
                recommended_channel_survey=survey_result
            )
        )
    return recommendations_list

# --- API Endpoint ---
@router.post("/interference/analyze", response_model=InterferenceAnalysisResponse)
async def analyze_interference(request: InterferenceAnalysisRequest):
    """
    Analyzes network data to detect client-level interference and provide AP-level
    channel change recommendations based on a simulated channel survey.
    """
    try:
        start_time = time.time()
        
        # Stage 1: File Resolution
        metric_name = "interference"
        try:
            file_path = resolve_input_file_path(request.date, metric_name, INTERFERENCE_METRIC_FILE_MAP)
        except HTTPException as e:
            # Re-raise the more specific HTTP exception from the file resolver
            raise e
        
        # Stage 2: Data Loading and Filtering
        df = _load_and_filter_data(file_path, request.date, request.start_time, request.end_time)
        
        if df.empty:
            return InterferenceAnalysisResponse(
                status="success", # Still a success, but with no data
                message="No data found for the specified date and time range.",
                top_interference_events=[],
                channel_change_recommendations=[]
            )
        
        # Stage 3: Data Processing and Score Calculation
        try:
            scored_df = _calculate_interference_score(df)
        except HTTPException as e:
            # Catch specific data processing errors from the helper function
            raise e
        except Exception as e:
            # Fallback for unexpected data processing errors
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An internal error occurred during score calculation: {str(e)}"
            )

        # Stage 4: Results Aggregation
        try:
            top_events_df = scored_df.head(request.limit)
            top_events_list = [
                InterferenceEvent(
                    ap_name=row['ap_name'],
                    client_mac=row['client_mac'],
                    ap_mac=row['ap_mac'],
                    timestamp=str(row['created_at']),
                    interference_score=row['interference_score'],
                    rssi=row['rssi'],
                    snr=row['snr'],
                    client_retry_count=row['client_retry_count'],
                    total_chutilization=row['total_chutilization']
                )
                for _, row in top_events_df.iterrows()
            ]
            
            channel_recs = _get_channel_recommendations(df, request.channel_utilization_threshold)
        except Exception as e:
            # Fallback for errors during final result preparation
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An internal error occurred while compiling the results: {str(e)}"
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Interference analysis completed in {duration:.2f} seconds.")
        
        return InterferenceAnalysisResponse(
            top_interference_events=top_events_list,
            channel_change_recommendations=channel_recs
        )

    except HTTPException:
        # Re-raise specific HTTP exceptions directly
        raise
    except Exception as e:
        # Final, generic fallback for unhandled exceptions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {str(e)}"
        )