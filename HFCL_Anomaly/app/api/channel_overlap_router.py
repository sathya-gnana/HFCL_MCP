import pandas as pd
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import random
import time
import os

# Import the centralized file resolver
from app.utils.file_resolver import resolve_input_file_path

# --- Router and Model Definitions ---
router = APIRouter()

# --- Metric to File Mapping ---
CHANNEL_METRIC_FILE_MAP = {
    "channel_overlap": {"type": "dynamic", "prefix": "ap_clients_5_min_daily"},
}

class ChannelOverlapRequest(BaseModel):
    date: str = Field(..., description="Date of the dataset (e.g., '2025-08-08').")
    site_name: str = Field(..., description="Name of the site to analyze.")
    start_time: Optional[str] = Field(None, description="Start time of the analysis period (e.g., '10:00:00').")
    end_time: Optional[str] = Field(None, description="End time of the analysis period (e.g., '12:00:00').")
    limit: int = Field(10, ge=1, le=20, description="The maximum number of top overlap events to return.")
    
# OverlapEvent now includes a score to rank severity
class OverlapEvent(BaseModel):
    ap_name_1: str
    ap_name_2: str
    channel: int
    interference_type: str
    rssi: float
    overlap_score: float # New field to quantify severity

# RecommendedChannel now includes more detail for the recommendation
class RecommendedChannel(BaseModel):
    ap_name: str
    current_channel: int
    recommended_channel: int
    reason: str
    ap_overall_score: float # New field to show how bad the AP is

class ChannelOverlapResponse(BaseModel):
    status: str = "success"
    message: str = "Advanced channel overlap analysis completed."
    top_overlap_events: List[OverlapEvent]
    channel_change_recommendations: List[RecommendedChannel]

# --- Helper Functions ---
def _load_and_filter_data(file_path: str, date: str, site_name: str, start_time: Optional[str], end_time: Optional[str]) -> pd.DataFrame:
    """
    Loads network data and filters it by date, site, and optional time range.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File not found: {file_path}")

    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df[df['site_name'] == site_name].copy()

    if start_time and end_time:
        try:
            start_datetime = pd.to_datetime(f"{date} {start_time}")
            end_datetime = pd.to_datetime(f"{date} {end_time}")
            df = df[(df['created_at'] >= start_datetime) & (df['created_at'] <= end_datetime)]
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid date or time format.")
    
    return df

def _get_ap_rssi_data(ap_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Simulates fetching real-world AP-to-AP RSSI data.
    In a real system, this would be a lookup from a database or a network management system.
    Returns a dictionary mapping 'ap_name_1' to another dictionary of 'ap_name_2' and their RSSI.
    """
    rssi_data = {}
    for ap1 in ap_names:
        rssi_data[ap1] = {}
        for ap2 in ap_names:
            if ap1 == ap2:
                continue
            # Simulate RSSI. APs with similar names are assumed to be closer
            # (e.g., 'Zone-D1' and 'Zone-D2')
            if ap1.split('-')[0] == ap2.split('-')[0]:
                simulated_rssi = random.uniform(-65.0, -45.0)
            else:
                simulated_rssi = random.uniform(-80.0, -60.0)
            
            rssi_data[ap1][ap2] = simulated_rssi
    return rssi_data

def _calculate_overlap_score(df: pd.DataFrame) -> List[OverlapEvent]:
    """
    Calculates a numerical score for each channel overlap event, using simulated RSSI data.
    """
    if df.empty: return []

    ap_summary = df.groupby(['ap_name', 'channel']).agg(
        client_count=('client_mac', 'nunique'),
        avg_ch_util=('total_chutilization', 'mean')
    ).reset_index()
    
    ap_names = ap_summary['ap_name'].unique().tolist()
    ap_rssi_data = _get_ap_rssi_data(ap_names)
    
    overlap_events = []
    
    for i, ap1 in ap_summary.iterrows():
        for j, ap2 in ap_summary.iterrows():
            if i >= j: continue
            
            # Use the simulated RSSI data
            simulated_rssi = ap_rssi_data.get(ap1['ap_name'], {}).get(ap2['ap_name'], -100.0)
            
            is_co_channel = ap1['channel'] == ap2['channel']
            is_adjacent = abs(ap1['channel'] - ap2['channel']) <= 4 and ap1['channel'] != ap2['channel']
            
            if (is_co_channel and simulated_rssi > -65.0) or (is_adjacent and simulated_rssi > -60.0):
                # A higher score indicates a more severe overlap
                interference_score = (abs(simulated_rssi) / 100) * (ap1['avg_ch_util'] / 100)
                
                overlap_events.append(OverlapEvent(
                    ap_name_1=ap1['ap_name'],
                    ap_name_2=ap2['ap_name'],
                    channel=ap1['channel'],
                    interference_type="co-channel" if is_co_channel else "adjacent-channel",
                    rssi=simulated_rssi,
                    overlap_score=interference_score
                ))
    
    return sorted(overlap_events, key=lambda x: x.overlap_score, reverse=True)

def _get_channel_band(channel: int) -> str:
    if 1 <= channel <= 14: return '2.4GHz'
    if 36 <= channel <= 165: return '5GHz'
    return 'unknown'
    
def _generate_recommendations(df: pd.DataFrame, overlap_events: List[OverlapEvent]) -> List[RecommendedChannel]:
    """
    Generates recommendations by prioritizing the most problematic APs.
    """
    recommendations = []
    
    # Identify unique APs involved in the top overlap events
    problematic_aps = {e.ap_name_1 for e in overlap_events}
    
    for ap_name in problematic_aps:
        # Get overall stats for the AP
        ap_df = df[df['ap_name'] == ap_name]
        if ap_df.empty: continue
        
        current_channel = ap_df['channel'].iloc[0]
        client_count = ap_df['client_mac'].nunique()
        avg_ch_util = ap_df['total_chutilization'].mean()
        
        # Calculate a single, comprehensive score for this AP
        # A higher score indicates a worse AP, making it a priority for a channel change.
        ap_overall_score = (avg_ch_util / 100) + (client_count / 10) + sum(e.overlap_score for e in overlap_events if e.ap_name_1 == ap_name)
        
        band = _get_channel_band(current_channel)
        
        new_channel = -1
        reason = "No alternative channel found."

        if band == '2.4GHz':
            non_overlapping = [1, 6, 11]
            new_channel = random.choice([ch for ch in non_overlapping if ch != current_channel])
            reason = f"High congestion (util: {avg_ch_util:.1f}%) and channel overlap. Moving to non-overlapping channel {new_channel} recommended."
        elif band == '5GHz':
            non_dfs_channels = [36, 40, 44, 48, 149, 153, 157, 161]
            new_channel = random.choice([ch for ch in non_dfs_channels if ch != current_channel])
            reason = f"Significant overlap and high client load. Recommending switch to a less-used 5GHz non-DFS channel {new_channel}."
            
        recommendations.append(RecommendedChannel(
            ap_name=ap_name,
            current_channel=current_channel,
            recommended_channel=new_channel,
            reason=reason,
            ap_overall_score=ap_overall_score
        ))

    # Sort recommendations by the overall score, so the worst APs are at the top
    recommendations.sort(key=lambda x: x.ap_overall_score, reverse=True)
    return recommendations
    
# --- API Endpoint ---
@router.post("/channel-overlap/analyze", response_model=ChannelOverlapResponse)
async def analyze_channel_overlap(request: ChannelOverlapRequest):
    """
    Analyzes APs for co-channel and adjacent-channel interference using an advanced scoring system,
    and recommends channel changes.
    """
    try:
        start_time = time.time()
        
        # Stage 1: File and Data Loading
        file_path = resolve_input_file_path(request.date, "channel_overlap", CHANNEL_METRIC_FILE_MAP)
        df = _load_and_filter_data(file_path, request.date, request.site_name, request.start_time, request.end_time)
        
        if df.empty:
            return ChannelOverlapResponse(
                message="No data found for the specified site and time range.",
                top_overlap_events=[],
                channel_change_recommendations=[]
            )
        
        # Stage 2: Advanced Overlap Analysis with Scoring
        overlap_events = _calculate_overlap_score(df)
        top_overlap_events = overlap_events[:request.limit]
        
        # Stage 3: Recommendation Generation based on comprehensive AP scores
        recommendations = _generate_recommendations(df, top_overlap_events)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Advanced channel overlap analysis for site '{request.site_name}' completed in {duration:.2f} seconds.")
        
        return ChannelOverlapResponse(
            top_overlap_events=top_overlap_events,
            channel_change_recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {str(e)}"
        )