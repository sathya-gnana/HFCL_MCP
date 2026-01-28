import pandas as pd
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import os

# Import the centralized file resolver. This assumes a file structure like:
# project_root/
# ├── app/
# │   ├── utils/
# │   │   └── file_resolver.py
# │   └── api/
# │       └── [your_api_file].py
from app.utils.file_resolver import resolve_input_file_path

# --- Router and Model Definitions ---
router = APIRouter()

# --- Metric to File Mapping (for the centralized resolver) ---
AP_OVERLOAD_METRIC_FILE_MAP = {
    "ap_overload": {"type": "dynamic", "prefix": "ap_clients_5_min_daily"},
}

# --- Pydantic Models ---
class OverloadMetrics(BaseModel):
    ap_name: str
    client_count: int
    avg_ch_util: float

class LoadBalancingRecommendation(BaseModel):
    overloaded_ap: str
    overloaded_metrics: OverloadMetrics
    recommendation: str
    neighbor_ap: str
    neighbor_metrics: OverloadMetrics

class APOverloadRequest(BaseModel):
    date: str = Field(..., description="Date of the dataset (e.g., '2025-08-08').")
    site_name: str = Field(..., description="Name of the site to analyze.")
    start_time: Optional[str] = Field(None, description="Start time of the analysis period (e.g., '10:00:00').")
    end_time: Optional[str] = Field(None, description="End time of the analysis period (e.g., '12:00:00').")
    max_clients_threshold: int = Field(50, ge=1, description="Threshold for max clients per AP.")
    max_utilization_threshold: int = Field(80, ge=1, le=100, description="Threshold for max channel utilization.")

class APOverloadResponse(BaseModel):
    status: str = "success"
    message: str = "AP overload analysis and load balancing recommendations completed."
    overloaded_aps: List[OverloadMetrics]
    recommendations: List[LoadBalancingRecommendation]

# --- Helper Functions ---
def _load_and_filter_data(file_path: str, site_name: str, date: str, start_time: Optional[str], end_time: Optional[str]) -> pd.DataFrame:
    """Loads network data and filters it by site and optional time range."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File not found: {file_path}")
    
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    filtered_df = df[df['site_name'] == site_name].copy()

    if start_time and end_time:
        try:
            start_datetime = pd.to_datetime(f"{date} {start_time}")
            end_datetime = pd.to_datetime(f"{date} {end_time}")
            filtered_df = filtered_df[(filtered_df['created_at'] >= start_datetime) & (filtered_df['created_at'] <= end_datetime)]
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid date or time format.")

    if filtered_df.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No data found for the specified site and time range.")
    
    return filtered_df

def _get_ap_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates client count and channel utilization per AP."""
    ap_summary = df.groupby('ap_name').agg(
        client_count=('client_mac', 'nunique'),
        avg_ch_util=('total_chutilization', 'mean')
    ).reset_index()
    return ap_summary

def _find_neighbor_aps(df: pd.DataFrame) -> Dict[str, str]:
    """
    Finds a single "best" neighboring AP for each AP based on shared clients with strong signals.
    Returns a dictionary mapping an AP to its best neighbor.
    """
    STRONG_SIGNAL_THRESHOLD = -65  # dBm
    strong_signal_df = df[df['rssi'] >= STRONG_SIGNAL_THRESHOLD].copy()

    # Create a list of shared AP pairs for each client
    shared_ap_list = strong_signal_df.groupby('client_mac')['ap_name'].apply(
        lambda x: [(ap1, ap2) for i, ap1 in enumerate(x) for ap2 in x[i+1:]]
    )
    
    # Flatten the list and count occurrences
    neighbor_counts = {}
    for ap_pairs in shared_ap_list:
        for ap1, ap2 in ap_pairs:
            # Count in both directions to make the map comprehensive
            neighbor_counts.setdefault((ap1, ap2), 0)
            neighbor_counts[(ap1, ap2)] += 1
            neighbor_counts.setdefault((ap2, ap1), 0)
            neighbor_counts[(ap2, ap1)] += 1
    
    # Find the best neighbor for each AP based on the highest shared client count
    neighbor_map = {}
    for ap, _ in df.groupby('ap_name'):
        best_neighbor = None
        max_shared_clients = 0
        
        # Iterate through the neighbor_counts to find the best neighbor for the current AP
        for (ap_a, ap_b), count in neighbor_counts.items():
            if ap_a == ap and count > max_shared_clients:
                best_neighbor = ap_b
                max_shared_clients = count
        
        if best_neighbor:
            neighbor_map[ap] = best_neighbor
            
    return neighbor_map


# --- API Endpoint ---
@router.post("/ap-overload/analyze", response_model=APOverloadResponse)
async def analyze_ap_overload(request: APOverloadRequest):
    """
    Analyzes APs for client overcrowding and channel utilization, providing
    load balancing recommendations based on neighboring APs.
    """
    try:
        start_time = time.time()
        
        # Stage 1: Data Loading and Preparation
        # Use the centralized file resolver
        file_path = resolve_input_file_path(request.date, "ap_overload", AP_OVERLOAD_METRIC_FILE_MAP)
        df = _load_and_filter_data(file_path, request.site_name, request.date, request.start_time, request.end_time)
        ap_summary = _get_ap_summary(df)
        neighbor_map = _find_neighbor_aps(df)
        
        # Stage 2: Identify Overloaded APs
        overloaded_aps_df = ap_summary[
            (ap_summary['client_count'] >= request.max_clients_threshold) |
            (ap_summary['avg_ch_util'] >= request.max_utilization_threshold)
        ].sort_values(by=['avg_ch_util', 'client_count'], ascending=False)
        
        overloaded_metrics = [
            OverloadMetrics(
                ap_name=row['ap_name'], 
                client_count=int(row['client_count']), 
                avg_ch_util=float(f"{row['avg_ch_util']:.2f}")
            ) for _, row in overloaded_aps_df.iterrows()
        ]
        
        # Stage 3: Generate Recommendations
        recommendations = []
        if not overloaded_aps_df.empty:
            # Get a list of all currently overloaded APs for easy lookup
            overloaded_ap_names = set(overloaded_aps_df['ap_name'])

            for _, overloaded_ap in overloaded_aps_df.iterrows():
                neighbor_ap_name = neighbor_map.get(overloaded_ap['ap_name'])
                
                if neighbor_ap_name and neighbor_ap_name not in overloaded_ap_names:
                    # Neighbor is found and is not overloaded
                    neighbor_metrics_row = ap_summary[ap_summary['ap_name'] == neighbor_ap_name].iloc[0]
                    neighbor_metrics = OverloadMetrics(
                        ap_name=neighbor_ap_name, 
                        client_count=int(neighbor_metrics_row['client_count']),
                        avg_ch_util=float(f"{neighbor_metrics_row['avg_ch_util']:.2f}")
                    )
                    
                    recommendations.append(
                        LoadBalancingRecommendation(
                            overloaded_ap=overloaded_ap['ap_name'],
                            overloaded_metrics=OverloadMetrics(
                                ap_name=overloaded_ap['ap_name'],
                                client_count=int(overloaded_ap['client_count']),
                                avg_ch_util=float(f"{overloaded_ap['avg_ch_util']:.2f}")
                            ),
                            recommendation="Steer clients to a less-utilized neighboring AP.",
                            neighbor_ap=neighbor_ap_name,
                            neighbor_metrics=neighbor_metrics
                        )
                    )
                else:
                    # No suitable neighbor for a recommendation
                    recommendations.append(
                        LoadBalancingRecommendation(
                            overloaded_ap=overloaded_ap['ap_name'],
                            overloaded_metrics=OverloadMetrics(
                                ap_name=overloaded_ap['ap_name'],
                                client_count=int(overloaded_ap['client_count']),
                                avg_ch_util=float(f"{overloaded_ap['avg_ch_util']:.2f}")
                            ),
                            recommendation="No suitable neighbor found for load balancing.",
                            neighbor_ap="N/A",
                            neighbor_metrics=OverloadMetrics(ap_name="N/A", client_count=0, avg_ch_util=0.0)
                        )
                    )

        end_time = time.time()
        duration = end_time - start_time
        print(f"AP overload analysis for site '{request.site_name}' completed in {duration:.2f} seconds.")
        
        return APOverloadResponse(
            overloaded_aps=overloaded_metrics,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected server error occurred: {str(e)}"
        )