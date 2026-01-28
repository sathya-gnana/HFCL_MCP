# app/api/disconnection_router.py
import pandas as pd
from datetime import datetime
import io
import os
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import re # Import the regular expression library

from app.utils.file_resolver import resolve_input_file_path

router = APIRouter()

# --- Metric to File Mapping ---
DISCONNECTION_METRIC_FILE_MAP = {
    "disconnections": {"type": "dynamic", "prefix": "ap_clients_5_min_daily"},
}

# --- Pydantic Models ---
class ClientDisconnectionDetails(BaseModel):
    client_mac: str
    disconnection_count: int
    kicked_off_count: int
    most_frequent_reason: str
    average_uptime_minutes: Optional[float]

class DisconnectionAnalysisResponseItem(BaseModel):
    ap_name: str
    ap_mac: str
    total_disconnections: int
    kicked_off_count: int
    average_uptime_minutes: float
    most_frequent_reason: str
    clients_with_most_disconnections: List[ClientDisconnectionDetails]
    recommendation: str

class DisconnectionAnalysisRequest(BaseModel):
    date: str = Field(
        ...,
        description="Date of the dataset (e.g., '08-08-2025'). Used for time-based filtering and file resolution."
    )
    start_datetime: Optional[str] = Field(
        None,
        description="Optional start datetime for the analysis period (e.g., '2025-08-08 10:00:00')."
    )
    end_datetime: Optional[str] = Field(
        None,
        description="Optional end datetime for the analysis period (e.g., '2025-08-08 12:00:00')."
    )
    ap_name: Optional[str] = Field(
        None,
        description="Optional filter for a specific Access Point name."
    )
    client_mac: Optional[str] = Field(
        None,
        description="Optional filter for a specific client MAC address."
    )
    limit: int = Field(
        10, ge=1, le=20,
        description="The maximum number of top APs to return. Defaults to 10, with a maximum limit of 20."
    )

class DisconnectionAnalysisResponse(BaseModel):
    status: str = "success"
    message: str = "Disconnection analysis completed."
    top_disconnection_events: List[DisconnectionAnalysisResponseItem]

# --- Helper Functions ---
def _load_and_filter_data(file_path: str, request: DisconnectionAnalysisRequest) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"The data file for {request.date} was not found."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV: {e}"
        )
    
    # Apply optional filters
    if request.ap_name:
        df = df[df['ap_name'] == request.ap_name]
    if request.client_mac:
        df = df[df['client_mac'] == request.client_mac]

    if df.empty:
        return df

    # Convert datetime columns and filter
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['association_time'] = pd.to_datetime(df['association_time'], errors='coerce')
    df['dissociation_time'] = pd.to_datetime(df['dissociation_time'], errors='coerce')

    # Filter by time range if provided
    if request.start_datetime and request.end_datetime:
        try:
            start_dt = pd.to_datetime(request.start_datetime, errors='raise')
            end_dt = pd.to_datetime(request.end_datetime, errors='raise')
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid date or time format. Please use 'YYYY-MM-DD HH:MM:SS'. Error: {e}"
            )
        df = df[(df['created_at'] >= start_dt) & (df['created_at'] <= end_dt)]

    return df

def _get_reason_code_text(code: Any) -> str:
    # A simplified mapping for clarity
    reasons = {
        1: "Unspecified reason",
        2: "Previous authentication no longer valid",
        3: "Sending station is leaving",
        4: "Disassociated due to inactivity",
        5: "AP is unable to handle all currently associated stations",
        8: "Sending station is leaving (deauth)",
        14: "Message integrity code (MIC) failure",
        15: "4-Way Handshake timeout",
        23: "IEEE 802.1X authentication failed"
    }
    return reasons.get(int(code), "Unknown reason code") if pd.notna(code) else "No reason code"

def _generate_recommendation(ap_mac: str, stats: pd.Series) -> str:
    if stats['kicked_off_count'] > stats['total_disconnections'] * 0.5:
        return "High proportion of 'kicked-off' events. Investigate AP configuration and channel utilization."
    
    most_common_reason = _get_reason_code_text(stats['most_frequent_reason'])
    if "inactivity" in most_common_reason.lower() and stats['average_uptime_minutes'] < 10:
        return "High disconnections with short uptime, likely due to inactivity. Check signal coverage and AP placement."
    
    if "AP is unable to handle" in most_common_reason:
        return "AP is likely overloaded. Consider load balancing or adding more APs to this area."
        
    return "Disconnection patterns appear normal. No specific recommendation at this time."

# New helper function to convert HH:MM:SS to minutes
def _convert_uptime_to_minutes(uptime_str: str) -> float:
    try:
        # Handle "days, HH:MM:SS" format
        if 'd' in uptime_str:
            days_part, time_part = uptime_str.split('d')
            days = int(days_part)
            h, m, s = map(int, time_part.strip().split(':'))
            total_minutes = (days * 24 * 60) + (h * 60) + m + (s / 60)
            return total_minutes
        
        # Handle "HH:MM:SS" or "MM:SS" format
        parts = uptime_str.split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            total_minutes = (h * 60) + m + (s / 60)
            return total_minutes
        elif len(parts) == 2:
            m, s = map(int, parts)
            total_minutes = m + (s / 60)
            return total_minutes
        else:
            return 0.0 # Return 0 for unrecognized format
    except (ValueError, IndexError):
        return 0.0 # Return 0 for invalid data

# --- API Endpoint ---
@router.post("/disconnection/analyze", response_model=DisconnectionAnalysisResponse)
async def analyze_disconnections(request: DisconnectionAnalysisRequest):
    try:
        # Stage 1: File Resolution and Data Loading
        file_path = resolve_input_file_path(request.date, "disconnections", DISCONNECTION_METRIC_FILE_MAP)
        df = _load_and_filter_data(file_path, request)

        if df.empty:
            return DisconnectionAnalysisResponse(
                message="No disconnection events found for the specified date and filters.",
                top_disconnection_events=[]
            )

        # Stage 2: Calculate and Filter Uptime
        # Use the provided 'up_time' column as uptime
        df['up_time_minutes'] = df['up_time'].apply(_convert_uptime_to_minutes)
        
        # Filter for actual disconnections (non-roaming events) AND short uptime (< 5 minutes)
        # Using a fixed threshold of 5 minutes as requested
        disconnection_events = df[
            (df['dissociation_time'].notna()) & 
            (df['up_time_minutes'] < 5)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        if disconnection_events.empty:
            return DisconnectionAnalysisResponse(
                message="No disconnection events with uptime less than 5 minutes found for the specified date and filters.",
                top_disconnection_events=[]
            )

        # Stage 3: Group and Aggregate by AP
        grouped_by_ap = disconnection_events.groupby(['ap_name', 'ap_mac']).agg(
            total_disconnections=('dissociation_time', 'count'),
            kicked_off_count=('kicked_off', 'sum'),
            average_uptime_minutes=('up_time_minutes', 'mean'),
            most_frequent_reason=('reason_code', lambda x: x.mode()[0] if not x.mode().empty else None)
        ).reset_index().sort_values(by='total_disconnections', ascending=False)
        
        # Stage 4: Gather client details for each top AP
        response_items = []
        for _, ap_row in grouped_by_ap.head(request.limit).iterrows():
            ap_name = ap_row['ap_name']
            ap_mac = ap_row['ap_mac']

            ap_clients_df = disconnection_events[disconnection_events['ap_mac'] == ap_mac]
            client_details = ap_clients_df.groupby('client_mac').agg(
                disconnection_count=('dissociation_time', 'count'),
                kicked_off_count=('kicked_off', 'sum'),
                most_frequent_reason=('reason_code', lambda x: x.mode()[0] if not x.mode().empty else None),
                average_uptime_minutes=('up_time_minutes', 'mean')
            ).reset_index().sort_values(by='disconnection_count', ascending=False).head(3)

            clients_list = [
                ClientDisconnectionDetails(
                    client_mac=row['client_mac'],
                    disconnection_count=int(row['disconnection_count']),
                    kicked_off_count=int(row['kicked_off_count']),
                    most_frequent_reason=_get_reason_code_text(row['most_frequent_reason']),
                    average_uptime_minutes=round(row['average_uptime_minutes'], 2) if pd.notna(row['average_uptime_minutes']) else None
                ) for _, row in client_details.iterrows()
            ]

            recommendation = _generate_recommendation(ap_mac, ap_row)

            response_items.append(
                DisconnectionAnalysisResponseItem(
                    ap_name=ap_name,
                    ap_mac=ap_mac,
                    total_disconnections=int(ap_row['total_disconnections']),
                    kicked_off_count=int(ap_row['kicked_off_count']),
                    average_uptime_minutes=round(ap_row['average_uptime_minutes'], 2),
                    most_frequent_reason=_get_reason_code_text(ap_row['most_frequent_reason']),
                    clients_with_most_disconnections=clients_list,
                    recommendation=recommendation
                )
            )

        return DisconnectionAnalysisResponse(top_disconnection_events=response_items)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )