import logging
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Literal, Optional
import numpy as np
import os
import aiofiles
from io import StringIO
import requests
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Assuming app/utils/file_resolver.py contains the provided code
from app.utils.file_resolver import BASE_DATA_REPORTS_DIR, resolve_input_file_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API base URL
API_BASE = "http://localhost:8000"

# --- Pydantic Models for Request Payloads ---
class FilterCondition(BaseModel):
    """Defines a single filter condition for a column."""
    operator: str = Field(..., description="Comparison operator (e.g., 'eq', 'gt', 'lt', 'gte', 'lte', 'between', 'in', 'contains', 'regex', 'is_null', 'is_not_null').")
    value: Any = Field(None, description="The value for the operator. For 'between', this is the lower bound. For 'in', this must be a list.")
    value2: Optional[Any] = Field(None, description="The second value for operators like 'between' (e.g., upper bound).")

class StoredDataQueryPayload(BaseModel):
    """
    Payload for querying stored CSV data with features for advanced analysis.
    """
    date_str: str = Field(..., description="Date for the report (e.g., '2025-08-08').")
    metric_name: Literal["client data"] = Field(..., description="The specific metric/report to fetch. Must be 'client data'.")
    metrics: List[str] = Field(default_factory=list, description="List of metric names (column names) to retrieve. Required if 'return_type' is 'records'.")
    filters: Optional[Dict[str, FilterCondition]] = Field(default_factory=dict, description="Dictionary of filters to apply (column_name: FilterCondition).")
    aggregate: Optional[Dict[str, List[str]]] = Field(default_factory=dict, description="Dictionary of aggregations to perform (e.g., {'count': ['*'], 'avg': ['rssi']}).")
    top_n: Optional[Dict[str, int]] = Field(default_factory=dict, description="Dictionary of columns and the number of top values to retrieve (e.g., {'ap_name': 5}).")
    bottom_n: Optional[Dict[str, int]] = Field(default_factory=dict, description="Dictionary of columns and the number of bottom values to retrieve (e.g., {'rssi': 5}).")
    column_aliases: Optional[Dict[str, str]] = Field(default_factory=dict, description="Dictionary of column names to alias in the output (e.g., {'ap_name': 'Access Point'}).")
    return_type: Optional[Literal["records", "count_only", "aggregated_values"]] = Field("records", description="Type of data to return: 'records', 'count_only', 'aggregated_values'.")
    start_time: Optional[str] = Field(None, description="Start time for filtering (e.g., '08:00:00').")
    end_time: Optional[str] = Field(None, description="End time for filtering (e.g., '12:00:00').")
    sort_by: Optional[str] = Field(None, description="Column name to sort the results by.")
    sort_order: Optional[Literal["asc", "desc"]] = Field("asc", description="Sorting order: 'asc' for ascending, 'desc' for descending.")
    limit: Optional[int] = Field(None, description="Maximum number of records to return.")
    group_by: Optional[List[str]] = Field(default_factory=list, description="List of columns to group by before aggregation.")
    having_filters: Optional[Dict[str, FilterCondition]] = Field(default_factory=dict, description="Filters to apply on aggregated results (analogous to a 'HAVING' clause in SQL).")
    calculated_metrics: Optional[Dict[str, str]] = Field(default_factory=dict, description="Dictionary of new column names and their calculation formulas (e.g., {'total_data': 'rx_bytes + tx_bytes'}).")

# --- FastAPI Router Initialization ---
router = APIRouter()

# Ensure BASE_DATA_REPORTS_DIR exists
os.makedirs(BASE_DATA_REPORTS_DIR, exist_ok=True)

# --- METRIC_FILE_MAP for dynamic file resolution ---
METRIC_FILE_MAP = {
    "client data": {
        "type": "dynamic",
        "prefix": "ap_clients_5_min_daily"
    }
}

# --- Caching Mechanism ---
DATA_CACHE: Dict[str, pd.DataFrame] = {}

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names and handles missing/infinite values for a DataFrame.
    """
    df.rename(columns={
        'site_id': 'site_id', 'serial_no': 'serial_no', 'created_at': 'created_at',
        'created_time': 'created_time', 'client_mac': 'client_mac', 'host_name': 'host_name',
        'rx_bytes': 'rx_bytes', 'device_type': 'device_type', 'client_classification': 'client_classification',
        'op_mode': 'op_mode', 'snr': 'snr', 'authenticated': 'authenticated',
        'ip_address': 'ip_address', 'rx_packets': 'rx_packets', 'ssid': 'ssid',
        'associated': 'associated', 'bss_mac': 'bss_mac', 'up_time': 'up_time',
        'total_chutilization': 'total_chutilization', 'tx_bytes': 'tx_bytes', 'rssi': 'rssi',
        'ip6_address': 'ip6_address', 'device_os': 'device_os', 'ssid_index': 'ssid_index',
        'interface': 'interface', 'tx_packets': 'tx_packets', 'association_time': 'association_time',
        'ipv6_rx_bytes': 'ipv6_rx_bytes', 'ipv6_tx_bytes': 'ipv6_tx_bytes', 'ipv6_rx_packets': 'ipv6_rx_packets',
        'ipv6_tx_packets': 'ipv6_tx_packets', 'client_retry_count': 'client_retry_count',
        'tx_active_time': 'tx_active_time', 'rx_active_time': 'rx_active_time',
        'total_active_time': 'total_active_time', 'data_rate_from': 'data_rate_from',
        'data_rate_to': 'data_rate_to', 'data_rate': 'data_rate',
        'ipv4_tx_bytes': 'ipv4_tx_bytes', 'ipv4_rx_bytes': 'ipv4_rx_bytes',
        'ipv4_tx_packets': 'ipv4_tx_packets', 'ipv4_rx_packets': 'ipv4_rx_packets',
        'user_name': 'user_name', 'site_name': 'site_name', 'group_name': 'group_name',
        'ap_group_name': 'ap_group_name', 'sub_ap_group1': 'sub_ap_group1',
        'sub_ap_group2': 'sub_ap_group2', 'sub_ap_group3': 'sub_ap_group3',
        'sub_ap_group4': 'sub_ap_group4', 'sub_ap_group5': 'sub_ap_group5',
        'ap_description': 'ap_description', 'sap_id': 'sap_id', 'ap_name': 'ap_name',
        'ipv4': 'ipv4', 'ipv6': 'ipv6', 'channel': 'channel', 'ap_mac': 'ap_mac',
        'kicked_off': 'kicked_off', 'kick_off_time': 'kick_off_time',
        'group_location': 'group_location', 'group_description': 'group_description',
        'protocol': 'protocol', 'radio_id': 'radio_id', 'dissociation_time': 'dissociation_time',
        'search_index': 'search_index', 'reason_code': 'reason_code', 'vlan_id': 'vlan_id',
        'bssid': 'bssid', 'created_at_timestamp': 'created_at_timestamp',
        'tx_bytes_usage': 'tx_bytes_usage', 'rx_bytes_usage': 'rx_bytes_usage', 'ru26': 'ru26',
        'ru52': 'ru52', 'ru106': 'ru106', 'ru242': 'ru242', 'ru484': 'ru484',
        'ru996': 'ru996', 'tx_mcs48': 'tx_mcs48', 'tx_mcs24': 'tx_mcs24',
        'tx_mcs12': 'tx_mcs12', 'tx_mcs6': 'tx_mcs6', 'tx_mcs54': 'tx_mcs54',
        'tx_mcs36': 'tx_mcs36', 'tx_mcs18': 'tx_mcs18', 'tx_mcs9': 'tx_mcs9',
        'rx_mcs48': 'rx_mcs48', 'rx_mcs24': 'rx_mcs24', 'rx_mcs12': 'rx_mcs12',
        'rx_mcs6': 'rx_mcs6', 'rx_mcs54': 'rx_mcs54', 'rx_mcs36': 'rx_mcs36',
        'rx_mcs18': 'rx_mcs18', 'rx_mcs9': 'rx_mcs9'
    }, inplace=True)

    df.columns = [col.replace(' ', '_').replace('.', '').lower() for col in df.columns]

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            converted_col = pd.to_numeric(df[col], errors='coerce')
            if not converted_col.isnull().all():
                df[col] = converted_col

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(0, inplace=True)
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col].fillna(False, inplace=True)
            else:
                df[col].fillna('unknown', inplace=True)
    return df

def apply_filters(df: pd.DataFrame, filters: Optional[Dict[str, FilterCondition]], start_time: Optional[str], end_time: Optional[str]) -> pd.DataFrame:
    """Applies a list of filters and optional time-based filters to a DataFrame."""
    filtered_df = df.copy()

    if start_time or end_time:
        time_col = 'created_at_timestamp'
        if time_col not in filtered_df.columns:
            raise HTTPException(status_code=400, detail="Time-based filtering requires a valid timestamp column like 'created_at_timestamp'.")
        try:
            filtered_df[time_col] = pd.to_datetime(filtered_df[time_col])
        except Exception:
            raise HTTPException(status_code=400, detail=f"The '{time_col}' column could not be converted to datetime.")

        filtered_df = filtered_df.set_index(time_col)

        if start_time and end_time:
            filtered_df = filtered_df.between_time(start_time, end_time)
        elif start_time:
            filtered_df = filtered_df.between_time(start_time, '23:59:59')
        elif end_time:
            filtered_df = filtered_df.between_time('00:00:00', end_time)

        filtered_df = filtered_df.reset_index()

    if not filters:
        return filtered_df

    for column, filter_cond in filters.items():
        if column not in filtered_df.columns:
            raise HTTPException(status_code=400, detail=f"Filter column '{column}' not found. Available columns: {list(filtered_df.columns)}")

        col_series = filtered_df[column]

        if filter_cond.operator == "eq": filtered_df = filtered_df[col_series == filter_cond.value]
        elif filter_cond.operator == "ne": filtered_df = filtered_df[col_series != filter_cond.value]
        elif filter_cond.operator == "gt": filtered_df = filtered_df[col_series > filter_cond.value]
        elif filter_cond.operator == "lt": filtered_df = filtered_df[col_series < filter_cond.value]
        elif filter_cond.operator == "gte": filtered_df = filtered_df[col_series >= filter_cond.value]
        elif filter_cond.operator == "lte": filtered_df = filtered_df[col_series <= filter_cond.value]
        elif filter_cond.operator == "between":
            if not pd.api.types.is_numeric_dtype(col_series):
                raise HTTPException(status_code=400, detail=f"Cannot apply 'between' filter to non-numeric column '{column}'.")
            if filter_cond.value is None or filter_cond.value2 is None:
                raise HTTPException(status_code=400, detail=f"'between' operator requires 'value' and 'value2' for column '{column}'.")
            lower = min(filter_cond.value, filter_cond.value2)
            upper = max(filter_cond.value, filter_cond.value2)
            filtered_df = filtered_df[(col_series >= lower) & (col_series <= upper)]
        elif filter_cond.operator == "in":
            if not isinstance(filter_cond.value, list):
                raise HTTPException(status_code=400, detail=f"'in' operator requires 'value' to be a list for column '{column}'.")
            filtered_df = filtered_df[col_series.isin(filter_cond.value)]
        elif filter_cond.operator == "contains":
            str_series = col_series.astype(str).fillna('')
            filtered_df = filtered_df[str_series.str.contains(str(filter_cond.value), na=False)]
        elif filter_cond.operator == "regex":
            str_series = col_series.astype(str).fillna('')
            filtered_df = filtered_df[str_series.str.contains(str(filter_cond.value), regex=True, na=False)]
        elif filter_cond.operator == "is_null": filtered_df = filtered_df[col_series.isnull()]
        elif filter_cond.operator == "is_not_null": filtered_df = filtered_df[col_series.notnull()]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported filter operator: '{filter_cond.operator}'.")
    return filtered_df

def perform_aggregation(df: pd.DataFrame, aggregate: Optional[Dict[str, List[str]]], top_n: Optional[Dict[str, int]], bottom_n: Optional[Dict[str, int]]) -> Dict[str, Any]:
    """Performs aggregation on a DataFrame, now with top_n and bottom_n."""
    aggregated_results = {}
    print(aggregate)
    if aggregate:
        for agg_type, cols in aggregate.items():
            if agg_type == "count":
                if "*" in cols:
                    aggregated_results["total_filtered_count"] = len(df)
                else:
                    for col in cols:
                        if col in df.columns:
                            aggregated_results[f"count_{col}"] = df[col].count()
            elif agg_type in ["avg", "sum", "min", "max", "median", "std"]:
                for col in cols:
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        if agg_type == "avg":
                            result = df[col].mean()
                            print(df[col].mean())
                            print(col)
                            aggregated_results[f"avg_{col}"] = float(result) if pd.notnull(result) else None
                        if agg_type == "sum":
                            result = df[col].sum()
                            aggregated_results[f"sum_{col}"] = int(result) if pd.notnull(result) else None
                        if agg_type == "min":
                            result = df[col].min()
                            aggregated_results[f"min_{col}"] = int(result) if pd.notnull(result) else None
                        if agg_type == "max":
                            result = df[col].max()
                            aggregated_results[f"max_{col}"] = int(result) if pd.notnull(result) else None
                        if agg_type == "median":
                            result = df[col].median()
                            aggregated_results[f"median_{col}"] = float(result) if pd.notnull(result) else None
                        if agg_type == "std":
                            result = df[col].std()
                            aggregated_results[f"std_{col}"] = float(result) if pd.notnull(result) else None
                    elif col in df.columns:
                        logger.warning(f"Cannot calculate {agg_type} for non-numeric column '{col}'. Skipping.")
            else:
                logger.warning(f"Unsupported aggregation type: '{agg_type}'. Skipping.")

    if top_n:
        for column, n in top_n.items():
            if column in df.columns:
                top_results = df[column].value_counts().head(n)
                aggregated_results[f"top_{n}_{column}"] = top_results.to_dict()
            else:
                logger.warning(f"Top N requested for non-existent column '{column}'. Skipping.")

    if bottom_n:
        for column, n in bottom_n.items():
            if column in df.columns:
                bottom_results = df[column].value_counts().tail(n)
                aggregated_results[f"bottom_{n}_{column}"] = bottom_results.to_dict()
            else:
                logger.warning(f"Bottom N requested for non-existent column '{column}'. Skipping.")

    return aggregated_results

def _process_query_results(
    filtered_df: pd.DataFrame, 
    payload: StoredDataQueryPayload
) -> Dict[str, Any]:
    """Internal helper to handle the common logic for returning query results."""
    return_type = payload.return_type
    metrics = payload.metrics
    filters = payload.filters
    aggregate = payload.aggregate
    top_n = payload.top_n
    bottom_n = payload.bottom_n
    column_aliases = payload.column_aliases
    sort_by = payload.sort_by
    sort_order = payload.sort_order
    limit = payload.limit
    group_by = payload.group_by
    having_filters = payload.having_filters
    
    if return_type == "count_only":
        return {
            "status": "success",
            "filters_applied": filters,
            "total_count_after_filters": len(filtered_df)
        }
    elif return_type == "aggregated_values":
        if not (aggregate or top_n or bottom_n):
            raise HTTPException(status_code=400, detail="The 'aggregate', 'top_n', or 'bottom_n' field is required when 'return_type' is 'aggregated_values'.")
        
        agg_df = filtered_df.copy()
        if group_by:
            if not all(col in agg_df.columns for col in group_by):
                non_existent = [col for col in group_by if col not in agg_df.columns]
                raise HTTPException(status_code=400, detail=f"Grouping columns not found: {non_existent}")
            
            # Map user-provided aggregation types to pandas functions
            agg_map = {
                "avg": "mean",
                "sum": "sum",
                "min": "min",
                "max": "max",
                "median": "median",
                "std": "std",
                "count": "count",
                "nunique": "nunique"  # Added support for unique counts
            }
            
            agg_dict = {}
            for agg_type, cols in aggregate.items():
                if agg_type not in agg_map:
                    logger.warning(f"Unsupported aggregation type: '{agg_type}'. Skipping.")
                    continue
                pandas_agg = agg_map[agg_type]
                for col in cols:
                    if col == '*':
                        agg_dict['count'] = 'count'
                    else:
                        if col in agg_df.columns:
                            agg_dict[col] = pandas_agg
                        else:
                            logger.warning(f"Column '{col}' not found for aggregation '{agg_type}'. Skipping.")
            
            if not agg_dict:
                raise HTTPException(status_code=400, detail="No valid aggregations provided for the grouped data.")

            try:
                grouped_data = agg_df.groupby(group_by).agg(agg_dict).reset_index()
                # Rename columns to match expected output format
                renamed_columns = {col: f"{agg_dict[col]}_{col}" if col != 'count' else 'total_count' for col in agg_dict}
                grouped_data.rename(columns=renamed_columns, inplace=True)
                
                # Apply sorting if sort_by and sort_order are provided
                if sort_by:
                    if sort_by not in grouped_data.columns:
                        # Handle case where sort_by is 'count' but column is renamed
                        if sort_by == 'count' and any(col.startswith('count_') or col == 'total_count' for col in grouped_data.columns):
                            sort_by = next(col for col in grouped_data.columns if col.startswith('count_') or col == 'total_count')
                        else:
                            raise HTTPException(status_code=400, detail=f"Sort column '{sort_by}' not found in aggregated results.")
                    ascending = sort_order == "asc"
                    grouped_data = grouped_data.sort_values(by=sort_by, ascending=ascending)
                
                # Apply limit if provided
                if limit is not None:
                    grouped_data = grouped_data.head(limit)
                
                aggregated_results = grouped_data.to_dict(orient="records")
            except Exception as e:
                logger.error(f"Error during groupby aggregation: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to perform aggregation: {str(e)}")
        else:
            aggregated_results = perform_aggregation(agg_df, aggregate, top_n, bottom_n)

        if not aggregated_results:
            raise HTTPException(status_code=400, detail="No valid aggregations could be performed with the provided payload. Check column names and types.")
        
        if having_filters:
            if isinstance(aggregated_results, dict):
                temp_agg_df = pd.DataFrame([aggregated_results])
            else:
                temp_agg_df = pd.DataFrame(aggregated_results)

            filtered_agg_df = apply_filters(temp_agg_df, having_filters, None, None)
            
            if filtered_agg_df.empty:
                return {
                    "status": "success",
                    "filters_applied": filters,
                    "aggregations_performed": {**aggregate, **{"top_n": top_n, "bottom_n": bottom_n}},
                    "having_filters_applied": having_filters,
                    "data": []
                }
            aggregated_results = filtered_agg_df.to_dict(orient="records")

        return {
            "status": "success",
            "filters_applied": filters,
            "aggregations_performed": {**aggregate, **{"top_n": top_n, "bottom_n": bottom_n}},
            "group_by": group_by,
            "having_filters_applied": having_filters,
            "data": aggregated_results
        }
    else:
        if not metrics:
            raise HTTPException(
                status_code=400,
                detail="The 'metrics' list cannot be empty when 'return_type' is 'records'. Please provide a list of columns to return."
            )

        if payload.calculated_metrics:
            for new_col, formula in payload.calculated_metrics.items():
                if new_col in filtered_df.columns:
                    logger.warning(f"Calculated metric '{new_col}' overwrites an existing column.")
                try:
                    filtered_df[new_col] = filtered_df.eval(formula, engine='python')
                    if new_col not in metrics:
                        metrics.append(new_col)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid formula for calculated metric '{new_col}': {str(e)}")

        existing_metrics = [m for m in metrics if m in filtered_df.columns]
        non_existent_metrics = [m for m in metrics if m not in existing_metrics]
        
        if not existing_metrics:
            raise HTTPException(
                status_code=400,
                detail=f"None of the requested metrics were found in the dataset. Non-existent: {non_existent_metrics}"
            )
        
        selected_data = filtered_df[existing_metrics]
        
        if column_aliases:
            selected_data = selected_data.rename(columns=column_aliases)

        if sort_by:
            if sort_by not in selected_data.columns:
                raise HTTPException(status_code=400, detail=f"Sort column '{sort_by}' not found in the selected metrics.")
            ascending = sort_order == "asc"
            selected_data = selected_data.sort_values(by=sort_by, ascending=ascending)
        
        if limit is not None:
            selected_data = selected_data.head(limit)
        
        return {
            "status": "success",
            "requested_metrics": metrics,
            "returned_metrics_count": len(existing_metrics),
            "total_records": len(selected_data),
            "non_existent_metrics": non_existent_metrics,
            "filters_applied": filters,
            "calculated_metrics_applied": payload.calculated_metrics,
            "data": selected_data.to_dict(orient="records")
        }

@router.post("/get_info")
async def query_ap_clients_by_date(
    payload: StoredDataQueryPayload
):
    """
    Fetches client data for a given date (from payload), and then applies
    filters and aggregations, including new features.
    """
    try:
        if payload.metric_name not in METRIC_FILE_MAP:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported metric_name: '{payload.metric_name}'"
            )

        # Check if data is already in cache
        cache_key = f"{payload.date_str}_{payload.metric_name}"
        if cache_key in DATA_CACHE:
            df = DATA_CACHE[cache_key].copy()
        else:
            parquet_file_path = resolve_input_file_path(payload.date_str, payload.metric_name, METRIC_FILE_MAP).replace('.csv', '.parquet')
            
            if os.path.exists(parquet_file_path):
                df = pd.read_parquet(parquet_file_path)
            else:
                csv_file_path = resolve_input_file_path(payload.date_str, payload.metric_name, METRIC_FILE_MAP)
                if not os.path.exists(csv_file_path):
                    raise HTTPException(status_code=404, detail=f"File not found at '{csv_file_path}'.")
                
                dtype_dict = {
                    'ip_address': str, 'ip6_address': str, 'ipv4': str, 'ipv6': str,
                    'kick_off_time': str, 'dissociation_time': str, 'created_at': str,
                    'created_time': str, 'association_time': str, 'created_at_timestamp': str
                }
                
                async with aiofiles.open(csv_file_path, mode='r') as f:
                    content = await f.read()

                df = pd.read_csv(StringIO(content), dtype=dtype_dict, low_memory=False)
                df = standardize_dataframe(df)
                df.to_parquet(parquet_file_path, index=False)
                logger.info(f"Successfully converted and saved {csv_file_path} to Parquet at {parquet_file_path}")

            DATA_CACHE[cache_key] = df.copy()

        filtered_df = apply_filters(df, payload.filters, payload.start_time, payload.end_time)
        return _process_query_results(filtered_df, payload)

    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail={"message": "Invalid payload format.", "errors": e.errors()}
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@router.post("/clear_cache")
async def clear_data_cache():
    """
    Clears the in-memory data cache.
    """
    global DATA_CACHE
    DATA_CACHE.clear()
    return {"status": "success", "message": "In-memory cache has been cleared."}

# --- MCP Tool Definition ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RequestException,)),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying request (attempt {retry_state.attempt_number}) due to {retry_state.outcome.exception()}"
    )
)
def get_information(payload: StoredDataQueryPayload, timeout: int = 15) -> Dict[str, Any]:
    """
    Retrieve detailed client connection and performance information (e.g., SNR, RSSI, AP Name)
    from the 'ap clients 5 min daily' report for a specified date. This tool supports advanced
    filtering, aggregation, and top/bottom N analysis, leveraging a powerful backend for efficient
    data processing.

    **Purpose:**
    Ideal for investigating specific data points, applying filters, or performing aggregations
    such as identifying APs with low SNR, retrieving specific metrics, or calculating averages
    per group. The tool interacts with a FastAPI endpoint that handles data processing, caching,
    and file management for 'ap_clients_5_min_daily' reports.

    **Example Usage:**

    1. **Find APs with low SNR (e.g., SNR < 15) and get top 5 APs:**
    ```python
    payload = StoredDataQueryPayload(
        date_str="2025-08-08",
        metric_name="client data",
        filters={"snr": FilterCondition(operator="lt", value=15.0)},
        top_n={"ap_name": 5},
        return_type="aggregated_values"
    )
    result = get_information(payload)
    print(result)
    ```

    2. **Get records for specific metrics with a time filter:**
    ```python
    payload = StoredDataQueryPayload(
        date_str="2025-08-08",
        metric_name="client data",
        metrics=["ap_name", "snr", "rssi"],
        start_time="08:00:00",
        end_time="12:00:00",
        return_type="records",
        sort_by="snr",
        sort_order="asc",
        limit=100
    )
    result = get_information(payload)
    print(result)
    ```

    3. **Perform aggregation (e.g., average RSSI and count per AP):**
    ```python
    payload = StoredDataQueryPayload(
        date_str="2025-08-08",
        metric_name="client data",
        aggregate={"avg": ["rssi"], "count": ["*"]},
        group_by=["ap_name"],
        return_type="aggregated_values"
    )
    result = get_information(payload)
    print(result)
    ```

    4. **Calculate a new metric (e.g., total bytes) and retrieve records:**
    ```python
    payload = StoredDataQueryPayload(
        date_str="2025-08-08",
        metric_name="client data",
        metrics=["ap_name", "total_bytes"],
        calculated_metrics={"total_bytes": "rx_bytes + tx_bytes"},
        return_type="records"
    )
    result = get_information(payload)
    print(result)
    ```

    **Args:**
        payload (StoredDataQueryPayload): A structured object defining query parameters, including
            date, metrics, filters, aggregations, group_by, calculated_metrics, and more.
            See `StoredDataQueryPayload` for details.
        timeout (int, optional): Request timeout in seconds. Defaults to 15.

    **Returns:**
        Dict[str, Any]: A dictionary containing the queried data (records, count, or aggregated values),
            operation status, applied filters, and metadata or error messages.

    **Raises:**
        ValidationError: If the payload is invalid (e.g., incorrect format or missing required fields).
        RequestException: If the HTTP request fails after retries (e.g., network issues or server errors).
    """
    try:
        url = f"{API_BASE}/get_info"
        logger.info(f"Sending request to {url} for date {payload.date_str} with metric {payload.metric_name}")
        resp = requests.post(url, json=payload.dict(), timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
        logger.info(f"Successfully retrieved data for date {payload.date_str}")
        return result
    except RequestException as e:
        if e.response is not None:
            try:
                error_detail = e.response.json()
            except ValueError:
                error_detail = {"detail": e.response.text}
            error_message = f"HTTP error {e.response.status_code}: {error_detail.get('detail', str(error_detail))}"
            logger.error(error_message)
            return {"status": "error", "message": error_message}
        else:
            error_message = f"Network error: {str(e)}"
            logger.error(error_message)
            return {"status": "error", "message": error_message}
    except ValidationError as e:
        error_message = f"Invalid payload format: {e.errors()}"
        logger.error(error_message)
        return {"status": "error", "message": error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        logger.error(error_message)
        return {"status": "error", "message": error_message}