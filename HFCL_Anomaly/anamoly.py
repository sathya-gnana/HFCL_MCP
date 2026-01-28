#anamoly.py
import logging
import requests
from  mcp.server.fastmcp import FastMCP # type: ignore
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import List, Dict, Any, Optional, Literal, Union
import os
import datetime
import re

# MCP server init
mcp = FastMCP("network anomaly")

# --- Payload definition for StoredDataQueryPayload (existing) ---

# --- MCP Tools ---
API_BASE = "http://localhost:8989/api"  # Change to your FastAPI server host/port
OUTPUT_CHARTS_DIR = "generated_charts" # Directory to save generated HTML charts

# --- Payload definition for ChartPlottingRequest (existing) ---
class ChartPlottingRequest(BaseModel):
    """Request payload for plotting charts using direct SVG generation."""
    x_data: List[Union[float, str]] = Field(..., description="**REQUIRED.** Data points for the X-axis. Can be numerical (e.g., 1, 2, 3) for quantitative axes or strings (e.g., 'Jan', 'Feb') for categorical labels. Ensure its length matches all y_data_series.")
    y_data_series: List[List[float]] = Field(..., description="**REQUIRED.** A list where each inner list represents a data series for the Y-axis. All inner lists (series) must have the same length as x_data. Example: `[[10, 20, 15], [5, 12, 8]]` for two series.")
    
    chart_type: Literal["line", "bar", "scatter", "area", "stacked_bar", "horizontal_bar"] = Field(
        "line", 
        description="**OPTIONAL.** Type of chart to generate. "
                    "Choose from 'line', 'bar' (grouped bars), 'scatter', 'area', 'stacked_bar', or 'horizontal_bar'."
    )
    
    title: str = Field("Chart", description="**OPTIONAL.** Main title of the chart.")
    x_label: str = Field("X-axis", description="**OPTIONAL.** Label for the X-axis.")
    y_label: str = Field("Y-axis", description="**OPTIONAL.** Label for the Y-axis.")
    
    width: int = Field(800, description="**OPTIONAL.** Width of the SVG chart in pixels. Defaults to 800.")
    height: int = Field(500, description="**OPTIONAL.** Height of the SVG chart in pixels. Defaults to 500.")
    
    series_labels: Optional[List[str]] = Field(
        None, 
        description="**OPTIONAL.** Labels for each `y_data_series`, used for the legend. "
                    "Must match the number of `y_data_series` provided. Example: `['Series A', 'Series B']`."
    )
    marker_style: bool = Field(
        False, 
        description="**OPTIONAL.** For 'line' and 'scatter' charts, set to `true` to display markers on data points. Defaults to `false`."
    )
    line_width: int = Field(
        2, 
        description="**OPTIONAL.** For 'line' and 'area' charts, the width of the lines/borders. Defaults to 2."
    )
    fill_opacity: float = Field(
        0.3, ge=0.0, le=1.0, 
        description="**OPTIONAL.** For 'area' charts, the opacity of the fill color (0.0 to 1.0). Defaults to 0.3."
    )
    bar_gap_ratio: float = Field(
        0.2, ge=0.0, le=1.0, 
        description="**OPTIONAL.** For 'bar' charts, the ratio of the gap between bars to total bar width. Defaults to 0.2."
    )
    
    x_axis_is_numeric: bool = Field(
        False, 
        description="**OPTIONAL.** For 'line' and 'scatter' charts, set to `true` if `x_data` represents numerical values that should be scaled (e.g., timestamps, continuous measurements). "
                    "If `false` (default), `x_data` is treated as categorical labels evenly spaced on the axis."
    )

# --- Payload definition for PerformanceAnomalyRequest (existing) ---
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
        2.0, ge=0.0,
        description="Multiplier for standard deviation to set the anomaly threshold."
    )
    
    if_contamination: Optional[float] = Field(
        0.1, ge=0.0, le=0.5,
        description="Contamination parameter for Isolation Forest (used if model needs training)."
    )
    if_n_estimators: Optional[int] = Field(
        100, ge=1,
        description="Number of estimators (trees) for Isolation Forest (used if model needs training)."
    )

# --- Payload definition for InterferenceDetectionRequest (existing) ---
class InterferenceDetectionRequest(BaseModel):
    date: str = Field(..., description="Date for the dataset (e.g., '2025-05-15').")
    metric: str = Field("interference report", description="Metric for dataset resolution.")
    output_directory: str = Field("./reports/interference_detection/", description="Where to save CSV report.")
    contamination: float = Field(0.1, ge=0.0, le=0.5, description="Expected anomaly proportion for Isolation Forest.")
    n_estimators: int = Field(200, ge=50, description="Number of trees in Isolation Forest.")
    min_snr_threshold: float = Field(10.0, le=30.0, description="SNR threshold (dB) for boosting anomaly scores.")
    max_rssi_threshold: float = Field(-80.0, le=-30.0, description="RSSI threshold (dBm) for boosting anomaly scores.")
    boost_factor: float = Field(3.0, ge=1.0, description="Factor to amplify anomaly scores for critical cases.")

# --- Payload definition for ConnectionDropRequest (existing) ---
class ConnectionDropRequest(BaseModel):
    date: str = Field(..., description="Date for the dataset (e.g., '2025-05-15').")
    output_directory: str = Field("./reports/connection_drops/", description="Directory to save CSV report of connection drop anomalies.")
    min_connection_duration_threshold_seconds: Optional[int] = Field(30, description="Minimum duration in seconds for a connection to be considered stable (not a rapid drop).")
    assoc_threshold: int = Field(3, description="Number of associations within the window to flag as rapid connection anomaly.")
    assoc_window_seconds: int = Field(60, description="Time window in seconds to check for rapid associations.")

# --- Payload definition for ChannelOptimizationRequest (NEW) ---
class ChannelOptimizationRequest(BaseModel):
    date: str = Field(
        ...,
        description="Date for the dataset (e.g., 'May 15', '2025-05-15'). Used for dynamic file resolution."
    )
    metric: str = Field(
        "channel overlap report",
        description="Metric name to identify the dataset (e.g., 'channel overlap report')."
    )
    output_directory: str = Field(
        "./reports/channel_optimization/",
        description="Directory to save channel optimization reports."
    )
    contamination_level: float = Field(
        0.05, ge=0.0, le=0.5,
        description="Expected proportion of outliers in the data for Isolation Forest anomaly detection."
    )




# --------------------------
# MCP Tool
# --------------------------
class FilterCondition(BaseModel):
    operator: str = Field(..., description="Comparison operator (e.g., 'eq', 'gt', 'lt', Cesium.EntityGraphics, 'lte', 'between', 'in', 'contains', 'regex', 'is_null', 'is_not_null').")
    value: Any = Field(None, description="The value for the operator. For 'between', this is the lower bound. For 'in', this must be a list.")
    value2: Optional[Any] = Field(None, description="The second value for operators like 'between' (e.g., upper bound).")

class StoredDataQueryPayload(BaseModel):
    date_str: str = Field(..., description="Date for the report (e.g., '2025-05-15').")
    metric_name: Literal["client data"] = Field("client data", description="The specific metric/report to fetch.")
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
    having_filters: Optional[Dict[str, FilterCondition]] = Field(default_factory=dict, description="Filters to apply on aggregated results.")
    calculated_metrics: Optional[Dict[str, str]] = Field(default_factory=dict, description="Dictionary of new column names and their calculation formulas (e.g., {'total_data': 'data_uploaded_mb + data_downloaded_mb'}).")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class FilterCondition(BaseModel):
    """Defines a single filter condition for a column."""
    operator: str = Field(..., description="Comparison operator (e.g., 'eq', 'gt', 'lt', 'gte', 'lte', 'between', 'in', 'contains', 'regex', 'is_null', 'is_not_null').")
    value: Any = Field(None, description="The value for the operator. For 'between', this is the lower bound. For 'in', this must be a list.")
    value2: Optional[Any] = Field(None, description="The second value for operators like 'between' (e.g., upper bound).")

class StoredDataQueryPayload(BaseModel):
    """
    Payload for querying stored CSV data with features for advanced analysis.
    """
    date_str: str = Field(..., description="Date for the report (e.g., '2025-07-29').")
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

@mcp.tool()
def get_information(payload: StoredDataQueryPayload, timeout: int = 60) -> Dict[str, Any]:
    """
    Retrieve detailed client connection and performance information (e.g., SNR, RSSI, AP Name)
    from the 'ap clients 5 min daily' report for a specified date. This tool supports advanced
    filtering, aggregation, and top/bottom N analysis, leveraging a powerful backend for efficient
    data processing.

    Purpose:
    Ideal for investigating specific data points, applying filters, or performing aggregations
    such as identifying APs with low SNR, retrieving specific metrics, or calculating averages
    per group. The tool interacts with a FastAPI endpoint that handles data processing, caching,
    and file management for 'ap_clients_5_min_daily' reports.

    Example Usage:

    1. Find APs with low SNR (e.g., SNR < 15) and get top 5 APs:
    python
    payload = StoredDataQueryPayload(
        date_str="2025-07-29",
        metric_name="client data",
        filters={"snr": FilterCondition(operator="lt", value=15.0)},
        top_n={"ap_name": 5},
        return_type="aggregated_values"
    )
    result = get_information(payload)
    print(result)
    

    2. Get records for specific metrics with a time filter:
    python
    payload = StoredDataQueryPayload(
        date_str="2025-07-29",
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
    

    3. Perform aggregation (e.g., average RSSI and count per AP):
    python
    payload = StoredDataQueryPayload(
        date_str="2025-07-29",
        metric_name="client data",
        aggregate={"avg": ["rssi"], "count": ["*"]},
        group_by=["ap_name"],
        return_type="aggregated_values"
    )
    result = get_information(payload)
    print(result)
    

    4. Calculate a new metric (e.g., total bytes) and retrieve records:
    python
    payload = StoredDataQueryPayload(
        date_str="2025-07-29",
        metric_name="client data",
        metrics=["ap_name", "total_bytes"],
        calculated_metrics={"total_bytes": "rx_bytes + tx_bytes"},
        return_type="records"
    )
    result = get_information(payload)
    print(result)
    

    Args:
        payload (StoredDataQueryPayload): A structured object defining query parameters, including
            date, metrics, filters, aggregations, group_by, calculated_metrics, and more.
            See StoredDataQueryPayload for details.
        timeout (int, optional): Request timeout in seconds. Defaults to 15.

    Returns:
        Dict[str, Any]: A dictionary containing the queried data (records, count, or aggregated values),
            operation status, applied filters, and metadata or error messages.

    Raises:
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
    except requests.RequestException as e:
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
@mcp.tool()
def clear_information_cache() -> Dict[str, Any]:
    """
    **Purpose:** Clears the in-memory data cache of processed client data.

    Use this tool when:
    - You want to free up memory by removing cached DataFrames.
    - You want to reload updated CSV/Parquet files on the next query.
    - You are troubleshooting and need to ensure fresh data is always loaded.

    Returns:
        Dict[str, Any]: A dictionary containing the status of the cache clear
                        operation and any relevant messages.
    """
    try:
        url = f"{API_BASE}/clear_cache"  # Endpoint in your FastAPI router
        resp = requests.post(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        if e.response is not None:
            try:
                error_detail = e.response.json()
            except ValueError:
                error_detail = {"detail": e.response.text}
            return {
                "status": "error",
                "message": f"HTTP error {e.response.status_code}: {error_detail.get('detail', str(error_detail))}"
            }
        else:
            return {"status": "error", "message": f"Network error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}


@mcp.tool()
def plot_chart(payload: ChartPlottingRequest) -> str: # <--- Changed return type to str
    """
    **Purpose:** Generate a dynamic SVG chart (e.g., line, bar, scatter, area, stacked bar, horizontal bar)
    from provided data and return the raw HTML content directly.

    This tool sends your chart data to the FastAPI `/plot_chart` endpoint, which generates an SVG
    chart embedded within an HTML document. The raw HTML of that document is then returned directly by this tool.
    The client consuming this tool's output is responsible for rendering or processing this HTML.

    **Key Features:**
    - Supports various chart types to suit different data visualization needs.
    - Customizable titles, axis labels, dimensions, and series labels.
    - Handles both numerical and categorical X-axis data for flexible plotting.
    - Provides basic styling options for markers, line width, and fill opacity.

    **How to use:**
    Provide your X-axis data (`x_data`) and one or more Y-axis data series (`y_data_series`).
    Choose a `chart_type` that best represents your data. For multiple series, consider providing `series_labels` for a legend.
    Ensure that the lengths of `x_data` and all `y_data_series` are identical.

    Args:
        payload (ChartPlottingRequest): A structured object defining all the parameters for chart generation.
            Refer to the `ChartPlottingRequest` model for detailed descriptions of each field.

    Returns:
        str: The raw HTML content of the generated chart. In case of an error, this will be an
             error message string, not a structured dictionary, as the tool is designed to return
             the HTML on success. The MCP client should handle errors appropriately.
    """
    try:
        url = f"{API_BASE}/plot_chart"
        headers = {"Content-Type": "application/json"}
        
        # payload.dict() is for Pydantic v1. If Pydantic v2, use payload.model_dump()
        resp = requests.post(url, json=payload.dict(), headers=headers, timeout=60)
        resp.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        
        return resp.text # <--- Directly return the HTML content
    
    except requests.RequestException as e:
        # For direct string return, we'd typically just return the error message as a string
        # The MCP client would then need to parse this string to identify if it's an error.
        error_message = f"Error generating chart: Network or HTTP error: {str(e)}"
        if e.response is not None:
            try:
                error_detail = e.response.json()
                error_message += f" - Detail: {error_detail.get('detail', error_detail)}"
            except ValueError:
                error_message += f" - Response: {e.response.text[:200]}..." # Log part of the response if not JSON
        print(f"DEBUG: {error_message}") # Print to server logs for debugging
        return error_message
    
    except Exception as e:
        error_message = f"Error generating chart: An unexpected error occurred: {str(e)}"
        print(f"DEBUG: {error_message}") # Print to server logs for debugging
        return error_message

@mcp.tool()
def detect_performance_anomalies(payload: PerformanceAnomalyRequest) -> Dict[str, Any]:
    """
    Detects network performance anomalies by calling the FastAPI endpoint /performance/detect_anomalies.
    This tool allows specifying the date, metric, output directory for reports and plots,
    and parameters for anomaly detection and model training.

    Args:
        payload (PerformanceAnomalyRequest): The request payload containing parameters for anomaly detection.

    Returns:
        Dict[str, Any]: The response from the FastAPI endpoint, including details about anomalies found,
                        report paths, and model training status.
    """
    try:
        url = f"{API_BASE}/performance/detect_anomalies"
        headers = {"Content-Type": "application/json"}
        
        resp = requests.post(url, json=payload.dict(), headers=headers, timeout=120) # Increased timeout for potential model training
        resp.raise_for_status()
        
        return resp.json()
    except requests.RequestException as e:
        if e.response is not None:
            try:
                error_detail = e.response.json()
            except ValueError:
                error_detail = {"detail": e.response.text}
            return {"status": "error", "message": f"HTTP error {e.response.status_code}: {error_detail.get('detail', str(error_detail))}"}
        else:
            return {"status": "error", "message": f"Network error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}




@mcp.tool(
    name="detect_interference",
    description=(
        "Analyzes network data to detect client-level interference and provide AP-level "
        "channel change recommendations based on a simulated channel survey."
    ),
)
def detect_interference_tool(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP wrapper for the /interference/analyze API.
    Compulsory input: date
    Optional inputs: start_time, end_time, channel_utilization_threshold, limit
    """
    import requests
    
    url = f"{API_BASE}/interference/analyze"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

@mcp.tool()
def analyze_connection_drops(payload: ConnectionDropRequest) -> Dict[str, Any]:
    """
    Analyzes connection drop events from client data, specifically looking for
    'rapid association anomalies' where a client rapidly re-associates with an AP
    multiple times within a short window. This indicates potential instability.

    Args:
        payload (ConnectionDropRequest): The request payload containing parameters
                                         for connection drop analysis, including
                                         the mandatory date of the dataset, optional
                                         output directory, thresholds for rapid association
                                         detection (e.g., time window, minimum events), and
                                         optional filters for AP or client MAC.

    Returns:
        Dict[str, Any]: A dictionary containing the status, message, total disconnection
                        events, unique clients affected, a breakdown of disconnection
                        counts per client, a list of detected anomaly events, and the
                        path to the CSV report.

    Raises:
        ValidationError: If the payload is invalid (e.g., missing required fields or incorrect format).
        requests.RequestException: If the HTTP request fails due to network issues or server errors.
        Exception: For unexpected errors during processing.

    Notes:
        - Validates the payload before making the API call.
        - Logs key steps and errors for debugging purposes.
        - Increases timeout to 180 seconds to handle larger datasets.
        - Returns detailed error messages for better diagnostics.
    """
    try:
        # Validate payload
        logger.info("Validating payload for connection drop analysis")
        payload_dict = payload.dict(exclude_unset=True)
        if not payload_dict.get('date'):
            raise ValidationError("Date is a required field", ConnectionDropRequest)

        # Log payload details (excluding sensitive data)
        logger.info(f"Processing connection drop analysis for date: {payload_dict.get('date')}")

        # Make API call
        url = f"{API_BASE}/disconnection/analyze"
        headers = {"Content-Type": "application/json"}
        logger.debug(f"Sending request to {url} with payload: {payload_dict}")
        
        resp = requests.post(url, json=payload_dict, headers=headers, timeout=180)  # Increased timeout
        resp.raise_for_status()

        response_data = resp.json()
        logger.info("Successfully received response from API")
        return response_data

    except ValidationError as e:
        logger.error(f"Payload validation failed: {str(e)}")
        return {"status": "error", "message": f"Invalid payload: {str(e)}"}
    except requests.RequestException as e:
        logger.error(f"HTTP request failed: {str(e)}")
        if e.response is not None:
            try:
                error_detail = e.response.json()
            except ValueError:
                error_detail = {"detail": e.response.text}
            return {
                "status": "error",
                "message": f"HTTP error {e.response.status_code}: {error_detail.get('detail', str(error_detail))}"
            }
        return {"status": "error", "message": f"Network error: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error during connection drop analysis: {str(e)}")
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

@mcp.tool()
def analyze_channel_optimization(payload: ChannelOptimizationRequest) -> Dict[str, Any]:
    """
    Analyzes channel usage within Access Points to detect rule-based overlaps
    and identifies anomalous channel performance using an ML model (Isolation Forest).
    It processes data for a specified date and saves a detailed report.

    Args:
        payload (ChannelOptimizationRequest): The request payload containing parameters
                                              for channel optimization analysis, including
                                              the date of the dataset, output directory,
                                              and contamination level for the ML model.

    Returns:
        Dict[str, Any]: A dictionary containing the status, message, total rule-based overlaps,
                        total ML anomalies, a list of detailed channel issues, and the path to
                        the CSV report.
    """
    try:
        url = f"{API_BASE}/channel_optimization/analyze_channels"
        headers = {"Content-Type": "application/json"}

        resp = requests.post(url, json=payload.dict(), headers=headers, timeout=120)
        resp.raise_for_status()

        return resp.json()
    except requests.RequestException as e:
        if e.response is not None:
            try:
                error_detail = e.response.json()
            except ValueError:
                error_detail = {"detail": e.response.text}
            return {"status": "error", "message": f"HTTP error {e.response.status_code}: {error_detail.get('detail', str(error_detail))}"}
        else:
            return {"status": "error", "message": f"Network error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected ersror occurred: {str(e)}"}


if __name__ == "__main__":
    mcp.run(transport="stdio")