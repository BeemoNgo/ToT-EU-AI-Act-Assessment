"""
UI components for status and logging display.
"""
import streamlit as st
from typing import Optional, Callable

from services.logger import StatusLogger

def create_status_box(label: str, expanded: bool = True) -> tuple:
    """
    Create a status box with a StatusLogger instance.
    
    Args:
        label: Initial label for the status box
        expanded: Whether the status box is expanded
        
    Returns:
        Tuple of (status object, StatusLogger instance)
    """
    status = st.status(label, expanded=expanded)
    status_logger = StatusLogger(status)
    return status, status_logger

def show_analysis_log(app_name: str, analysis_callback: Callable, *args, **kwargs) -> Optional[dict]:
    """
    Display a status box for analysis and run the analysis callback.
    
    Args:
        app_name: Name of the app being analyzed
        analysis_callback: Function to run for analysis
        *args: Arguments to pass to the analysis callback
        **kwargs: Keyword arguments to pass to the analysis callback
        
    Returns:
        Result of the analysis callback, or None if an error occurred
    """
    st.subheader(f"Analysis Log for: {app_name}")
    
    # Create status box
    with st.status(f"Starting analysis for {app_name}...", expanded=True) as status:
        # Create status logger
        log = StatusLogger(status)
        
        try:
            # Add status_logger to kwargs
            kwargs['status_logger'] = log
            
            # Run the analysis callback
            result = analysis_callback(*args, **kwargs)
            
            # Mark as complete if the analysis was successful
            if result is not None and not getattr(result, 'error', False):
                status.update(label=f"Analysis complete for {app_name}!", state="complete", expanded=False)
            elif getattr(result, 'error', False):
                status.update(label=f"Analysis halted for {app_name}: {result.error}", state="warning", expanded=False)
            
            return result
            
        except Exception as e:
            # Log the error
            log.error(f"Critical error during analysis for {app_name}: {e}", exc_info=True)
            
            # Update the status
            status.update(label=f"Analysis failed for {app_name}: {e}", state="error")
            
            return None 