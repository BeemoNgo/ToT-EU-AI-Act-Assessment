"""
Main Streamlit application entry point.
"""
import streamlit as st
import pandas as pd
import traceback
import os
from typing import Optional

from models.app_data import AppDetails, AnalysisResults
from services.analysis import AnalysisService
from services.playstore import PlayStoreService
from services.logger import StatusLogger
from ui.analysis import display_analysis_results, display_app_details_table, display_footer
from ui.input import render_api_key_input
from ui.search import search_input, debounced_search, display_app_list

def load_eu_ai_act_prompts() -> pd.DataFrame:
    """Load EU AI Act assessment questions from CSV."""
    try:
        path = os.path.join("asset", "EU_AI_Act_Assessment_Questions.csv")
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load EU AI Act prompts: {str(e)}")
        return pd.DataFrame(columns=["Type", "Prompt"])

def main():
    st.set_page_config(
        page_title="App Privacy & AI Act Analysis Tool",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    .st-emotion-cache-16idsys p {
        font-size: 14px;
    }
    .app-footer {
        margin-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("🔍 App Privacy & EU AI Act Analysis Tool")
    st.markdown("""
    Analyze mobile apps for privacy practices, user feedback, and EU AI Act compliance.
    
    👉 Search for an app by name to get started.
    """)

    # Load EU AI Act prompts
    eu_ai_act_prompts = load_eu_ai_act_prompts()
    
    # Get OpenAI API key
    api_key = render_api_key_input()
    
    if not api_key:
        st.warning("Please enter your OpenAI API key to use this app.")
        display_footer()
        return
    
    # App search functionality
    query = search_input()
    app_results = debounced_search(query)
    selected_app_title, app_id = display_app_list(app_results)
    
    # If an app is selected, proceed with analysis
    if app_id:
        # Prepare services
        try:
            analysis_service = AnalysisService(api_key=api_key)
            playstore_service = PlayStoreService()
            
            # Setup status logger
            status_logger = StatusLogger(st.status(f"Starting analysis for {selected_app_title}..."))
            
            # Get app info
            status_logger.update(label=f"Fetching app details for {app_id}...")
            app_details = playstore_service.get_app_details(app_id=app_id, app_name=selected_app_title)
            
            if app_details:
                # Display app details
                display_app_details_table(app_details)
                
                # Get reviews
                status_logger.update(label=f"Fetching reviews for {app_details.name}...")
                reviews_df = playstore_service.get_app_reviews(app_id=app_id, app_name=selected_app_title)
                
                # Analyze app
                analysis_results = analysis_service.analyze_app(app_details, reviews_df, status_logger)
                
                # Perform EU AI Act classification
                if not analysis_results.has_error():
                    status_logger.update(label=f"Performing EU AI Act classification for {app_details.name}...")
                    analysis_results.eu_ai_act_classification = analysis_service.perform_eu_ai_act_classification(
                        app_details, 
                        analysis_results.user_review_analysis,
                        analysis_results.difference_analysis,
                        eu_ai_act_prompts,
                        status_logger
                    )
                
                # Display analysis results
                display_analysis_results(analysis_results, app_details.name)
                
                # Complete status
                status_logger.update(label="Analysis completed!", state="complete")
            else:
                st.error("Failed to fetch app details. Please try another app.")
            
        except Exception as e:
            traceback_details = traceback.format_exc()
            st.error(f"An error occurred: {str(e)}")
            with st.expander("Error Details", expanded=False):
                st.code(traceback_details)
    
    # Display footer
    display_footer()

if __name__ == "__main__":
    main() 