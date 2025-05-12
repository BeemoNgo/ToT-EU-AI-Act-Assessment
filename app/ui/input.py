import streamlit as st
from typing import Dict, Any, Optional

def render_api_key_input() -> str:
    """
    Render the OpenAI API key input form.
    
    Returns:
        str: The API key entered by the user.
    """
    # Check if API key is already in session state
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
        
    # Check if API key is in secrets
    try:
        if st.secrets["OPENAI_API_KEY"]:
            return st.secrets["OPENAI_API_KEY"]
    except:
        pass
    
    # Create a container for API key input
    with st.sidebar:
        st.subheader("OpenAI API Key")
        
        # Create form for API key
        with st.form(key="api_key_form"):
            api_key = st.text_input(
                "Enter your OpenAI API key",
                type="password",
                value=st.session_state.openai_api_key,
                help="Required to access OpenAI API for analysis"
            )
            
            submit_button = st.form_submit_button(label="Save API Key")
            
            if submit_button and api_key:
                st.session_state.openai_api_key = api_key
                st.success("API key saved!")
    
    return st.session_state.openai_api_key

def render_input_form() -> Optional[Dict[str, Any]]:
    with st.form(key="url_form"):
        st.subheader("Enter App Details")
        
        # URL input
        url = st.text_input(
            "Google Play Store URL",
            placeholder="https://play.google.com/store/apps/details?id=com.example.app",
            help="Enter the full URL of the app from the Google Play Store"
        )
        
        # Submit button
        submit_button = st.form_submit_button(label="Analyze App")
        
        if submit_button:
            if not url:
                st.error("Please enter a URL")
                return None
                
            if "play.google.com" not in url:
                st.error("Please enter a valid Google Play Store URL")
                return None
                
            return {"url": url.strip()}
            
    return None 