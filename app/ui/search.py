"""
UI components for app search.
"""
import streamlit as st
from typing import List, Dict, Tuple, Optional
import time

from services.playstore import PlayStoreService
from services.logger import logger

def search_input(value: str = "") -> str:
    # Initialize search debounce time in session state if not exists
    if 'last_search_time' not in st.session_state:
        st.session_state.last_search_time = 0
    
    # Get input from user
    query = st.text_input(
        "Search for an app on Google Play:", 
        placeholder="e.g., ChatGPT, Gemini, Midjourney",
        value=value,
        key="search_query"
    )
    
    # Return the query for further processing
    return query

def debounced_search(query: str, debounce_time: float = 0.8) -> List[Dict[str, str]]:
    """
    Execute a search with debounce to prevent excessive API calls.
    
    Args:
        query: Search query string
        debounce_time: Time in seconds to wait before executing another search
    
    Returns:
        List of app dictionaries (title, appId, developer)
    """
    # Initialize search results in session state if not exists
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    
    # Skip empty queries
    if not query:
        return []
    
    # Get current time
    current_time = time.time()
    
    # Check if enough time has passed since last search
    if current_time - st.session_state.last_search_time < debounce_time:
        return st.session_state.search_results
    
    # Update last search time
    st.session_state.last_search_time = current_time
    
    # Make sure PlayStoreService is initialized
    if 'playstore_service' not in st.session_state:
        st.session_state.playstore_service = PlayStoreService()
    
    # Perform the search
    results = perform_search(query, st.session_state.playstore_service)
    
    # Store results in session state
    st.session_state.search_results = results
    
    return results

def display_app_list(app_results: List[Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Displays a list of apps (title and developer) as buttons and returns the selected app.

    Args:
        app_results: List of app dictionaries containing title, appId, and developer.

    Returns:
        A tuple containing the selected app title and ID, or (None, None) if no app is selected.
    """
    if not app_results:
        return None, None

    st.subheader("Search Results")
    st.write("**Select an app to analyze:**")
    selected_app_title = None
    selected_app_id = None

    # Create a container with fixed height and scroll
    with st.container(height=400):  # Set fixed height of 400px
        # Display as a vertical list
        for app_data in app_results:
            title = app_data['title']
            app_id = app_data['appId']
            developer = app_data.get('developer', 'N/A') # Use .get for safety
            
            # Create a button label showing Title and Developer
            button_label = f"**{title}** by {developer}"
            
            # Use markdown in the button for formatting, ensure unique key
            if st.button(button_label, key=f"app_button_{app_id}", use_container_width=True):
                selected_app_title = title
                selected_app_id = app_id
                st.success(f"Selected app: {title}")
                
    return selected_app_title, selected_app_id

def perform_search(query: str, playstore_service: PlayStoreService = None) -> List[Dict[str, str]]:
    if not query:
        return []
    
    # Create PlayStoreService if not provided
    if playstore_service is None:
        if 'playstore_service' in st.session_state:
            playstore_service = st.session_state.playstore_service
        else:
            playstore_service = PlayStoreService()
            st.session_state.playstore_service = playstore_service
    
    with st.spinner(f"Searching for apps matching '{query}'..."):
        search_results = playstore_service.search_apps(query) # Now returns list of dicts
        
        if not search_results:
            st.info(f"No apps found matching '{query}'. Try a different search term.")
            return []
        
        # Limit to top 10 apps (already a list)
        top_results = search_results[:10]
        return top_results