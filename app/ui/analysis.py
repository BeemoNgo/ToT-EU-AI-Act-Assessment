"""
UI components for displaying analysis results.
"""
import streamlit as st
import pandas as pd
from typing import Optional
import re
from models.app_data import AnalysisResults, AppDetails

def display_developer_info(app_details: AppDetails) -> None:
    with st.expander("Developer Information & Data Safety Claims", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("App Details")
            st.markdown(f"**Publisher:** {app_details.publisher}")
            st.markdown(f"**Link:** [{app_details.link}]({app_details.link})")
            st.markdown("**Description:**")
            st.markdown(f"_{app_details.description}_")
            
        with col2:
            st.subheader("Data Safety Claims")
            st.markdown(f"**Shared Data:** {app_details.shared_data}")
            st.markdown(f"**Collected Data:** {app_details.collected_data}")
            st.markdown(f"**Security Practices:** {app_details.security_practices}")

def display_app_details_table(app_details: AppDetails) -> None:
    """Displays key app details in a table format with columns."""
    st.subheader("Selected App Details")
    
    # Create a dictionary with app details
    data = {
        "App Name": app_details.name,
        "Publisher": app_details.publisher,
        "Shared Data": app_details.shared_data,
        "Collected Data": app_details.collected_data,
        "Security Practices": app_details.security_practices
    }
    
    # Convert dictionary to DataFrame, transpose it, and reset index
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
    df.index.name = 'Attribute'
    df = df.reset_index()
    
    # Display the transposed DataFrame
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.markdown(f"**App Link:** [{app_details.link}]({app_details.link})")
    
    # Show full description
    with st.expander("Full Description", expanded=False):
        st.markdown(app_details.description)

def display_review_summary(analysis: str, filtered_count: int, raw_count: int) -> None:
    st.subheader("AI Review Summary (via Tree of Thought Reasoning)")
    st.markdown("This summary was generated using Tree of Thought logic, synthesizing multiple analytical steps for deeper insight.")
    st.markdown(analysis)

    review_numbers = set(re.findall(r'Review #(\d+)', analysis))
    if review_numbers:
        st.markdown("**Cited Reviews:**")
        st.markdown(f"This analysis cites {len(review_numbers)} specific reviews (Reviews #{', #'.join(sorted(review_numbers))})")

    st.caption(f"Analyzed {filtered_count} reviews (out of {raw_count} fetched) with sufficient length.")

def display_difference_analysis(analysis: str) -> None:
    st.subheader("Discrepancy Analysis (ToT-Based)")
    st.markdown("This section highlights differences between user experience and developer claims using Tree of Thought evaluation.")
    st.markdown(analysis)

def display_sample_reviews(reviews_sample: Optional[pd.DataFrame]) -> None:
    if reviews_sample is not None and not reviews_sample.empty:
        with st.expander("Sample Filtered Reviews Used for Analysis", expanded=False):
            # Select columns to display
            display_df = reviews_sample[['score', 'content', 'date', 'thumbs_up']].copy()
            
            # Format date if it's a datetime
            if 'date' in display_df.columns and pd.api.types.is_datetime64_any_dtype(display_df['date']):
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Display the DataFrame
            st.dataframe(display_df)

def display_error(error_message: str) -> None:
    st.error(f"Analysis could not be completed: {error_message}")

def display_eu_ai_act_note() -> None:
    st.info("""
        **Note:** This analysis focuses on summarizing user reviews and comparing them to developer descriptions.
        Assessing specific risks or threats according to the EU AI Act criteria would require a separate, dedicated analysis based on the Act's defined risk categories and rules, which is **not** implemented here.
    """)

def display_cited_reviews(citations: str, reviews_df: pd.DataFrame) -> None:
    """Display the actual content of cited reviews."""
    if not citations or reviews_df.empty:
        return
        
    # Extract review numbers from citations
    review_numbers = set(re.findall(r'Review #(\d+)', citations))
    
    if not review_numbers:
        return
        
    st.markdown("**Cited Reviews:**")
    
    # Filter reviews to show only the cited ones
    cited_reviews = reviews_df[reviews_df['review_index'].isin([int(num) for num in review_numbers])]
    if not cited_reviews.empty:
        # Select and format columns for display
        display_df = cited_reviews[['review_index', 'score', 'content', 'date', 'thumbs_up']].copy()
        
        # Format date if it's a datetime
        if 'date' in display_df.columns and pd.api.types.is_datetime64_any_dtype(display_df['date']):
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Rename columns for better display
        display_df.columns = ['Review #', 'Rating', 'Content', 'Date', 'Thumbs Up']
        
        # Display the DataFrame
        st.dataframe(display_df, use_container_width=True)

def display_supporting_reviews(supporting_reviews: str, reviews_df: pd.DataFrame) -> None:
    """Display the supporting reviews with their explanations."""
    if not supporting_reviews or reviews_df.empty:
        return
        
    st.markdown("**Supporting Reviews:**")
    
    # Split the supporting reviews into individual review sections
    review_sections = supporting_reviews.split('- Review #')
    review_sections = [s for s in review_sections if s.strip()]
    
    for section_text in review_sections:
        original_section_text_for_warning = section_text[:150] # For cleaner error messages
        try:
            # Extract review number part and the rest of the section
            num_part, rest_of_section = section_text.split(':', 1)
            review_num_str = num_part.strip()
            review_num = int(review_num_str) # Potential ValueError

            # Extract content (quote) and explanation
            # The rest_of_section should be "QUOTE How it supports: EXPLANATION"
            # Make the split case-insensitive by converting to lower case
            rest_lower = rest_of_section.lower()
            separator_pos = rest_lower.find('how it supports:')
            
            content = ""
            explanation = ""

            if separator_pos != -1:
                # Use the position from the lowercase search to split the original string
                content = rest_of_section[:separator_pos].strip()
                explanation = rest_of_section[separator_pos + len('how it supports:'):].strip()
            else:
                # 'How it supports:' separator is missing
                st.warning(f"Missing 'How it supports:' separator for review #{review_num_str}. Treating all as quote. Section: '{original_section_text_for_warning}...'")
                content = rest_of_section.strip()
                explanation = "N/A (separator missing)"

            # Get the review from the DataFrame using review_index
            try:
                review = reviews_df[reviews_df['review_index'] == review_num].iloc[0]
                
                # Display review information
                st.markdown(f"**Review #{review_num} (Rating: {review['score']}⭐)**")
                st.markdown("**Original Review:**")
                st.markdown(f"_{review['content']}_")
                st.markdown("**Relevant Quote:**")
                st.markdown(f"_{content}_")
                st.markdown("**How it supports the classification:**")
                st.markdown(explanation)
                st.caption(f"Date: {review['date']} | Thumbs Up: {review['thumbs_up']}")
                st.divider()
            except IndexError:
                st.warning(f"No review found with index {review_num}. Total reviews in dataset: {len(reviews_df)}")
                continue
                
        except ValueError: # Handles error from int(review_num_str)
            st.warning(f"Could not parse review number from '{review_num_str if 'review_num_str' in locals() else section_text.split(':')[0].strip()}'. Section: '{original_section_text_for_warning}...'")
            continue
        except Exception as e: # Catch any other unexpected error during section processing
            st.warning(f"Error processing review section starting with '{original_section_text_for_warning}...', Error: {str(e)}")
            continue

def display_eu_ai_act_classification(classification_result: dict, reviews_df: Optional[pd.DataFrame] = None) -> None:
    st.subheader("EU AI Act Classification (ToT Evaluation)")

    risk_colors = {
        "Unacceptable risk": "red",
        "High risk": "orange",
        "Limited risk": "blue",
        "Minimal Risk": "green"
    }

    risk_type = classification_result.get('risk_type', 'Unknown')
    color = risk_colors.get(risk_type, "gray")

    st.markdown(f"<h3 style='color: {color};'>Classification: {risk_type}</h3>", unsafe_allow_html=True)

    triggered_questions = classification_result.get('triggered_questions', [])

    if triggered_questions:
        st.markdown("**Triggering Criteria & Reasoning (from ToT):**")
        for trigger in triggered_questions:
            with st.expander(f"Question: {trigger['question']}", expanded=True):
                if trigger.get('reasoning'):
                    st.markdown("**Reasoning:**")
                    st.markdown(trigger['reasoning'])

                if trigger.get('supporting_reviews') and reviews_df is not None:
                    display_supporting_reviews(trigger['supporting_reviews'], reviews_df)
    elif risk_type == "Minimal Risk":
        st.markdown("No risk triggers identified. App likely poses minimal regulatory concern.")

    confidence = classification_result.get('confidence_score', 0.0)
    st.progress(confidence)
    st.caption(f"Overall Confidence (ToT-based Evaluation): {confidence:.2f}")

    with st.expander("About EU AI Act Risk Categories", expanded=False):
        st.markdown("""
        **EU AI Act Risk Categories:**

        - **Unacceptable risk:** Prohibited AI systems that harm safety, rights, or democracy.
        - **High risk:** AI in critical domains like health, law enforcement, or education.
        - **Limited risk:** AI requiring transparency (e.g., chatbot disclosure).
        - **Minimal Risk:** Most consumer AI — like entertainment or productivity apps.
        """)

def display_footer() -> None:
    """Display the footer inside a div for CSS targeting."""
    footer_html = """
    <div class="app-footer">
        <hr>
        <p style="text-align: center; font-size: 0.9em; color: gray;">
            Powered by Streamlit, OpenAI, and google-play-scraper.
        </p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

def display_analysis_results(results: AnalysisResults, app_name: str) -> None:
    st.divider()
    st.subheader(f"Analysis Results for: {app_name}")
    
    if results.has_error():
        display_error(results.error)
    else:
        display_review_summary(results.user_review_analysis, 
                               results.filtered_review_count, 
                               results.raw_review_count)
        display_difference_analysis(results.difference_analysis)
        display_sample_reviews(results.filtered_reviews_sample)

        # Show ToT Trace if available
        if results.tot_trace_review:
            trace = results.tot_trace_review
            display_tree_of_thought_trace(
                prompt_root=trace["prompt_root"],
                thought_prompts=trace["thought_prompts"],
                branch_responses=trace["branches"],
                final_response=trace["final_response"]
            )

        if hasattr(results, 'eu_ai_act_classification') and results.eu_ai_act_classification:
            reviews_for_display = getattr(results, 'filtered_reviews', results.filtered_reviews_sample)
            display_eu_ai_act_classification(
                results.eu_ai_act_classification,
                reviews_for_display
            )
        else:
            display_eu_ai_act_note()
    
    # Removed footer call from here
    # display_footer() 
def display_tree_of_thought_trace(prompt_root: str, thought_prompts: list[str], branch_responses: list[str], final_response: str) -> None:
    """
    Visualizes the Tree-of-Thought reasoning: shows each branch and the final synthesis.
    
    Parameters:
    - prompt_root: The base prompt shared by all thought branches.
    - thought_prompts: The individual reasoning prompts (branches).
    - branch_responses: The model's output for each thought prompt.
    - final_response: The final synthesized output.
    """
    st.subheader("Tree of Thought Reasoning (ToT Trace)")
    st.markdown("This section shows how the system used multiple analytical 'thoughts' to arrive at a final judgment.")

    with st.expander("Root Prompt", expanded=False):
        st.code(prompt_root, language='markdown')

    for idx, (thought, response) in enumerate(zip(thought_prompts, branch_responses), start=1):
        with st.expander(f"Thought #{idx}: {thought}", expanded=False):
            st.markdown("**LLM Response:**")
            st.markdown(response)

    with st.expander("Final Synthesized Judgment", expanded=True):
        st.markdown("This response is generated by combining the best reasoning paths from above.")
        st.success(final_response)
