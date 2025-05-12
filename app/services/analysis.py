"""
Service for interacting with OpenAI for analysis.
"""
import time
from typing import Optional, Literal, Dict, Any
import pandas as pd
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

from models.app_data import AppDetails, AnalysisResults
from services.logger import logger, StatusLogger
from services.cache import CacheService
from config.settings import OPENAI_MODEL
from utils.data_utils import filter_reviews_by_length, prepare_reviews_for_analysis

class EUAIActResponse(BaseModel):
    answer: Literal["Yes", "No"]
    reasoning: str
    supporting_reviews: Optional[str] = None

class EvaluationResult(BaseModel):
    question: str
    reasoning: str
    supporting_reviews: Optional[str] = None
    confidence: float

class ReviewAnalysisResponse(BaseModel):
    content: str

class DifferenceAnalysisResponse(BaseModel):
    content: str

class FallbackResponse(BaseModel):
    answer: Literal["Yes", "No"]

class AnalysisService:
    def __init__(self, api_key: Optional[str] = None):
        try:
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            
            # Initialize cache service
            self.cache_service = CacheService()
            
            # Test the connection (optional)
            self.client.models.list()
            logger.info("OpenAI client initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise
    
    def run_tree_of_thought(self, prompt_root: str, thought_prompts: list[str], model: str = OPENAI_MODEL) -> str:
        """Executes multiple thoughts and synthesizes the best answer."""
        branches = []
        for thought in thought_prompts:
            full_prompt = f"{prompt_root}\n\n{thought}"
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.1
            )
            branches.append(response.choices[0].message.content.strip())

        # Synthesize best version
        synthesis_prompt = f"""You are given several analytical paths about an app. Synthesize the best points from each into one final response.\n\n{chr(10).join(branches)}"""
        final_response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.1
        )
        return final_response.choices[0].message.content.strip()

    def analyze_reviews(self, app_name: str, app_id: str, reviews_text: str, 
                       status_logger: Optional[StatusLogger] = None) -> str:
        log = status_logger or logger

        cached_result = self.cache_service.get_cached_analysis(app_id, "reviews")
        if cached_result:
            log.info(f"Using cached review analysis for {app_name}")
            return cached_result

        if not reviews_text:
            log.warning(f"No reviews text provided for analysis of {app_name}.")
            return "No reviews provided for analysis."

        prompt_root = f"The following reviews are from users of the app '{app_name}'. Analyze them."\
                       f" Provide responses to the following questions based only on the reviews:\n\nReviews:\n---\n{reviews_text}\n---"

        thought_prompts = [
            "1. What are the most frequently mentioned features (positive or negative)? Cite review numbers.",
            "2. What are the most common complaints or bugs users reported? Cite review numbers.",
            "3. What do users most appreciate or praise in the app? Cite review numbers.",
            "4. What is the overall tone or sentiment expressed by the users? Summarize in 1–2 sentences."
        ]

        log.update(label=f"Analyzing reviews for {app_name} using Tree of Thought logic...")
        log.info(f"Generating multi-path review analysis for {app_name}")

        try:
            analysis = self.run_tree_of_thought(prompt_root, thought_prompts)
            self.cache_service.cache_analysis(app_id, "reviews", analysis)
            log.write(f"✓ Completed review analysis for {app_name}.")
            return analysis

        except Exception as e:
            log.error(f"ToT review analysis failed for {app_name}: {e}", exc_info=True)
            return f"Error analyzing reviews: {str(e)}"

    def analyze_difference(self, app_details: AppDetails, user_review_summary: str,
                          status_logger: Optional[StatusLogger] = None) -> str:
        log = status_logger or logger
        app_name = app_details.name
        app_id = app_details.app_id

        cached_result = self.cache_service.get_cached_analysis(app_id, "difference")
        if cached_result:
            log.info(f"Using cached difference analysis for {app_name}")
            return cached_result

        developer_desc = "\n".join([
            f"Full Description: {app_details.description}",
            f"Shared Data Claim: {app_details.shared_data}",
            f"Collected Data Claim: {app_details.collected_data}",
            f"Security Practices Claim: {app_details.security_practices}"
        ])

        prompt_root = f"Compare this user summary with the developer's official claims for app '{app_name}':\n\n" \
                      f"User Review Summary:\n{user_review_summary}\n\nDeveloper Description:\n{developer_desc}"

        thought_prompts = [
            "1. Are there any promised features that users say don't work or are missing?",
            "2. Do users report performance issues that contradict claimed stability or quality?",
            "3. Are there user concerns about data use or security not addressed by the developer?",
            "4. Does the marketing tone differ from how users describe the app's experience?"
        ]

        log.update(label=f"Analyzing difference for {app_name} using ToT...")
        log.info(f"Generating discrepancy analysis using multiple thoughts for {app_name}")

        try:
            analysis_text = self.run_tree_of_thought(prompt_root, thought_prompts)
            self.cache_service.cache_analysis(app_id, "difference", analysis_text)
            log.write(f"✓ Completed difference analysis for {app_name}.")
            return analysis_text

        except Exception as e:
            log.error(f"ToT difference analysis failed for {app_name}: {e}", exc_info=True)
            return f"Error analyzing differences: {str(e)}"
    
    def analyze_app(self, app_details: AppDetails, reviews_df: pd.DataFrame, 
                   status_logger: Optional[StatusLogger] = None) -> AnalysisResults:
        log = status_logger or logger
        
        # Initialize results object
        results = AnalysisResults(developer_details=app_details)
        
        # Check if we have reviews
        if reviews_df.empty:
            log.warning(f"No reviews found for {app_details.name}.")
            results.error = "No reviews found."
            return results
        
        # Store raw review count
        results.raw_review_count = len(reviews_df)
        log.write(f"Raw reviews fetched: {results.raw_review_count}")
        
        # Try to get filtered reviews from cache
        filtered_df = self.cache_service.get_cached_dataframe(app_details.app_id, "filtered_reviews")
        if filtered_df is not None:
            log.info(f"Using cached filtered reviews for {app_details.name}")
            results.filtered_review_count = len(filtered_df)
            results.filtered_reviews = filtered_df  # Store complete filtered DataFrame
            results.filtered_reviews_sample = filtered_df.head()  # Store sample for display
            log.write(f"✓ Using cached filtered reviews: {results.filtered_review_count} reviews.")
        else:
            # Filter reviews
            log.update(label=f"Filtering reviews for {app_details.name}...")
            # Make sure we have review_index column
            if 'review_index' not in reviews_df.columns:
                reviews_df['review_index'] = range(1, len(reviews_df) + 1)
            
            filtered_df = filter_reviews_by_length(reviews_df)
            results.filtered_review_count = len(filtered_df)
            results.filtered_reviews = filtered_df  # Store complete filtered DataFrame
            results.filtered_reviews_sample = filtered_df.head()  # Store sample for display
            
            # Cache the filtered reviews
            self.cache_service.cache_dataframe(app_details.app_id, "filtered_reviews", filtered_df)
            log.write(f"✓ Filtered reviews: {results.filtered_review_count} remaining (with sufficient length).")
        
        # Check if we have filtered reviews
        if filtered_df.empty:
            log.warning(f"No reviews met the minimum length requirement for {app_details.name}.")
            results.error = "No sufficiently long reviews found for analysis."
            return results
        
        # Prepare text for analysis
        reviews_text = prepare_reviews_for_analysis(filtered_df)
        
        # Analyze reviews
        results.user_review_analysis = self.analyze_reviews(
            app_details.name, app_details.app_id, reviews_text, status_logger=log
        )
        
        # Check if review analysis succeeded
        if results.user_review_analysis.startswith("Error analyzing reviews:"):
            results.error = results.user_review_analysis
            return results
        
        # Analyze difference
        results.difference_analysis = self.analyze_difference(
            app_details, results.user_review_analysis, status_logger=log
        )
        
        # Check if difference analysis succeeded
        if results.difference_analysis.startswith("Error analyzing differences:"):
            results.error = results.difference_analysis
            return results
        
        return results
    
    def _evaluate_single_prompt(self, question_text: str, input_description: str, log: StatusLogger) -> Optional[EvaluationResult]:
        """Evaluate a single EU AI Act question using ToT reasoning and produce full traceable output."""
        try:
            prompt_root = (
                f"You are assessing whether this AI app raises concerns under the following question from the EU AI Act:\n"
                f"\nQuestion: {question_text}\n"
                f"\nBased on this information:\n{input_description}\n"
                f"\nUse Tree of Thought reasoning to answer with full justification and cited evidence."
            )

            thought_prompts = [
                "Step 1: What is this question really asking about in terms of risk or harm?",
                "Step 2: Does the app (based on description + reviews) exhibit anything matching this risk?",
                "Step 3: Identify user reviews that mention this concern. Quote them and link them to the question.",
                "Step 4: Final judgment: Answer Yes or No, explain reasoning, and summarize supporting evidence."
            ]

            final_response = self.run_tree_of_thought(prompt_root, thought_prompts)

            # Parse answer and evidence (rudimentary but traceable)
            answer = "Yes" if "Answer: Yes" in final_response else "No"
            return EvaluationResult(
                question=question_text,
                reasoning=final_response,
                supporting_reviews=final_response if answer == "Yes" else None,
                confidence=1.0 if answer == "Yes" else 0.5
            )

        except Exception as e:
            log.logger.error(f"Error evaluating question with ToT: {e}", exc_info=True)
            return None

    def perform_eu_ai_act_classification(self, app_details: AppDetails, reviews_analysis: str,
                                        difference_analysis: str,
                                        prompts_df: pd.DataFrame, status_logger: Optional[StatusLogger] = None) -> dict:
        log = status_logger or logger
        app_id = app_details.app_id
        
        # Try to get from cache first
        cached_result = self.cache_service.get_cached_analysis(app_id, "eu_ai_act")
        if cached_result:
            log.info(f"Using cached EU AI Act classification for {app_details.name}")
            return cached_result
        
        log.update(label=f"Classifying {app_details.name} according to EU AI Act...")
        
        # Prepare base description
        base_description = (
            f"App Name: {app_details.name}. "
            f"Publisher: {app_details.publisher}. "
            f"Full Description: {app_details.description}. "
            f"Data shared with third parties: {app_details.shared_data}. "
            f"Data collected by the app: {app_details.collected_data}. "
            f"Security practices: {app_details.security_practices}."
        )
        
        # Add review analysis and difference analysis if available
        input_description = base_description
        if reviews_analysis and not reviews_analysis.startswith("Error"):
            input_description += f"\n\nUser Review Analysis Summary:\n---\n{reviews_analysis}\n---"
        if difference_analysis and not difference_analysis.startswith("Error"):
            input_description += f"\n\nAnalysis of Differences (User Reviews vs. Developer Claims):\n---\n{difference_analysis}\n---"
        
        # Process risk types from highest to lowest risk
        risk_types = ["Unacceptable risk", "High risk", "Limited risk"]
        
        for risk_type in risk_types:
            # Filter prompts by risk type
            risk_prompts = prompts_df[prompts_df['Type'] == risk_type]
            
            if risk_prompts.empty:
                continue
                
            log.update(label=f"Evaluating {risk_type} criteria for {app_details.name}...")
            
            # Use ThreadPoolExecutor to process prompts in parallel
            triggered_questions_info = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all prompts for this risk level
                future_to_prompt = {
                    executor.submit(
                        self._evaluate_single_prompt,
                        row['Prompt'],
                        input_description,
                        log
                    ): row['Prompt']
                    for _, row in risk_prompts.iterrows()
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_prompt):
                    result = future.result()
                    if result:
                        triggered_questions_info.append(result)

            # If ANY questions triggered this risk level, classify and return
            if triggered_questions_info:
                log.write(f"✓ Classified as {risk_type} based on {len(triggered_questions_info)} trigger(s).")
                overall_confidence = max(info.confidence for info in triggered_questions_info) if triggered_questions_info else 0.0
                
                result = {
                    'risk_type': risk_type,
                    'confidence_score': overall_confidence,
                    'triggered_questions': [
                        {
                            'question': info.question,
                            'reasoning': info.reasoning,
                            'supporting_reviews': info.supporting_reviews,
                            'confidence': info.confidence
                        }
                        for info in triggered_questions_info
                    ]
                }
                
                # Cache the result
                self.cache_service.cache_analysis(app_id, "eu_ai_act", result)
                return result
        
        # If no risk type was triggered, classify as minimal risk
        result = {
            'risk_type': "Minimal Risk",
            'confidence_score': 1.0,
            'triggered_questions': []
        }
        
        # Cache the result
        self.cache_service.cache_analysis(app_id, "eu_ai_act", result)
        return result 
    