"""
Data models for app analysis.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd

@dataclass
class AppDetails:
    """Details about an app from the Play Store."""
    name: str
    app_id: str
    publisher: str
    description: str
    shared_data: str
    collected_data: str
    security_practices: str
    rating: float = 0.0
    review_count: int = 0
    install_count: str = ""
    size: str = ""
    version: str = ""
    last_updated: str = ""
    content_rating: str = ""
    category: str = ""
    price: str = "Free"
    in_app_purchases: bool = False
    developer_website: str = ""
    developer_email: str = ""
    developer_address: str = ""

@dataclass
class AnalysisResults:
    """Results of the app analysis."""
    developer_details: AppDetails
    user_review_analysis: str = ""
    difference_analysis: str = ""
    raw_review_count: int = 0
    filtered_review_count: int = 0
    filtered_reviews: Optional[pd.DataFrame] = None
    filtered_reviews_sample: Optional[pd.DataFrame] = None
    error: str = ""
    eu_ai_act_classification: Optional[Dict[str, Any]] = None
    
    def has_error(self) -> bool:
        """Check if there's an error in the results."""
        return bool(self.error)
    
    def has_reviews(self) -> bool:
        """Check if there are any reviews in the results."""
        return self.filtered_review_count > 0 