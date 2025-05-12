from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd

@dataclass
class AppDetails:
    app_id: str
    name: str
    publisher: str = "N/A"
    link: str = ""
    description: str = "N/A"
    html_description: str = "N/A"
    shared_data: str = "Not specified"
    collected_data: str = "Not specified"
    security_practices: str = "Not specified"
    
    @classmethod
    def from_dict(cls, app_id: str, name: str, details_dict: Dict[str, Any]) -> 'AppDetails':
        link = details_dict.get('url', f"https://play.google.com/store/apps/details?id={app_id}")
        
        return cls(
            app_id=app_id,
            name=details_dict.get('title', name),
            publisher=details_dict.get('developer', 'N/A'),
            link=link,
            description=details_dict.get('description', 'N/A'),
            html_description=details_dict.get('descriptionHTML', 'N/A'),
            shared_data=cls._format_data_safety(details_dict.get('dataSafety', {}).get('dataShared')),
            collected_data=cls._format_data_safety(details_dict.get('dataSafety', {}).get('dataCollected')),
            security_practices=cls._format_data_safety(details_dict.get('dataSafety', {}).get('securityPractices'))
        )
    
    @classmethod
    def from_minimal(cls, app_id: str, name: str) -> 'AppDetails':
        return cls(
            app_id=app_id,
            name=name,
            link=f"https://play.google.com/store/apps/details?id={app_id}"
        )
    
    @staticmethod
    def _format_data_safety(data) -> str:
        if not data:
            return "Not specified"
        
        if isinstance(data, list):
            # Extract data_type or practice_type from dictionaries in the list
            processed_items = []
            for item in data:
                if isinstance(item, dict):
                    processed_items.append(item.get('data_type', item.get('practice_type', str(item))))
                else:
                    processed_items.append(str(item))
            return ', '.join(filter(None, processed_items)) if processed_items else 'Not specified'
        
        # If it's already a string, return it
        return str(data)
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "App Name": self.name,
            "Publisher": self.publisher,
            "Link": self.link,
            "Full Description": self.description,
            "App Info Modal": self.html_description,
            "Shared Data": self.shared_data,
            "Collected Data": self.collected_data,
            "Security Practices": self.security_practices
        }

@dataclass
class Review:
    app_name: str
    score: float
    content: str
    date: Any  # Could be datetime or string
    thumbs_up: int = 0
    
    @classmethod
    def from_dict(cls, app_name: str, review_dict: Dict[str, Any]) -> 'Review':
        return cls(
            app_name=app_name,
            score=review_dict.get('score'),
            content=review_dict.get('content', ''),
            date=review_dict.get('at'),
            thumbs_up=review_dict.get('thumbsUpCount', 0)
        )

@dataclass
class AnalysisResults:
    developer_details: AppDetails
    raw_review_count: int = 0
    filtered_review_count: int = 0
    user_review_analysis: str = ""
    difference_analysis: str = ""
    filtered_reviews_sample: Optional[pd.DataFrame] = None
    error: str = ""
    
    def has_error(self) -> bool:
        return bool(self.error)
    
    def has_reviews(self) -> bool:
        return self.filtered_review_count > 0 