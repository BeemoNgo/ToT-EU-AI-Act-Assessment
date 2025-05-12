from typing import Dict, List, Tuple, Optional
import pandas as pd
from google_play_scraper import app, search, Sort, reviews, exceptions

from models.app_data import AppDetails, Review
from services.logger import logger, StatusLogger
from services.selenium_scraper import SeleniumScraperService
from config.settings import MAX_REVIEWS_TO_FETCH, MAX_SEARCH_RESULTS
from utils.data_utils import create_reviews_dataframe

class PlayStoreService:
    def __init__(self):
        # Initialize the selenium scraper
        self.selenium_scraper = None
    
    def _get_selenium_scraper(self) -> SeleniumScraperService:
        """
        Lazy initialization of the selenium scraper to avoid creating it when not needed
        """
        if self.selenium_scraper is None:
            self.selenium_scraper = SeleniumScraperService()
        return self.selenium_scraper
    
    def close_selenium_scraper(self):
        """
        Close the selenium scraper when done to free resources
        """
        if self.selenium_scraper:
            self.selenium_scraper.close_driver()
            self.selenium_scraper = None
    
    def search_apps(self, query: str, status_logger: Optional[StatusLogger] = None) -> List[Dict[str, str]]:
        log = status_logger or logger
        log.info(f"Searching for '{query}' apps (max {MAX_SEARCH_RESULTS})...")
        
        try:
            search_results = search(
                query,
                lang="en",
                country="us",
                n_hits=MAX_SEARCH_RESULTS
            )
            
            # Extract title, appId, and developer
            app_list = [
                {
                    'title': result['title'], 
                    'appId': result['appId'], 
                    'developer': result.get('developer', 'N/A') # Use .get for safety
                } 
                for result in search_results
            ]
            
            log.info(f"Found {len(app_list)} apps matching '{query}'.")
            
            return app_list
            
        except exceptions.NotFoundError:
            log.warning(f"No apps found for query: '{query}'")
            return []
            
        except Exception as e:
            log.error(f"Error searching apps for '{query}': {e}", exc_info=True)
            return []
    
    def get_app_details(self, app_id: str, app_name: str, status_logger: Optional[StatusLogger] = None) -> AppDetails:
        log = status_logger or logger
        log.info(f"Fetching details for {app_name} ({app_id})...")
        
        try:
            details = app(app_id, lang='en', country='us')
            log.info(f"Successfully fetched details for {app_name}.")
            
            app_details = AppDetails.from_dict(app_id, app_name, details)
            
            # Check if we need to get data safety info using Selenium
            if app_details.shared_data == "Not specified" or app_details.collected_data == "Not specified":
                log.info(f"Fetching detailed data safety info for {app_name} using Selenium...")
                
                selenium_scraper = self._get_selenium_scraper()
                data_safety = selenium_scraper.extract_data_safety(app_id, status_logger)
                
                # Update app details with the data safety info
                app_details.shared_data = data_safety["shared_data"]
                app_details.collected_data = data_safety["collected_data"]
                app_details.security_practices = data_safety["security_practices"]
                
                # If description is short, try to get a better one
                if len(app_details.description) < 100:
                    app_link = f"https://play.google.com/store/apps/details?id={app_id}"
                    app_details.description = selenium_scraper.get_full_description(app_link, status_logger)
            
            return app_details
            
        except exceptions.NotFoundError:
            log.warning(f"App details not found for {app_name} ({app_id}).")
            return AppDetails.from_minimal(app_id, app_name)
            
        except Exception as e:
            log.error(f"Error fetching details for {app_name}: {e}", exc_info=True)
            return AppDetails.from_minimal(app_id, app_name)
    
    def get_app_reviews(self, app_id: str, app_name: str, status_logger: Optional[StatusLogger] = None, 
                        max_reviews: int = MAX_REVIEWS_TO_FETCH) -> pd.DataFrame:
        log = status_logger or logger
        log.info(f"Fetching reviews for {app_name} (up to {max_reviews})...")
        
        try:
            review_results, continuation_token = reviews(
                app_id,
                lang="en",
                country="us",
                sort=Sort.NEWEST,
                count=max_reviews,
                filter_score_with=None  # Fetch all scores
            )
            
            log.info(f"Fetched {len(review_results)} raw reviews for {app_name}.")
            
            # Convert to Review objects
            review_objects = []
            for review_dict in review_results:
                review = Review.from_dict(app_name, review_dict)
                review_objects.append(review)
            
            # Convert list of Review objects to DataFrame
            reviews_df = create_reviews_dataframe(review_objects)
            log.info(f"Converted {len(reviews_df)} reviews to DataFrame.")
            
            return reviews_df
            
        except exceptions.NotFoundError:
            log.warning(f"Reviews not found for {app_name} ({app_id}).")
            return pd.DataFrame() # Return empty DataFrame
            
        except Exception as e:
            log.error(f"Error retrieving reviews for {app_name}: {e}", exc_info=True)
            return pd.DataFrame() # Return empty DataFrame
    
    def get_app_details_and_reviews(self, app_id: str, app_name: str, 
                                   status_logger: Optional[StatusLogger] = None) -> Tuple[AppDetails, pd.DataFrame]:
        
        app_details = self.get_app_details(app_id, app_name, status_logger)
        reviews_df = self.get_app_reviews(app_id, app_name, status_logger)
        
        # Make sure to clean up Selenium driver when done
        self.close_selenium_scraper()
        
        return app_details, reviews_df 