"""
Service for scraping Google Play Store using Selenium for data that's not available via the google-play-scraper library.
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException, TimeoutException
from bs4 import BeautifulSoup
import time
from typing import Dict, Optional

from services.logger import logger, StatusLogger

class SeleniumScraperService:
    def __init__(self):
        # Set up driver options
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.driver = None
    
    def initialize_driver(self):
        if self.driver is None:
            try:
                # Use webdriver_manager for automatic ChromeDriver installation
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=self.chrome_options)
                self.driver.set_page_load_timeout(30)  # Set timeout to avoid hanging
            except Exception as e:
                logger.error(f"Failed to initialize Chrome driver: {e}", exc_info=True)
                raise RuntimeError(f"Failed to initialize Chrome driver: {e}")
    
    def close_driver(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.error(f"Error closing Chrome driver: {e}")
            finally:
                self.driver = None
    
    def extract_data_safety(self, app_id: str, status_logger: Optional[StatusLogger] = None) -> Dict[str, str]:
        """
        Extract data safety information for an app from the Google Play Store.
        
        Args:
            app_id: The package ID of the app
            status_logger: Optional logger for status updates
            
        Returns:
            Dictionary containing Shared Data, Collected Data, and Security Practices
        """
        log = status_logger or logger
        log.info(f"Extracting data safety information for {app_id}...")
        
        try:
            self.initialize_driver()
            
            data_safety_url = f"https://play.google.com/store/apps/datasafety?id={app_id}&hl=en"
            self.driver.get(data_safety_url)
            time.sleep(2)  # Wait for page to load
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            shared_data = []
            collected_data = []
            security_practices = []
            
            # Try different CSS selectors as Google Play Store may change them
            for section in soup.find_all(["h2", "div"], class_=["q1rIdc", "bUWb7c"]):
                if not section.get_text(strip=True):
                    continue
                    
                title = section.get_text(strip=True)
                
                if "Data shared" in title:
                    block = section.find_next("div", class_=["XgPdwe", "ULeqic"])
                    if block:
                        items = block.find_all(["h3", "div"], class_=["aFEzEb", "kzpmxe"])
                        shared_data = [item.get_text(strip=True) for item in items]
                
                elif "Data collected" in title:
                    block = section.find_next("div", class_=["XgPdwe", "ULeqic"]) 
                    if block:
                        items = block.find_all(["h3", "div"], class_=["aFEzEb", "kzpmxe"])
                        collected_data = [item.get_text(strip=True) for item in items]
                
                elif "Security practices" in title:
                    block = section.find_next("div", class_=["XgPdwe", "ULeqic"])
                    if block:
                        items = block.find_all(["h3", "div"], class_=["aFEzEb", "kzpmxe"])
                        security_practices = [item.get_text(strip=True) for item in items]
            
            # Check if we found any data
            if not any([shared_data, collected_data, security_practices]):
                log.warning(f"No data safety information found using primary selectors for {app_id}. Trying alternative approach...")
                
                # Try a different approach to extract data safety info
                sections = soup.find_all("div", class_=["bUWb7c", "VfPpkd-cnJW5e"])
                for section in sections:
                    header = section.find(["h2", "div"], class_=["q1rIdc", "VfPpkd-WsjYwc"])
                    if not header:
                        continue
                        
                    title = header.get_text(strip=True)
                    
                    if "Data shared" in title.lower():
                        items = section.find_all(["div", "li"], class_=["kzpmxe", "VfPpkd-gIZMFc"])
                        shared_data = [item.get_text(strip=True) for item in items if item.get_text(strip=True)]
                    
                    elif "Data collected" in title.lower():
                        items = section.find_all(["div", "li"], class_=["kzpmxe", "VfPpkd-gIZMFc"])
                        collected_data = [item.get_text(strip=True) for item in items if item.get_text(strip=True)]
                    
                    elif "Security practices" in title.lower():
                        items = section.find_all(["div", "li"], class_=["kzpmxe", "VfPpkd-gIZMFc"])
                        security_practices = [item.get_text(strip=True) for item in items if item.get_text(strip=True)]
            
            result = {
                "shared_data": ", ".join(shared_data) if shared_data else "Not specified",
                "collected_data": ", ".join(collected_data) if collected_data else "Not specified",
                "security_practices": ", ".join(security_practices) if security_practices else "Not specified"
            }
            
            log.info(f"Successfully extracted data safety information for {app_id}")
            return result
            
        except TimeoutException:
            log.error(f"Timeout while extracting data safety information for {app_id}")
            return {
                "shared_data": "Timeout retrieving data",
                "collected_data": "Timeout retrieving data",
                "security_practices": "Timeout retrieving data"
            }
        except Exception as e:
            log.error(f"Error extracting data safety information for {app_id}: {e}", exc_info=True)
            return {
                "shared_data": "Error retrieving data",
                "collected_data": "Error retrieving data",
                "security_practices": "Error retrieving data"
            }
        finally:
            # Don't close the driver here, as it might be reused for other operations
            pass
            
    def get_full_description(self, app_link: str, status_logger: Optional[StatusLogger] = None) -> str:
        """
        Get the full description of an app from its Play Store page.
        
        Args:
            app_link: The Google Play Store link for the app
            status_logger: Optional logger for status updates
            
        Returns:
            The full app description
        """
        log = status_logger or logger
        log.info(f"Extracting full description from {app_link}...")
        
        try:
            self.initialize_driver()
            
            self.driver.get(app_link + "&hl=en")
            time.sleep(2)  # Wait for page to load
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # First try meta description
            desc_tag = soup.find('meta', {'name': 'description'})
            if desc_tag and desc_tag.get('content'):
                description = desc_tag['content'].strip()
                if len(description) > 100:  # If it's a reasonably sized description
                    return description
            
            # If meta description isn't available or is too short, try app description element
            desc_element = soup.find("div", class_=["bARER", "W4P4ne"])
            if desc_element:
                description = desc_element.get_text(strip=True)
                return description
            
            # Another fallback
            desc_element = soup.find("div", itemprop="description")
            if desc_element:
                description = desc_element.get_text(strip=True)
                return description
            
            log.warning(f"Could not find description for {app_link}")
            return "No description found"
            
        except TimeoutException:
            log.error(f"Timeout extracting description from {app_link}")
            return "Timeout retrieving description"
        except Exception as e:
            log.error(f"Error extracting description from {app_link}: {e}", exc_info=True)
            return "Error retrieving description"
        finally:
            # Don't close the driver here, as it might be reused for other operations
            pass 