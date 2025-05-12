"""
Service for caching analysis results.
"""
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib
import pandas as pd

from models.app_data import AnalysisResults, AppDetails

class CacheService:
    def __init__(self, cache_dir: str = "cache"):
        """Initialize the cache service with a cache directory."""
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self) -> None:
        """Ensure the cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _generate_cache_key(self, app_id: str, analysis_type: str) -> str:
        """Generate a unique cache key for an app and analysis type."""
        key = f"{app_id}_{analysis_type}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the full path for a cache file."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _get_df_cache_path(self, cache_key: str) -> str:
        """Get the full path for a DataFrame cache file."""
        return os.path.join(self.cache_dir, f"{cache_key}.csv")
    
    def _is_cache_valid(self, cache_data: Dict[str, Any], max_age_days: int = 7) -> bool:
        """Check if the cache is still valid based on its age."""
        if 'timestamp' not in cache_data:
            return False
        
        cache_time = datetime.fromisoformat(cache_data['timestamp'])
        max_age = timedelta(days=max_age_days)
        return datetime.now() - cache_time < max_age
    
    def get_cached_analysis(self, app_id: str, analysis_type: str, max_age_days: int = 7) -> Optional[Dict[str, Any]]:
        """Get cached analysis results if they exist and are valid."""
        cache_key = self._generate_cache_key(app_id, analysis_type)
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            if not self._is_cache_valid(cache_data, max_age_days):
                return None
            
            return cache_data['data']
            
        except Exception:
            return None
    
    def cache_analysis(self, app_id: str, analysis_type: str, data: Dict[str, Any]) -> None:
        """Cache analysis results."""
        cache_key = self._generate_cache_key(app_id, analysis_type)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
        except Exception:
            pass  # Silently fail if caching fails
    
    def get_cached_dataframe(self, app_id: str, analysis_type: str, max_age_days: int = 7) -> Optional[pd.DataFrame]:
        """Get cached DataFrame if it exists and is valid."""
        cache_key = self._generate_cache_key(app_id, analysis_type)
        cache_path = self._get_df_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            # Read the DataFrame
            df = pd.read_csv(cache_path)
            
            # Check if the cache is still valid
            if 'timestamp' in df.attrs and self._is_cache_valid({'timestamp': df.attrs['timestamp']}, max_age_days):
                return df
            
            return None
            
        except Exception:
            return None
    
    def cache_dataframe(self, app_id: str, analysis_type: str, df: pd.DataFrame) -> None:
        """Cache a DataFrame."""
        cache_key = self._generate_cache_key(app_id, analysis_type)
        cache_path = self._get_df_cache_path(cache_key)
        
        try:
            # Store timestamp in DataFrame attributes
            df.attrs['timestamp'] = datetime.now().isoformat()
            
            # Save the DataFrame as CSV
            df.to_csv(cache_path, index=False)
        except Exception:
            pass  # Silently fail if caching fails
    
    def clear_cache(self, app_id: Optional[str] = None) -> None:
        """Clear cache for a specific app or all apps."""
        if app_id:
            # Clear specific app cache
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(hashlib.md5(app_id.encode()).hexdigest()):
                    os.remove(os.path.join(self.cache_dir, filename))
        else:
            # Clear all cache
            for filename in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, filename)) 