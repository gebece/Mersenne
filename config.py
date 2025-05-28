"""
Configuration settings for MersenneHunter
"""

import os
from typing import Dict, Any

class Config:
    """Configuration management for MersenneHunter"""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        # Search parameters
        'DEFAULT_THREAD_COUNT': 10,
        'MAX_THREAD_COUNT': 1000000,  # 1 million threads max
        'MIN_THREAD_COUNT': 10,
        'DEFAULT_BATCH_SIZE': 100,
        'DEFAULT_SEARCH_MODE': 'sequential',
        'START_EXPONENT': 82589933,  # After M83123485
        
        # Database settings
        'POSITIVE_DB_PATH': 'positivos.db',
        'NEGATIVE_DB_PATH': 'negativos.db',
        'DB_CACHE_SIZE': 10000,
        'DB_SYNCHRONOUS_MODE': 'NORMAL',
        'DB_JOURNAL_MODE': 'WAL',
        
        # Bloom Filter settings
        'BLOOM_FILTER_CAPACITY': 10000000,
        'BLOOM_FILTER_ERROR_RATE': 0.001,
        'BLOOM_FILTER_SAVE_INTERVAL': 3600,  # Save every hour
        
        # Logging settings
        'LOG_LEVEL': 'INFO',
        'LOG_DIRECTORY': 'logs',
        'LOG_ROTATION_SIZE': 10 * 1024 * 1024,  # 10MB
        'LOG_BACKUP_COUNT': 5,
        'LOG_RETENTION_DAYS': 30,
        
        # Performance settings
        'WORK_QUEUE_SIZE': 1000,
        'RESULT_QUEUE_SIZE': 1000,
        'WORKER_TIMEOUT': 1.0,
        'SHUTDOWN_TIMEOUT': 5.0,
        
        # Mathematical settings
        'MILLER_RABIN_ROUNDS': 10,
        'SMALL_PRIME_LIMIT': 1000,
        'CONFIDENCE_THRESHOLD': 0.99,
        'STRONG_CANDIDATE_THRESHOLD': 0.95,
        
        # Network settings (for future distributed features)
        'COORDINATOR_PORT': 8000,
        'WORKER_PORT_RANGE': (8001, 8100),
        'HEARTBEAT_INTERVAL': 30,
        'NODE_TIMEOUT': 120,
        
        # Security settings
        'ENABLE_RESULT_VERIFICATION': True,
        'HASH_ALGORITHM': 'sha256',
        'AUDIT_LOG_RETENTION': 365,  # days
        
        # Web interface settings
        'WEB_INTERFACE_HOST': '0.0.0.0',
        'WEB_INTERFACE_PORT': 5000,
        'WEB_DEBUG_MODE': False,
        'WEB_TEMPLATE_FOLDER': 'templates',
        'WEB_STATIC_FOLDER': 'static',
        
        # Optimization settings
        'ENABLE_GPU_ACCELERATION': True,
        'GPU_DEVICE_ID': 0,
        'GPU_BATCH_SIZE': 100,
        'GPU_MEMORY_LIMIT': 4.0,  # GB
        'ENABLE_QUANTUM_SIMULATION': False,
        'QUANTUM_BACKEND': 'qasm_simulator',
        
        # Discovery rewards (for tracking purposes)
        'REWARD_MILLIONS_DIGITS': 1500,  # USD
        'REWARD_BILLIONS_DIGITS': 150000,  # USD
        'DIGIT_THRESHOLD_MILLIONS': 1000000,
        'DIGIT_THRESHOLD_BILLIONS': 1000000000,
    }
    
    def __init__(self, config_file: str = None):
        """
        Initialize configuration
        
        Args:
            config_file: Optional configuration file path
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from environment variables
        self._load_from_environment()
        
        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'MERSENNE_THREAD_COUNT': 'DEFAULT_THREAD_COUNT',
            'MERSENNE_BATCH_SIZE': 'DEFAULT_BATCH_SIZE',
            'MERSENNE_SEARCH_MODE': 'DEFAULT_SEARCH_MODE',
            'MERSENNE_START_EXPONENT': 'START_EXPONENT',
            'MERSENNE_LOG_LEVEL': 'LOG_LEVEL',
            'MERSENNE_LOG_DIR': 'LOG_DIRECTORY',
            'MERSENNE_POSITIVE_DB': 'POSITIVE_DB_PATH',
            'MERSENNE_NEGATIVE_DB': 'NEGATIVE_DB_PATH',
            'MERSENNE_WEB_PORT': 'WEB_INTERFACE_PORT',
            'MERSENNE_WEB_HOST': 'WEB_INTERFACE_HOST',
            'MERSENNE_BLOOM_CAPACITY': 'BLOOM_FILTER_CAPACITY',
            'MERSENNE_BLOOM_ERROR_RATE': 'BLOOM_FILTER_ERROR_RATE',
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert to appropriate type
                if config_key in ['DEFAULT_THREAD_COUNT', 'DEFAULT_BATCH_SIZE', 
                                'START_EXPONENT', 'WEB_INTERFACE_PORT', 'BLOOM_FILTER_CAPACITY']:
                    self.config[config_key] = int(value)
                elif config_key in ['BLOOM_FILTER_ERROR_RATE']:
                    self.config[config_key] = float(value)
                else:
                    self.config[config_key] = value
    
    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            import json
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Update config with file values
            for key, value in file_config.items():
                if key in self.config:
                    self.config[key] = value
                    
        except Exception as e:
            print(f"Warning: Failed to load config file {config_file}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value
    
    def validate(self) -> list:
        """
        Validate configuration values
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate thread count
        thread_count = self.get('DEFAULT_THREAD_COUNT')
        min_threads = self.get('MIN_THREAD_COUNT')
        max_threads = self.get('MAX_THREAD_COUNT')
        
        if not (min_threads <= thread_count <= max_threads):
            errors.append(f"Thread count {thread_count} must be between {min_threads} and {max_threads}")
        
        # Validate bloom filter settings
        bloom_capacity = self.get('BLOOM_FILTER_CAPACITY')
        bloom_error_rate = self.get('BLOOM_FILTER_ERROR_RATE')
        
        if bloom_capacity < 1000:
            errors.append("Bloom filter capacity must be at least 1000")
        
        if not (0.0 < bloom_error_rate < 1.0):
            errors.append("Bloom filter error rate must be between 0.0 and 1.0")
        
        # Validate search mode
        search_mode = self.get('DEFAULT_SEARCH_MODE')
        valid_modes = ['sequential', 'random', 'mixed']
        if search_mode not in valid_modes:
            errors.append(f"Search mode must be one of: {', '.join(valid_modes)}")
        
        # Validate log level
        log_level = self.get('LOG_LEVEL')
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if log_level not in valid_levels:
            errors.append(f"Log level must be one of: {', '.join(valid_levels)}")
        
        # Validate start exponent
        start_exponent = self.get('START_EXPONENT')
        if start_exponent < 2:
            errors.append("Start exponent must be at least 2")
        
        return errors
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database-specific configuration"""
        return {
            'positive_db_path': self.get('POSITIVE_DB_PATH'),
            'negative_db_path': self.get('NEGATIVE_DB_PATH'),
            'cache_size': self.get('DB_CACHE_SIZE'),
            'synchronous_mode': self.get('DB_SYNCHRONOUS_MODE'),
            'journal_mode': self.get('DB_JOURNAL_MODE'),
        }
    
    def get_bloom_filter_config(self) -> Dict[str, Any]:
        """Get Bloom filter configuration"""
        return {
            'capacity': self.get('BLOOM_FILTER_CAPACITY'),
            'error_rate': self.get('BLOOM_FILTER_ERROR_RATE'),
            'save_interval': self.get('BLOOM_FILTER_SAVE_INTERVAL'),
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'log_level': self.get('LOG_LEVEL'),
            'log_directory': self.get('LOG_DIRECTORY'),
            'rotation_size': self.get('LOG_ROTATION_SIZE'),
            'backup_count': self.get('LOG_BACKUP_COUNT'),
            'retention_days': self.get('LOG_RETENTION_DAYS'),
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration"""
        return {
            'thread_count': self.get('DEFAULT_THREAD_COUNT'),
            'batch_size': self.get('DEFAULT_BATCH_SIZE'),
            'work_queue_size': self.get('WORK_QUEUE_SIZE'),
            'result_queue_size': self.get('RESULT_QUEUE_SIZE'),
            'worker_timeout': self.get('WORKER_TIMEOUT'),
            'shutdown_timeout': self.get('SHUTDOWN_TIMEOUT'),
        }
    
    def get_mathematical_config(self) -> Dict[str, Any]:
        """Get mathematical computation configuration"""
        return {
            'miller_rabin_rounds': self.get('MILLER_RABIN_ROUNDS'),
            'small_prime_limit': self.get('SMALL_PRIME_LIMIT'),
            'confidence_threshold': self.get('CONFIDENCE_THRESHOLD'),
            'strong_candidate_threshold': self.get('STRONG_CANDIDATE_THRESHOLD'),
        }
    
    def get_web_config(self) -> Dict[str, Any]:
        """Get web interface configuration"""
        return {
            'host': self.get('WEB_INTERFACE_HOST'),
            'port': self.get('WEB_INTERFACE_PORT'),
            'debug': self.get('WEB_DEBUG_MODE'),
            'template_folder': self.get('WEB_TEMPLATE_FOLDER'),
            'static_folder': self.get('WEB_STATIC_FOLDER'),
        }
    
    def save_to_file(self, config_file: str) -> bool:
        """
        Save current configuration to file
        
        Args:
            config_file: File path to save to
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            import json
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2, sort_keys=True)
            return True
        except Exception as e:
            print(f"Failed to save config to {config_file}: {e}")
            return False
    
    def get_reward_info(self, digit_count: int) -> Dict[str, Any]:
        """
        Get reward information for a given digit count
        
        Args:
            digit_count: Number of digits in the discovered prime
            
        Returns:
            Dictionary with reward information
        """
        millions_threshold = self.get('DIGIT_THRESHOLD_MILLIONS')
        billions_threshold = self.get('DIGIT_THRESHOLD_BILLIONS')
        
        if digit_count >= billions_threshold:
            return {
                'eligible': True,
                'tier': 'billions',
                'amount': self.get('REWARD_BILLIONS_DIGITS'),
                'currency': 'USD'
            }
        elif digit_count >= millions_threshold:
            return {
                'eligible': True,
                'tier': 'millions', 
                'amount': self.get('REWARD_MILLIONS_DIGITS'),
                'currency': 'USD'
            }
        else:
            return {
                'eligible': False,
                'tier': 'none',
                'amount': 0,
                'currency': 'USD'
            }

# Global configuration instance
config = Config()
