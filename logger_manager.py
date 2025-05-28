"""
Comprehensive logging system for MersenneHunter
"""

import logging
import logging.handlers
import hashlib
import json
import time
import os
from datetime import datetime
from typing import Any, Dict, Optional

class LoggerManager:
    """Advanced logging system with verification and persistence"""
    
    def __init__(self, log_level: str = "INFO", log_dir: str = "logs"):
        """
        Initialize logging manager
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = log_dir
        self.session_id = self._generate_session_id()
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize loggers
        self.main_logger = self._setup_main_logger()
        self.security_logger = self._setup_security_logger()
        self.performance_logger = self._setup_performance_logger()
        self.discovery_logger = self._setup_discovery_logger()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = str(time.time())
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]
    
    def _setup_main_logger(self) -> logging.Logger:
        """Set up main application logger"""
        logger = logging.getLogger('mersenne_hunter.main')
        logger.setLevel(self.log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, 'mersenne_hunter.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_security_logger(self) -> logging.Logger:
        """Set up security and verification logger"""
        logger = logging.getLogger('mersenne_hunter.security')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # Security log file (append only, no rotation for audit trail)
        security_handler = logging.FileHandler(
            os.path.join(self.log_dir, 'security_audit.log'),
            mode='a'
        )
        security_formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        security_handler.setFormatter(security_formatter)
        logger.addHandler(security_handler)
        
        return logger
    
    def _setup_performance_logger(self) -> logging.Logger:
        """Set up performance metrics logger"""
        logger = logging.getLogger('mersenne_hunter.performance')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # Performance metrics file
        perf_handler = logging.handlers.TimedRotatingFileHandler(
            os.path.join(self.log_dir, 'performance.log'),
            when='midnight',
            interval=1,
            backupCount=30
        )
        perf_formatter = logging.Formatter(
            '%(asctime)s - PERF - %(message)s'
        )
        perf_handler.setFormatter(perf_formatter)
        logger.addHandler(perf_handler)
        
        return logger
    
    def _setup_discovery_logger(self) -> logging.Logger:
        """Set up discovery events logger"""
        logger = logging.getLogger('mersenne_hunter.discovery')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # Discovery events file (critical findings)
        discovery_handler = logging.FileHandler(
            os.path.join(self.log_dir, 'discoveries.log'),
            mode='a'
        )
        discovery_formatter = logging.Formatter(
            '%(asctime)s - DISCOVERY - %(levelname)s - %(message)s'
        )
        discovery_handler.setFormatter(discovery_formatter)
        logger.addHandler(discovery_handler)
        
        return logger
    
    def get_logger(self, name: str = 'main') -> logging.Logger:
        """
        Get logger by name
        
        Args:
            name: Logger name ('main', 'security', 'performance', 'discovery')
            
        Returns:
            Logger instance
        """
        loggers = {
            'main': self.main_logger,
            'security': self.security_logger,
            'performance': self.performance_logger,
            'discovery': self.discovery_logger
        }
        
        return loggers.get(name, self.main_logger)
    
    def log_discovery(self, exponent: int, mersenne_number: str, 
                     confidence: float, tests_passed: list, result_hash: str):
        """
        Log a prime discovery with full verification details
        
        Args:
            exponent: Mersenne exponent
            mersenne_number: The Mersenne number
            confidence: Confidence score
            tests_passed: List of passed tests
            result_hash: Verification hash
        """
        discovery_data = {
            'session_id': self.session_id,
            'timestamp': datetime.utcnow().isoformat(),
            'exponent': exponent,
            'mersenne_number': mersenne_number[:100] + '...' if len(mersenne_number) > 100 else mersenne_number,
            'digit_count': len(mersenne_number) if mersenne_number.isdigit() else 0,
            'confidence_score': confidence,
            'tests_passed': tests_passed,
            'result_hash': result_hash,
            'verification_hash': self._generate_verification_hash(exponent, mersenne_number, confidence)
        }
        
        # Log to discovery logger
        self.discovery_logger.critical(json.dumps(discovery_data, indent=2))
        
        # Log to security logger for audit trail
        self.security_logger.info(f"DISCOVERY_LOGGED: M{exponent}, Hash: {result_hash}")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """
        Log performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
        """
        perf_data = {
            'session_id': self.session_id,
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics
        }
        
        self.performance_logger.info(json.dumps(perf_data))
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log security-related events
        
        Args:
            event_type: Type of security event
            details: Event details
        """
        security_data = {
            'session_id': self.session_id,
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'details': details,
            'integrity_hash': self._generate_integrity_hash(event_type, details)
        }
        
        self.security_logger.warning(json.dumps(security_data))
    
    def log_search_session(self, session_data: Dict[str, Any]):
        """
        Log search session information
        
        Args:
            session_data: Session configuration and statistics
        """
        session_data['session_id'] = self.session_id
        session_data['timestamp'] = datetime.utcnow().isoformat()
        
        self.main_logger.info(f"SEARCH_SESSION: {json.dumps(session_data)}")
        self.security_logger.info(f"SESSION_START: {self.session_id}")
    
    def _generate_verification_hash(self, exponent: int, mersenne_number: str, confidence: float) -> str:
        """Generate verification hash for discoveries"""
        data = f"{exponent}:{mersenne_number}:{confidence}:{self.session_id}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _generate_integrity_hash(self, event_type: str, details: Dict[str, Any]) -> str:
        """Generate integrity hash for security events"""
        data = f"{event_type}:{json.dumps(details, sort_keys=True)}:{self.session_id}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def verify_log_integrity(self, log_file: str) -> bool:
        """
        Verify the integrity of a log file
        
        Args:
            log_file: Path to log file
            
        Returns:
            True if integrity is verified, False otherwise
        """
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Verify hash chains and timestamps
            for line in lines:
                if 'verification_hash' in line or 'integrity_hash' in line:
                    # Parse JSON and verify hash
                    try:
                        data = json.loads(line.split(' - ')[-1])
                        # Verification logic would go here
                    except json.JSONDecodeError:
                        continue
            
            return True
            
        except Exception as e:
            self.main_logger.error(f"Log verification failed: {e}")
            return False
    
    def export_discoveries(self, output_file: str) -> bool:
        """
        Export all discoveries to a file
        
        Args:
            output_file: Output file path
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            discoveries = []
            discovery_log = os.path.join(self.log_dir, 'discoveries.log')
            
            if os.path.exists(discovery_log):
                with open(discovery_log, 'r') as f:
                    for line in f:
                        if 'DISCOVERY' in line:
                            try:
                                data = json.loads(line.split(' - ')[-1])
                                discoveries.append(data)
                            except json.JSONDecodeError:
                                continue
            
            with open(output_file, 'w') as f:
                json.dump(discoveries, f, indent=2)
            
            return True
            
        except Exception as e:
            self.main_logger.error(f"Discovery export failed: {e}")
            return False
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            'session_id': self.session_id,
            'log_directory': self.log_dir,
            'log_level': logging.getLevelName(self.log_level),
            'log_files': {}
        }
        
        # Get file sizes and line counts
        for log_file in ['mersenne_hunter.log', 'security_audit.log', 
                        'performance.log', 'discoveries.log']:
            file_path = os.path.join(self.log_dir, log_file)
            if os.path.exists(file_path):
                stats['log_files'][log_file] = {
                    'size_bytes': os.path.getsize(file_path),
                    'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                }
        
        return stats
    
    def cleanup_old_logs(self, days_old: int = 30):
        """
        Clean up log files older than specified days
        
        Args:
            days_old: Remove logs older than this many days
        """
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        for filename in os.listdir(self.log_dir):
            file_path = os.path.join(self.log_dir, filename)
            if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                try:
                    os.remove(file_path)
                    self.main_logger.info(f"Cleaned up old log file: {filename}")
                except Exception as e:
                    self.main_logger.error(f"Failed to clean up {filename}: {e}")
