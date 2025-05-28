"""
Database management for storing positive candidates and negative results
"""

import sqlite3
import threading
import time
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

class DatabaseManager:
    """Manages SQLite databases for regenerative prime search"""
    
    def __init__(self, positive_db: str = "positivos.db", negative_db: str = "negativos.db"):
        """
        Initialize database manager
        
        Args:
            positive_db: Path to positive candidates database
            negative_db: Path to negative results database
        """
        self.positive_db_path = positive_db
        self.negative_db_path = negative_db
        self.lock = threading.Lock()
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections with proper settings"""
        # Enable WAL mode and other optimizations
        self.positive_conn = sqlite3.connect(
            self.positive_db_path, 
            check_same_thread=False,
            isolation_level=None  # Autocommit mode
        )
        self.positive_conn.execute("PRAGMA journal_mode=WAL")
        self.positive_conn.execute("PRAGMA synchronous=NORMAL")
        self.positive_conn.execute("PRAGMA cache_size=10000")
        
        self.negative_conn = sqlite3.connect(
            self.negative_db_path,
            check_same_thread=False,
            isolation_level=None
        )
        self.negative_conn.execute("PRAGMA journal_mode=WAL")
        self.negative_conn.execute("PRAGMA synchronous=NORMAL")
        self.negative_conn.execute("PRAGMA cache_size=10000")
    
    @contextmanager
    def get_positive_cursor(self):
        """Get thread-safe cursor for positive database"""
        with self.lock:
            cursor = self.positive_conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()
    
    @contextmanager
    def get_negative_cursor(self):
        """Get thread-safe cursor for negative database"""
        with self.lock:
            cursor = self.negative_conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()
    
    def initialize(self):
        """Initialize database tables"""
        self._create_positive_tables()
        self._create_negative_tables()
        self._create_indexes()
    
    def _create_positive_tables(self):
        """Create tables for positive candidates"""
        with self.get_positive_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exponent INTEGER UNIQUE NOT NULL,
                    mersenne_number TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    tests_passed TEXT NOT NULL,
                    result_hash TEXT NOT NULL,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    digit_count INTEGER,
                    is_verified BOOLEAN DEFAULT FALSE
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_end TIMESTAMP,
                    candidates_found INTEGER DEFAULT 0,
                    total_tested INTEGER DEFAULT 0,
                    thread_count INTEGER,
                    search_mode TEXT
                )
            """)
    
    def _create_negative_tables(self):
        """Create tables for negative results"""
        with self.get_negative_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS failed_exponents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exponent INTEGER UNIQUE NOT NULL,
                    failure_reason TEXT NOT NULL,
                    tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    test_duration REAL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE DEFAULT CURRENT_DATE,
                    total_tested INTEGER DEFAULT 0,
                    failed_lucas_lehmer INTEGER DEFAULT 0,
                    failed_miller_rabin INTEGER DEFAULT 0,
                    failed_basic_prime INTEGER DEFAULT 0,
                    other_failures INTEGER DEFAULT 0
                )
            """)
    
    def _create_indexes(self):
        """Create indexes for optimal query performance"""
        with self.get_positive_cursor() as cursor:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_exponent ON candidates(exponent)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_confidence ON candidates(confidence_score DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_discovered ON candidates(discovered_at DESC)")
        
        with self.get_negative_cursor() as cursor:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_neg_exponent ON failed_exponents(exponent)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tested_at ON failed_exponents(tested_at DESC)")
    
    def store_positive_candidate(self, exponent: int, mersenne_number: str,
                                confidence_score: float, tests_passed: str,
                                result_hash: str) -> bool:
        """
        Store a positive candidate in the database
        
        Args:
            exponent: Mersenne exponent
            mersenne_number: The Mersenne number as string
            confidence_score: Confidence score (0.0 to 1.0)
            tests_passed: Comma-separated list of passed tests
            result_hash: SHA-256 hash for verification
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Calculate digit count
            digit_count = len(mersenne_number) if mersenne_number.isdigit() else 0
            
            with self.get_positive_cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO candidates 
                    (exponent, mersenne_number, confidence_score, tests_passed, result_hash, digit_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (exponent, mersenne_number, confidence_score, tests_passed, result_hash, digit_count))
            
            return True
            
        except sqlite3.Error as e:
            print(f"Error storing positive candidate: {e}")
            return False
    
    def store_negative_result(self, exponent: int, failure_reason: str,
                            test_duration: Optional[float] = None) -> bool:
        """
        Store a negative result in the database
        
        Args:
            exponent: Failed exponent
            failure_reason: Reason for failure
            test_duration: Time taken for the test
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            with self.get_negative_cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO failed_exponents 
                    (exponent, failure_reason, test_duration)
                    VALUES (?, ?, ?)
                """, (exponent, failure_reason, test_duration))
            
            return True
            
        except sqlite3.Error as e:
            print(f"Error storing negative result: {e}")
            return False
    
    def get_negative_exponents(self) -> List[int]:
        """Get all negative exponents for bloom filter initialization"""
        try:
            with self.get_negative_cursor() as cursor:
                cursor.execute("SELECT exponent FROM failed_exponents")
                return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error:
            return []
    
    def get_top_candidates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top candidates ordered by confidence score
        
        Args:
            limit: Maximum number of candidates to return
            
        Returns:
            List of candidate dictionaries
        """
        try:
            with self.get_positive_cursor() as cursor:
                cursor.execute("""
                    SELECT exponent, confidence_score, tests_passed, discovered_at, digit_count, result_hash
                    FROM candidates
                    ORDER BY confidence_score DESC, discovered_at DESC
                    LIMIT ?
                """, (limit,))
                
                candidates = []
                for row in cursor.fetchall():
                    candidates.append({
                        'exponent': row[0],
                        'confidence_score': row[1],
                        'tests_passed': row[2].split(',') if row[2] else [],
                        'discovered_at': row[3],
                        'digit_count': row[4],
                        'result_hash': row[5]
                    })
                
                return candidates
                
        except sqlite3.Error as e:
            print(f"Error retrieving candidates: {e}")
            return []
    
    def get_positive_count(self) -> int:
        """Get count of positive candidates"""
        try:
            with self.get_positive_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM candidates")
                return cursor.fetchone()[0]
        except sqlite3.Error:
            return 0
    
    def get_negative_count(self) -> int:
        """Get count of negative results"""
        try:
            with self.get_negative_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM failed_exponents")
                return cursor.fetchone()[0]
        except sqlite3.Error:
            return 0
    
    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get comprehensive statistics from both databases"""
        stats = {
            'positive_candidates': self.get_positive_count(),
            'negative_results': self.get_negative_count(),
            'highest_confidence': 0.0,
            'largest_candidate': 0,
            'recent_discoveries': 0
        }
        
        try:
            # Get highest confidence score
            with self.get_positive_cursor() as cursor:
                cursor.execute("SELECT MAX(confidence_score) FROM candidates")
                result = cursor.fetchone()
                if result and result[0]:
                    stats['highest_confidence'] = result[0]
                
                # Get largest candidate by exponent
                cursor.execute("SELECT MAX(exponent) FROM candidates")
                result = cursor.fetchone()
                if result and result[0]:
                    stats['largest_candidate'] = result[0]
                
                # Get recent discoveries (last 24 hours)
                cursor.execute("""
                    SELECT COUNT(*) FROM candidates 
                    WHERE discovered_at > datetime('now', '-1 day')
                """)
                result = cursor.fetchone()
                if result:
                    stats['recent_discoveries'] = result[0]
        
        except sqlite3.Error as e:
            print(f"Error getting statistics: {e}")
        
        return stats
    
    def cleanup_old_negatives(self, days_old: int = 30) -> int:
        """
        Clean up old negative results to manage database size
        
        Args:
            days_old: Remove negatives older than this many days
            
        Returns:
            Number of records removed
        """
        try:
            with self.get_negative_cursor() as cursor:
                cursor.execute("""
                    DELETE FROM failed_exponents 
                    WHERE tested_at < datetime('now', '-{} days')
                """.format(days_old))
                return cursor.rowcount
        except sqlite3.Error as e:
            print(f"Error cleaning up negatives: {e}")
            return 0
    
    def export_candidates(self, filename: str, min_confidence: float = 0.95) -> bool:
        """
        Export high-confidence candidates to a file
        
        Args:
            filename: Output filename
            min_confidence: Minimum confidence score to export
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            with self.get_positive_cursor() as cursor:
                cursor.execute("""
                    SELECT exponent, confidence_score, tests_passed, result_hash, discovered_at
                    FROM candidates
                    WHERE confidence_score >= ?
                    ORDER BY confidence_score DESC
                """, (min_confidence,))
                
                with open(filename, 'w') as f:
                    f.write("Exponent,Confidence,Tests_Passed,Hash,Discovered_At\n")
                    for row in cursor.fetchall():
                        f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")
                
                return True
                
        except (sqlite3.Error, IOError) as e:
            print(f"Error exporting candidates: {e}")
            return False
    
    def close(self):
        """Close database connections"""
        try:
            self.positive_conn.close()
            self.negative_conn.close()
        except sqlite3.Error as e:
            print(f"Error closing databases: {e}")
