"""
Bloom Filter implementation for fast negative lookups
"""

import hashlib
import math
from typing import Any

class BloomFilter:
    """
    Bloom Filter for efficient negative exponent lookups
    Provides O(1) lookup time with configurable false positive rate
    """
    
    def __init__(self, capacity: int = 1000000, error_rate: float = 0.001):
        """
        Initialize Bloom Filter
        
        Args:
            capacity: Expected number of elements
            error_rate: Desired false positive rate (0.0 to 1.0)
        """
        self.capacity = capacity
        self.error_rate = error_rate
        
        # Calculate optimal bit array size and hash function count
        self.bit_size = self._calculate_bit_size(capacity, error_rate)
        self.hash_count = self._calculate_hash_count(self.bit_size, capacity)
        
        # Initialize bit array
        self.bit_array = [False] * self.bit_size
        self.item_count = 0
    
    def _calculate_bit_size(self, capacity: int, error_rate: float) -> int:
        """Calculate optimal bit array size"""
        return int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
    
    def _calculate_hash_count(self, bit_size: int, capacity: int) -> int:
        """Calculate optimal number of hash functions"""
        return int((bit_size / capacity) * math.log(2))
    
    def _hash_functions(self, item: Any) -> list:
        """Generate hash values for an item"""
        # Convert item to string and encode
        item_str = str(item).encode('utf-8')
        
        # Generate multiple hash values using different methods
        hashes = []
        
        # MD5 hash
        md5_hash = hashlib.md5(item_str).hexdigest()
        hashes.append(int(md5_hash, 16) % self.bit_size)
        
        # SHA1 hash
        sha1_hash = hashlib.sha1(item_str).hexdigest()
        hashes.append(int(sha1_hash, 16) % self.bit_size)
        
        # Generate additional hashes by combining with salt values
        for i in range(2, self.hash_count):
            salt = f"salt{i}".encode('utf-8')
            combined = item_str + salt
            hash_value = hashlib.sha256(combined).hexdigest()
            hashes.append(int(hash_value, 16) % self.bit_size)
        
        return hashes[:self.hash_count]
    
    def add(self, item: Any):
        """
        Add an item to the Bloom Filter
        
        Args:
            item: Item to add (will be converted to string)
        """
        hash_values = self._hash_functions(item)
        
        for hash_val in hash_values:
            self.bit_array[hash_val] = True
        
        self.item_count += 1
    
    def contains(self, item: Any) -> bool:
        """
        Check if an item might be in the set
        
        Args:
            item: Item to check
            
        Returns:
            True if item might be in set (possible false positive)
            False if item is definitely not in set
        """
        hash_values = self._hash_functions(item)
        
        for hash_val in hash_values:
            if not self.bit_array[hash_val]:
                return False
        
        return True
    
    def clear(self):
        """Clear all items from the Bloom Filter"""
        self.bit_array = [False] * self.bit_size
        self.item_count = 0
    
    def estimated_false_positive_rate(self) -> float:
        """
        Calculate current estimated false positive rate
        
        Returns:
            Estimated false positive rate
        """
        if self.item_count == 0:
            return 0.0
        
        # Calculate probability that a bit is still 0
        prob_bit_zero = (1 - 1/self.bit_size) ** (self.hash_count * self.item_count)
        
        # False positive rate is probability all k bits are 1
        return (1 - prob_bit_zero) ** self.hash_count
    
    def get_statistics(self) -> dict:
        """
        Get Bloom Filter statistics
        
        Returns:
            Dictionary with filter statistics
        """
        bits_set = sum(self.bit_array)
        fill_ratio = bits_set / self.bit_size
        
        return {
            'capacity': self.capacity,
            'item_count': self.item_count,
            'bit_size': self.bit_size,
            'hash_count': self.hash_count,
            'bits_set': bits_set,
            'fill_ratio': fill_ratio,
            'configured_error_rate': self.error_rate,
            'estimated_error_rate': self.estimated_false_positive_rate()
        }
    
    def save_to_file(self, filename: str) -> bool:
        """
        Save Bloom Filter state to file
        
        Args:
            filename: File to save to
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            import pickle
            
            state = {
                'capacity': self.capacity,
                'error_rate': self.error_rate,
                'bit_size': self.bit_size,
                'hash_count': self.hash_count,
                'bit_array': self.bit_array,
                'item_count': self.item_count
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(state, f)
            
            return True
            
        except Exception as e:
            print(f"Error saving Bloom Filter: {e}")
            return False
    
    def load_from_file(self, filename: str) -> bool:
        """
        Load Bloom Filter state from file
        
        Args:
            filename: File to load from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            import pickle
            
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            
            self.capacity = state['capacity']
            self.error_rate = state['error_rate']
            self.bit_size = state['bit_size']
            self.hash_count = state['hash_count']
            self.bit_array = state['bit_array']
            self.item_count = state['item_count']
            
            return True
            
        except Exception as e:
            print(f"Error loading Bloom Filter: {e}")
            return False
    
    def union(self, other_filter: 'BloomFilter') -> 'BloomFilter':
        """
        Create union of two Bloom Filters
        
        Args:
            other_filter: Another Bloom Filter
            
        Returns:
            New Bloom Filter containing union
        """
        if (self.bit_size != other_filter.bit_size or 
            self.hash_count != other_filter.hash_count):
            raise ValueError("Bloom Filters must have same parameters for union")
        
        union_filter = BloomFilter(self.capacity, self.error_rate)
        union_filter.bit_size = self.bit_size
        union_filter.hash_count = self.hash_count
        union_filter.item_count = self.item_count + other_filter.item_count
        
        # Bitwise OR operation
        union_filter.bit_array = [
            self.bit_array[i] or other_filter.bit_array[i] 
            for i in range(self.bit_size)
        ]
        
        return union_filter
    
    def optimize_for_size(self) -> 'BloomFilter':
        """
        Create optimized Bloom Filter based on current item count
        
        Returns:
            New optimized Bloom Filter
        """
        if self.item_count == 0:
            return BloomFilter(self.capacity, self.error_rate)
        
        # Create new filter optimized for actual item count
        optimized_filter = BloomFilter(self.item_count, self.error_rate)
        
        return optimized_filter
