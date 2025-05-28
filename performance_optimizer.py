"""
Performance Optimizer for MersenneHunter
Advanced optimizations for maximum computational efficiency
"""

import time
import threading
import psutil
import gc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque
import math

@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    tests_per_second: float = 0.0
    cache_hit_rate: float = 0.0
    thread_efficiency: float = 0.0
    optimization_score: float = 0.0
    
class AdvancedCacheManager:
    """Advanced caching system for computational results"""
    
    def __init__(self, max_size: int = 100000):
        """Initialize cache manager"""
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
    
    def get(self, key: Any) -> Optional[Any]:
        """Get cached value with LRU tracking"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hit_count += 1
                return self.cache[key]
            else:
                self.miss_count += 1
                return None
    
    def put(self, key: Any, value: Any) -> None:
        """Store value in cache with automatic cleanup"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used items"""
        if not self.access_times:
            return
        
        # Remove 10% of least recently used items
        evict_count = max(1, len(self.cache) // 10)
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        
        for key, _ in sorted_items[:evict_count]:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def clear(self) -> None:
        """Clear all cache data"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0

class MemoryOptimizer:
    """Memory usage optimization"""
    
    def __init__(self):
        """Initialize memory optimizer"""
        self.gc_threshold = 0.8  # 80% memory usage
        self.last_gc = time.time()
        self.gc_interval = 30  # seconds
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage"""
        memory_info = psutil.virtual_memory()
        current_usage = memory_info.percent / 100.0
        
        optimization_actions = []
        
        # Force garbage collection if needed
        if (current_usage > self.gc_threshold or 
            time.time() - self.last_gc > self.gc_interval):
            
            collected = gc.collect()
            optimization_actions.append(f"Garbage collected {collected} objects")
            self.last_gc = time.time()
        
        # Optimize large data structures
        if current_usage > 0.9:  # 90% memory usage
            optimization_actions.append("High memory usage detected")
        
        return {
            'memory_usage': current_usage,
            'actions_taken': optimization_actions,
            'available_mb': memory_info.available / (1024 * 1024)
        }

class CPUOptimizer:
    """CPU usage optimization"""
    
    def __init__(self):
        """Initialize CPU optimizer"""
        self.cpu_samples = deque(maxlen=10)
        self.optimal_thread_count = psutil.cpu_count()
    
    def calculate_optimal_threads(self, current_performance: float) -> int:
        """Calculate optimal thread count based on performance"""
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        self.cpu_samples.append(cpu_usage)
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples)
        
        # Adaptive thread calculation
        if avg_cpu < 50:  # CPU underutilized
            suggested_threads = min(cpu_count * 2, self.optimal_thread_count + 2)
        elif avg_cpu > 85:  # CPU overutilized
            suggested_threads = max(cpu_count // 2, self.optimal_thread_count - 2)
        else:  # Optimal range
            suggested_threads = self.optimal_thread_count
        
        # Performance-based adjustment
        if current_performance > 0:
            performance_factor = min(2.0, current_performance / 100.0)
            suggested_threads = int(suggested_threads * performance_factor)
        
        return max(1, min(suggested_threads, cpu_count * 4))
    
    def get_cpu_efficiency(self) -> float:
        """Calculate CPU efficiency score"""
        if not self.cpu_samples:
            return 0.0
        
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples)
        
        # Optimal range is 60-80% CPU usage
        if 60 <= avg_cpu <= 80:
            return 1.0
        elif avg_cpu < 60:
            return avg_cpu / 60.0
        else:  # avg_cpu > 80
            return max(0.1, (100 - avg_cpu) / 20.0)

class AlgorithmOptimizer:
    """Algorithm-level optimizations"""
    
    def __init__(self):
        """Initialize algorithm optimizer"""
        self.prime_cache = AdvancedCacheManager(50000)
        self.precomputed_values = {}
        self._precompute_common_values()
    
    def _precompute_common_values(self) -> None:
        """Precompute commonly used values"""
        # Precompute small prime factorials for optimization
        self.precomputed_values['small_primes'] = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
        ]
        
        # Precompute powers of 2 for efficiency
        self.precomputed_values['powers_of_2'] = {
            i: 2**i for i in range(1, 65)
        }
        
        # Precompute modular arithmetic tables
        self.precomputed_values['mod_tables'] = {}
        for prime in self.precomputed_values['small_primes']:
            self.precomputed_values['mod_tables'][prime] = {
                i: (i % prime) for i in range(prime * 2)
            }
    
    def optimized_lucas_lehmer_test(self, p: int) -> bool:
        """Optimized Lucas-Lehmer test with caching"""
        # Check cache first
        cached_result = self.prime_cache.get(p)
        if cached_result is not None:
            return cached_result
        
        # Quick elimination checks
        if p < 2:
            result = False
        elif p == 2:
            result = True
        elif p in self.precomputed_values['small_primes']:
            result = True
        elif any(p % prime == 0 for prime in self.precomputed_values['small_primes'][:10]):
            result = False
        else:
            # Perform optimized Lucas-Lehmer test
            result = self._perform_optimized_test(p)
        
        # Cache the result
        self.prime_cache.put(p, result)
        return result
    
    def _perform_optimized_test(self, p: int) -> bool:
        """Perform optimized Lucas-Lehmer test"""
        # Use precomputed powers where possible
        if p in self.precomputed_values['powers_of_2']:
            mersenne = self.precomputed_values['powers_of_2'][p] - 1
        else:
            mersenne = (1 << p) - 1  # Optimized 2^p - 1
        
        # Optimized Lucas-Lehmer sequence
        s = 4
        for _ in range(p - 2):
            s = ((s * s) - 2) % mersenne
            
            # Early termination optimization
            if s == 0:
                return True
        
        return s == 0
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get caching statistics"""
        return {
            'hit_rate': self.prime_cache.get_hit_rate(),
            'cache_size': len(self.prime_cache.cache),
            'max_size': self.prime_cache.max_size
        }

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.metrics_history = deque(maxlen=100)
        self.start_time = time.time()
        self.last_update = time.time()
        
    def update_metrics(self, tests_completed: int, primes_found: int, 
                      threads_active: int) -> PerformanceMetrics:
        """Update and calculate performance metrics"""
        current_time = time.time()
        time_delta = current_time - self.last_update
        
        # Calculate performance metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        tests_per_second = tests_completed / max(time_delta, 0.1)
        
        metrics = PerformanceMetrics(
            cpu_usage=cpu_percent,
            memory_usage=memory_info.percent,
            tests_per_second=tests_per_second,
            thread_efficiency=min(1.0, tests_per_second / max(threads_active, 1)),
            optimization_score=self._calculate_optimization_score(
                cpu_percent, memory_info.percent, tests_per_second
            )
        )
        
        self.metrics_history.append(metrics)
        self.last_update = current_time
        
        return metrics
    
    def _calculate_optimization_score(self, cpu_usage: float, 
                                    memory_usage: float, 
                                    tests_per_second: float) -> float:
        """Calculate overall optimization score"""
        # CPU efficiency (optimal range 60-80%)
        cpu_score = 1.0 if 60 <= cpu_usage <= 80 else max(0.1, 1.0 - abs(cpu_usage - 70) / 70)
        
        # Memory efficiency (under 80% is good)
        memory_score = max(0.1, (100 - memory_usage) / 100) if memory_usage < 90 else 0.1
        
        # Performance score (higher is better)
        performance_score = min(1.0, tests_per_second / 100.0)
        
        # Weighted average
        return (cpu_score * 0.4 + memory_score * 0.3 + performance_score * 0.3)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 samples
        
        return {
            'avg_cpu_usage': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            'avg_memory_usage': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            'avg_tests_per_second': sum(m.tests_per_second for m in recent_metrics) / len(recent_metrics),
            'avg_optimization_score': sum(m.optimization_score for m in recent_metrics) / len(recent_metrics),
            'uptime_hours': (time.time() - self.start_time) / 3600,
            'total_samples': len(self.metrics_history)
        }

class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self):
        """Initialize performance optimizer"""
        self.cache_manager = AdvancedCacheManager()
        self.memory_optimizer = MemoryOptimizer()
        self.cpu_optimizer = CPUOptimizer()
        self.algorithm_optimizer = AlgorithmOptimizer()
        self.performance_monitor = PerformanceMonitor()
        
        self.optimization_enabled = True
        self.auto_tune_threads = True
        
    def optimize_system(self, current_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive system optimization"""
        optimization_results = {
            'timestamp': time.time(),
            'optimizations_applied': [],
            'recommendations': []
        }
        
        if not self.optimization_enabled:
            return optimization_results
        
        # Memory optimization
        memory_result = self.memory_optimizer.optimize_memory()
        optimization_results['memory'] = memory_result
        
        # CPU optimization
        if self.auto_tune_threads and 'tests_per_second' in current_stats:
            optimal_threads = self.cpu_optimizer.calculate_optimal_threads(
                current_stats['tests_per_second']
            )
            optimization_results['recommended_threads'] = optimal_threads
            
            if optimal_threads != current_stats.get('threads_active', 0):
                optimization_results['recommendations'].append(
                    f"Adjust threads to {optimal_threads} for optimal performance"
                )
        
        # Algorithm optimization statistics
        cache_stats = self.algorithm_optimizer.get_cache_statistics()
        optimization_results['cache'] = cache_stats
        
        # Performance monitoring
        performance_metrics = self.performance_monitor.update_metrics(
            current_stats.get('candidates_tested', 0),
            current_stats.get('primes_found', 0),
            current_stats.get('threads_active', 1)
        )
        optimization_results['performance_metrics'] = performance_metrics
        
        # Performance summary
        performance_summary = self.performance_monitor.get_performance_summary()
        optimization_results['performance_summary'] = performance_summary
        
        return optimization_results
    
    def get_optimization_suggestions(self, current_performance: float) -> List[str]:
        """Get specific optimization suggestions"""
        suggestions = []
        
        if current_performance < 50:
            suggestions.append("Consider increasing thread count for better parallelization")
            suggestions.append("Enable aggressive caching for repeated calculations")
            suggestions.append("Optimize memory usage by clearing old results")
        
        cpu_efficiency = self.cpu_optimizer.get_cpu_efficiency()
        if cpu_efficiency < 0.7:
            suggestions.append(f"CPU efficiency is {cpu_efficiency:.1%}, consider thread adjustment")
        
        cache_hit_rate = self.algorithm_optimizer.prime_cache.get_hit_rate()
        if cache_hit_rate < 0.3:
            suggestions.append(f"Cache hit rate is {cache_hit_rate:.1%}, increase cache size")
        
        return suggestions

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()