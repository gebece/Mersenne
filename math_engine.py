"""
Mathematical engine for primality testing and Mersenne number operations
"""

import hashlib
import random
from typing import Optional, List
from gpu_acceleration import gpu_accelerator

class MathEngine:
    """Core mathematical operations for Mersenne prime discovery"""
    
    def __init__(self, enable_gpu: bool = True):
        """Initialize mathematical engine"""
        self.small_primes = self._generate_small_primes(1000)
        self.enable_gpu = enable_gpu
        self.gpu_info = gpu_accelerator.get_gpu_info() if enable_gpu else None
        
        if enable_gpu and gpu_accelerator.preferred_backend:
            print(f"🚀 GPU Acceleration enabled: {gpu_accelerator.preferred_backend}")
        else:
            print("⚠️ Using CPU-only computations")
    
    def _generate_small_primes(self, limit: int) -> list:
        """Generate small primes using Sieve of Eratosthenes"""
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    def is_prime_basic(self, n: int) -> bool:
        """Basic primality test using small prime divisors"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        # Check divisibility by small primes
        for prime in self.small_primes:
            if prime * prime > n:
                break
            if n % prime == 0:
                return False
        
        return True
    
    def lucas_lehmer_test(self, p: int) -> bool:
        """
        GPU-accelerated Lucas-Lehmer primality test for Mersenne numbers
        Tests if 2^p - 1 is prime where p is prime
        
        Args:
            p: The exponent (must be prime)
            
        Returns:
            True if 2^p - 1 is prime, False otherwise
        """
        if p == 2:
            return True  # M2 = 3 is prime
        
        if not self.is_prime_basic(p):
            return False  # p must be prime for Mersenne prime
        
        try:
            # Try GPU acceleration first if enabled and suitable
            if self.enable_gpu and gpu_accelerator.preferred_backend and p <= 10000:
                gpu_result = self._lucas_lehmer_gpu(p)
                if gpu_result is not None:
                    return gpu_result
            
            # Fall back to CPU computation
            return self._lucas_lehmer_cpu(p)
            
        except Exception as e:
            raise RuntimeError(f"Lucas-Lehmer test failed for p={p}: {e}")
    
    def _lucas_lehmer_gpu(self, p: int) -> Optional[bool]:
        """GPU-accelerated Lucas-Lehmer test"""
        try:
            if gpu_accelerator.preferred_backend == 'cupy':
                return gpu_accelerator.lucas_lehmer_gpu_cupy(p)
            elif gpu_accelerator.preferred_backend == 'numba_cuda':
                return gpu_accelerator.lucas_lehmer_gpu_numba(p)
            elif gpu_accelerator.preferred_backend == 'opencl':
                return gpu_accelerator.lucas_lehmer_gpu_opencl(p)
            else:
                return None
        except Exception:
            return None  # Fall back to CPU
    
    def _lucas_lehmer_cpu(self, p: int) -> bool:
        """CPU-based Lucas-Lehmer test"""
        # Lucas-Lehmer sequence: s_0 = 4, s_i = (s_{i-1}^2 - 2) mod (2^p - 1)
        mersenne = (1 << p) - 1  # 2^p - 1
        s = 4
        
        for _ in range(p - 2):
            s = (s * s - 2) % mersenne
        
        return s == 0
    
    def miller_rabin_test(self, n: int, rounds: int = 10) -> bool:
        """
        Miller-Rabin probabilistic primality test
        
        Args:
            n: Number to test
            rounds: Number of test rounds (higher = more accurate)
            
        Returns:
            True if probably prime, False if composite
        """
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        
        # Write n-1 as d * 2^r
        r = 0
        d = n - 1
        while d % 2 == 0:
            d //= 2
            r += 1
        
        # Perform rounds of testing
        for _ in range(rounds):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    def solovay_strassen_test(self, n: int, rounds: int = 10) -> bool:
        """
        Solovay-Strassen probabilistic primality test
        
        Args:
            n: Number to test
            rounds: Number of test rounds
            
        Returns:
            True if probably prime, False if composite
        """
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        
        def jacobi_symbol(a: int, n: int) -> int:
            """Compute Jacobi symbol (a/n)"""
            if a == 0:
                return 0
            if a == 1:
                return 1
            
            result = 1
            while a != 0:
                while a % 2 == 0:
                    a //= 2
                    if n % 8 in (3, 5):
                        result = -result
                
                a, n = n, a
                if a % 4 == 3 and n % 4 == 3:
                    result = -result
                a %= n
            
            return result if n == 1 else 0
        
        for _ in range(rounds):
            a = random.randrange(2, n)
            x = pow(a, (n - 1) // 2, n)
            j = jacobi_symbol(a, n) % n
            
            if x != j:
                return False
        
        return True
    
    def trial_division(self, n: int, limit: Optional[int] = None) -> bool:
        """
        Trial division primality test
        
        Args:
            n: Number to test
            limit: Maximum divisor to test (default: sqrt(n))
            
        Returns:
            True if prime, False if composite
        """
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        if limit is None:
            limit = int(n**0.5) + 1
        
        for i in range(3, min(limit + 1, int(n**0.5) + 1), 2):
            if n % i == 0:
                return False
        
        return True
    
    def generate_result_hash(self, exponent: int, mersenne_number: int) -> str:
        """
        Generate SHA-256 hash for result verification
        
        Args:
            exponent: The Mersenne exponent
            mersenne_number: The Mersenne number (2^exponent - 1)
            
        Returns:
            SHA-256 hash as hex string
        """
        data = f"M{exponent}={mersenne_number}".encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    
    def estimate_mersenne_digits(self, exponent: int) -> int:
        """
        Estimate number of digits in a Mersenne number
        
        Args:
            exponent: The Mersenne exponent
            
        Returns:
            Estimated number of digits
        """
        import math
        return int(exponent * math.log10(2)) + 1
    
    def is_mersenne_form(self, n: int) -> tuple:
        """
        Check if a number is of Mersenne form (2^p - 1)
        
        Args:
            n: Number to check
            
        Returns:
            (is_mersenne, exponent) tuple
        """
        if n < 3:
            return False, 0
        
        # Check if n + 1 is a power of 2
        m = n + 1
        if m & (m - 1) == 0:  # Power of 2 check
            exponent = m.bit_length() - 1
            return True, exponent
        
        return False, 0
    
    def fermat_test(self, n: int, base: int = 2) -> bool:
        """
        Fermat primality test
        
        Args:
            n: Number to test
            base: Base for the test (default: 2)
            
        Returns:
            True if passes test, False otherwise
        """
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        return pow(base, n - 1, n) == 1
    
    def calculate_mersenne_probability(self, exponent: int) -> float:
        """
        Estimate probability that 2^exponent - 1 is prime
        Based on heuristic analysis of Mersenne prime distribution
        
        Args:
            exponent: The Mersenne exponent
            
        Returns:
            Estimated probability (0.0 to 1.0)
        """
        import math
        
        # Heuristic based on prime number theorem and Mersenne prime distribution
        # This is a rough approximation
        log_mersenne = exponent * math.log(2)
        probability = 2.0 / log_mersenne  # Approximation based on PNT
        
        # Adjust for known Mersenne prime patterns
        if exponent % 4 == 3:  # Slight preference for p ≡ 3 (mod 4)
            probability *= 1.1
        
        return min(probability, 1.0)
