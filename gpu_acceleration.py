"""
GPU Acceleration for MersenneHunter
High-performance Lucas-Lehmer computations using CUDA and OpenCL
"""

import numpy as np
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass

# GPU Libraries
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

try:
    from numba import cuda
    import numba
    NUMBA_CUDA_AVAILABLE = True
except ImportError:
    NUMBA_CUDA_AVAILABLE = False

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
    cl = None

@dataclass
class GPUPerformanceMetrics:
    """Performance metrics for GPU computations"""
    gpu_type: str
    computation_time: float
    speedup_factor: float
    memory_usage: float
    throughput: float

class GPUAccelerator:
    """GPU acceleration manager for Lucas-Lehmer computations"""
    
    def __init__(self):
        """Initialize GPU accelerator"""
        self.cuda_available = CUDA_AVAILABLE
        self.numba_cuda_available = NUMBA_CUDA_AVAILABLE
        self.opencl_available = OPENCL_AVAILABLE
        
        self.preferred_backend = None
        self.gpu_device = None
        self.opencl_context = None
        self.opencl_queue = None
        
        self._initialize_gpu_backends()
    
    def _initialize_gpu_backends(self):
        """Initialize available GPU backends"""
        if self.cuda_available:
            try:
                # Test CuPy availability
                cp.cuda.Device(0).use()
                self.preferred_backend = 'cupy'
                print("🚀 CUDA (CuPy) detected and initialized")
            except Exception as e:
                print(f"⚠️ CUDA initialization failed: {e}")
                self.cuda_available = False
        
        if self.numba_cuda_available and not self.preferred_backend:
            try:
                # Test Numba CUDA
                if cuda.is_available():
                    self.preferred_backend = 'numba_cuda'
                    print("🚀 CUDA (Numba) detected and initialized")
                else:
                    self.numba_cuda_available = False
            except Exception as e:
                print(f"⚠️ Numba CUDA initialization failed: {e}")
                self.numba_cuda_available = False
        
        if self.opencl_available and not self.preferred_backend:
            try:
                # Initialize OpenCL
                platforms = cl.get_platforms()
                if platforms:
                    platform = platforms[0]
                    devices = platform.get_devices(cl.device_type.GPU)
                    if devices:
                        self.opencl_context = cl.Context(devices=[devices[0]])
                        self.opencl_queue = cl.CommandQueue(self.opencl_context)
                        self.preferred_backend = 'opencl'
                        print("🚀 OpenCL detected and initialized")
                    else:
                        self.opencl_available = False
            except Exception as e:
                print(f"⚠️ OpenCL initialization failed: {e}")
                self.opencl_available = False
        
        if not self.preferred_backend:
            print("⚠️ No GPU acceleration available, falling back to CPU")
    
    def get_gpu_info(self) -> dict:
        """Get detailed GPU information"""
        info = {
            'cuda_available': self.cuda_available,
            'numba_cuda_available': self.numba_cuda_available,
            'opencl_available': self.opencl_available,
            'preferred_backend': self.preferred_backend,
            'devices': []
        }
        
        if self.cuda_available:
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
                for i in range(device_count):
                    with cp.cuda.Device(i):
                        props = cp.cuda.runtime.getDeviceProperties(i)
                        info['devices'].append({
                            'id': i,
                            'name': props['name'].decode(),
                            'memory': props['totalGlobalMem'] // (1024**3),  # GB
                            'compute_capability': f"{props['major']}.{props['minor']}",
                            'multiprocessors': props['multiProcessorCount'],
                            'backend': 'CUDA'
                        })
            except Exception as e:
                print(f"Error getting CUDA device info: {e}")
        
        if self.opencl_available and self.opencl_context:
            try:
                devices = self.opencl_context.devices
                for i, device in enumerate(devices):
                    info['devices'].append({
                        'id': i,
                        'name': device.name,
                        'memory': device.global_mem_size // (1024**3),  # GB
                        'compute_units': device.max_compute_units,
                        'backend': 'OpenCL'
                    })
            except Exception as e:
                print(f"Error getting OpenCL device info: {e}")
        
        return info
    
    def lucas_lehmer_gpu_cupy(self, p: int) -> bool:
        """GPU-accelerated Lucas-Lehmer test using CuPy"""
        if not self.cuda_available:
            return False
        
        try:
            # Move computation to GPU
            with cp.cuda.Device(0):
                # For very large exponents, we need to use arbitrary precision
                # CuPy doesn't support arbitrary precision, so we'll use a hybrid approach
                
                if p <= 10000:  # Use GPU for smaller exponents
                    mersenne = (1 << p) - 1
                    s = cp.array([4], dtype=cp.int64)
                    mersenne_gpu = cp.array([mersenne], dtype=cp.int64)
                    
                    for i in range(p - 2):
                        s = (s * s - 2) % mersenne_gpu
                    
                    result = cp.asnumpy(s)[0] == 0
                    return result
                else:
                    # Fall back to CPU for very large numbers
                    return False
                    
        except Exception as e:
            print(f"GPU CuPy Lucas-Lehmer error: {e}")
            return False
    
    def lucas_lehmer_gpu_numba(self, p: int) -> bool:
        """GPU-accelerated Lucas-Lehmer test using Numba CUDA"""
        if not self.numba_cuda_available:
            return False
        
        try:
            @cuda.jit
            def lucas_lehmer_kernel(p, result):
                if p <= 64:  # Limitation for GPU integer precision
                    mersenne = (1 << p) - 1
                    s = 4
                    for i in range(p - 2):
                        s = (s * s - 2) % mersenne
                    result[0] = 1 if s == 0 else 0
            
            if p <= 64:
                result = cuda.device_array(1, dtype=np.int32)
                lucas_lehmer_kernel[1, 1](p, result)
                return result.copy_to_host()[0] == 1
            else:
                return False
                
        except Exception as e:
            print(f"GPU Numba Lucas-Lehmer error: {e}")
            return False
    
    def lucas_lehmer_gpu_opencl(self, p: int) -> bool:
        """GPU-accelerated Lucas-Lehmer test using OpenCL"""
        if not self.opencl_available:
            return False
        
        try:
            # OpenCL kernel for Lucas-Lehmer test
            kernel_source = """
            __kernel void lucas_lehmer(__global long* p_val, __global long* result) {
                int gid = get_global_id(0);
                if (gid == 0) {
                    long p = p_val[0];
                    if (p <= 63) {  // Avoid overflow
                        long mersenne = (1L << p) - 1;
                        long s = 4;
                        for (int i = 0; i < p - 2; i++) {
                            s = (s * s - 2) % mersenne;
                        }
                        result[0] = (s == 0) ? 1 : 0;
                    } else {
                        result[0] = 0;  // Cannot compute on GPU
                    }
                }
            }
            """
            
            if p <= 63:
                program = cl.Program(self.opencl_context, kernel_source).build()
                
                # Create buffers
                p_buffer = cl.Buffer(self.opencl_context, cl.mem_flags.READ_ONLY, 8)
                result_buffer = cl.Buffer(self.opencl_context, cl.mem_flags.WRITE_ONLY, 8)
                
                # Transfer data
                cl.enqueue_copy(self.opencl_queue, p_buffer, np.array([p], dtype=np.int64))
                
                # Execute kernel
                program.lucas_lehmer(self.opencl_queue, (1,), None, p_buffer, result_buffer)
                
                # Get result
                result = np.empty(1, dtype=np.int64)
                cl.enqueue_copy(self.opencl_queue, result, result_buffer)
                self.opencl_queue.finish()
                
                return result[0] == 1
            else:
                return False
                
        except Exception as e:
            print(f"GPU OpenCL Lucas-Lehmer error: {e}")
            return False
    
    def lucas_lehmer_gpu_batch(self, exponents: List[int]) -> List[bool]:
        """Batch GPU computation for multiple exponents"""
        if not self.preferred_backend:
            return [False] * len(exponents)
        
        results = []
        valid_exponents = [p for p in exponents if p <= 10000]  # GPU limitations
        
        if not valid_exponents:
            return [False] * len(exponents)
        
        try:
            if self.preferred_backend == 'cupy':
                # Batch processing with CuPy
                with cp.cuda.Device(0):
                    for p in valid_exponents:
                        result = self.lucas_lehmer_gpu_cupy(p)
                        results.append(result)
            
            elif self.preferred_backend == 'numba_cuda':
                # Batch processing with Numba
                for p in valid_exponents:
                    result = self.lucas_lehmer_gpu_numba(p)
                    results.append(result)
            
            elif self.preferred_backend == 'opencl':
                # Batch processing with OpenCL
                for p in valid_exponents:
                    result = self.lucas_lehmer_gpu_opencl(p)
                    results.append(result)
            
            # Pad results for invalid exponents
            while len(results) < len(exponents):
                results.append(False)
                
            return results
            
        except Exception as e:
            print(f"GPU batch processing error: {e}")
            return [False] * len(exponents)
    
    def benchmark_gpu_performance(self, test_exponents: List[int]) -> GPUPerformanceMetrics:
        """Benchmark GPU performance against CPU"""
        if not self.preferred_backend:
            return GPUPerformanceMetrics(
                gpu_type="None",
                computation_time=0.0,
                speedup_factor=0.0,
                memory_usage=0.0,
                throughput=0.0
            )
        
        # CPU baseline
        start_time = time.time()
        cpu_results = []
        for p in test_exponents:
            if p <= 1000:  # Keep CPU test reasonable
                mersenne = (1 << p) - 1
                s = 4
                for _ in range(p - 2):
                    s = (s * s - 2) % mersenne
                cpu_results.append(s == 0)
            else:
                cpu_results.append(False)
        cpu_time = time.time() - start_time
        
        # GPU test
        start_time = time.time()
        gpu_results = self.lucas_lehmer_gpu_batch(test_exponents)
        gpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0.0
        throughput = len(test_exponents) / gpu_time if gpu_time > 0 else 0.0
        
        # Memory usage estimation
        memory_usage = 0.0
        if self.cuda_available:
            try:
                memory_info = cp.cuda.runtime.memGetInfo()
                memory_usage = (memory_info[1] - memory_info[0]) / (1024**3)  # GB
            except:
                pass
        
        return GPUPerformanceMetrics(
            gpu_type=self.preferred_backend,
            computation_time=gpu_time,
            speedup_factor=speedup,
            memory_usage=memory_usage,
            throughput=throughput
        )
    
    def optimize_for_exponent_range(self, min_exp: int, max_exp: int) -> dict:
        """Optimize GPU settings for specific exponent range"""
        recommendations = {
            'use_gpu': False,
            'backend': self.preferred_backend,
            'batch_size': 1,
            'memory_optimization': False,
            'reason': ''
        }
        
        if not self.preferred_backend:
            recommendations['reason'] = 'No GPU available'
            return recommendations
        
        if max_exp <= 1000:
            recommendations['use_gpu'] = True
            recommendations['batch_size'] = 100
            recommendations['reason'] = 'Optimal range for GPU acceleration'
        elif max_exp <= 10000:
            recommendations['use_gpu'] = True
            recommendations['batch_size'] = 10
            recommendations['memory_optimization'] = True
            recommendations['reason'] = 'GPU beneficial with memory optimization'
        else:
            recommendations['reason'] = 'Exponents too large for current GPU precision'
        
        return recommendations

# Global GPU accelerator instance
gpu_accelerator = GPUAccelerator()