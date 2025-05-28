"""
Parallel processing engine for distributed Mersenne prime search
"""

import threading
import queue
import time
from typing import Callable, Any, Generator, List
from concurrent.futures import ThreadPoolExecutor, Future
import multiprocessing as mp

class ParallelProcessor:
    """Manages parallel execution of Mersenne prime testing"""
    
    def __init__(self, max_workers: int = 10):
        """
        Initialize parallel processor
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.is_processing = False
        self.thread_pool = None
        self.work_queue = queue.Queue(maxsize=1000)
        self.result_queue = queue.Queue()
        self.active_threads = 0
        self.threads_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Performance metrics
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.start_time = 0
        
    def start_processing(self, worker_function: Callable, 
                        data_generator: Generator, thread_count: int):
        """
        Start parallel processing
        
        Args:
            worker_function: Function to execute in parallel
            data_generator: Generator that yields work items
            thread_count: Number of worker threads to use
        """
        if self.is_processing:
            raise RuntimeError("Processing is already running")
        
        self.max_workers = min(thread_count, 1000000)  # Cap at 1 million threads
        self.is_processing = True
        self.shutdown_event.clear()
        self.start_time = time.time()
        
        # Start thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Start producer thread
        producer_thread = threading.Thread(
            target=self._producer_worker,
            args=(data_generator,),
            daemon=True
        )
        producer_thread.start()
        
        # Start consumer threads
        for _ in range(self.max_workers):
            future = self.thread_pool.submit(self._consumer_worker, worker_function)
            future.add_done_callback(self._worker_completion_callback)
    
    def _producer_worker(self, data_generator: Generator):
        """Producer thread that feeds work items to the queue"""
        try:
            for work_item in data_generator:
                if self.shutdown_event.is_set():
                    break
                
                # Add work item to queue (blocks if queue is full)
                self.work_queue.put(work_item, timeout=1.0)
                
        except queue.Full:
            pass  # Queue full, continue
        except Exception as e:
            print(f"Producer error: {e}")
    
    def _consumer_worker(self, worker_function: Callable):
        """Consumer thread that processes work items"""
        with self.threads_lock:
            self.active_threads += 1
        
        try:
            while self.is_processing and not self.shutdown_event.is_set():
                try:
                    # Get work item from queue
                    work_item = self.work_queue.get(timeout=1.0)
                    
                    if work_item is None:  # Poison pill
                        break
                    
                    # Process work item
                    start_time = time.time()
                    result = worker_function(work_item)
                    end_time = time.time()
                    
                    # Store result
                    self.result_queue.put({
                        'work_item': work_item,
                        'result': result,
                        'duration': end_time - start_time,
                        'thread_id': threading.current_thread().ident
                    })
                    
                    self.tasks_completed += 1
                    self.work_queue.task_done()
                    
                except queue.Empty:
                    continue  # Timeout, check if should continue
                except Exception as e:
                    self.tasks_failed += 1
                    print(f"Worker error: {e}")
                    
        finally:
            with self.threads_lock:
                self.active_threads -= 1
    
    def _worker_completion_callback(self, future: Future):
        """Callback for when a worker thread completes"""
        try:
            future.result()  # Get result to check for exceptions
        except Exception as e:
            print(f"Worker thread failed: {e}")
    
    def stop_processing(self):
        """Stop all processing threads"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        self.shutdown_event.set()
        
        # Send poison pills to stop workers
        for _ in range(self.max_workers):
            try:
                self.work_queue.put(None, timeout=0.1)
            except queue.Full:
                pass
        
        # Shutdown thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True, timeout=5.0)
            self.thread_pool = None
    
    def get_active_thread_count(self) -> int:
        """Get number of currently active threads"""
        with self.threads_lock:
            return self.active_threads
    
    def get_queue_sizes(self) -> tuple:
        """Get current queue sizes (work_queue, result_queue)"""
        return self.work_queue.qsize(), self.result_queue.qsize()
    
    def get_performance_metrics(self) -> dict:
        """Get performance metrics"""
        elapsed_time = time.time() - self.start_time if self.start_time > 0 else 0
        
        return {
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'elapsed_time': elapsed_time,
            'tasks_per_second': self.tasks_completed / elapsed_time if elapsed_time > 0 else 0,
            'active_threads': self.get_active_thread_count(),
            'work_queue_size': self.work_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'max_workers': self.max_workers
        }
    
    def pause_processing(self):
        """Pause processing (workers will wait for resume)"""
        # Implementation would involve a pause event that workers check
        pass
    
    def resume_processing(self):
        """Resume paused processing"""
        # Implementation would clear the pause event
        pass
    
    def adjust_thread_count(self, new_count: int):
        """
        Dynamically adjust the number of worker threads
        
        Args:
            new_count: New number of threads (10-1000)
        """
        new_count = max(10, min(new_count, 1000000))
        
        if not self.is_processing:
            self.max_workers = new_count
            return
        
        # For running processes, this would require more complex logic
        # to gracefully increase/decrease thread count
        current_count = self.get_active_thread_count()
        
        if new_count > current_count:
            # Add more threads
            additional_threads = new_count - current_count
            if self.thread_pool:
                for _ in range(additional_threads):
                    future = self.thread_pool.submit(self._consumer_worker, None)
                    future.add_done_callback(self._worker_completion_callback)
        
        self.max_workers = new_count
    
    def get_worker_statistics(self) -> dict:
        """Get detailed worker thread statistics"""
        work_queue_size, result_queue_size = self.get_queue_sizes()
        
        return {
            'configured_workers': self.max_workers,
            'active_workers': self.get_active_thread_count(),
            'pending_work_items': work_queue_size,
            'completed_results': result_queue_size,
            'is_processing': self.is_processing,
            'cpu_count': mp.cpu_count(),
            'memory_usage_mb': self._get_memory_usage()
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_processing()
        
        # Clear queues
        while not self.work_queue.empty():
            try:
                self.work_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

class DistributedProcessor:
    """Extension for multi-machine distributed processing"""
    
    def __init__(self):
        """Initialize distributed processor"""
        self.node_id = None
        self.coordinator_address = None
        self.worker_nodes = []
    
    def setup_coordinator(self, port: int = 8000):
        """Set up this instance as a coordinator node"""
        # Implementation for distributed coordination would go here
        # This could use technologies like Redis, RabbitMQ, or custom TCP
        pass
    
    def connect_to_coordinator(self, coordinator_address: str):
        """Connect to a coordinator node as a worker"""
        # Implementation for connecting to distributed coordinator
        pass
    
    def distribute_work(self, work_items: List[Any]) -> List[Any]:
        """Distribute work items across available nodes"""
        # Implementation for work distribution
        pass
