"""
Distributed Computing Network for Remote Mersenne Prime Analysis
"""

import asyncio
import aiohttp
import hashlib
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import threading
import queue
import logging

@dataclass
class RemoteNode:
    """Remote computation node information"""
    id: str
    url: str
    capacity: int
    current_load: int
    last_ping: datetime
    status: str  # 'online', 'offline', 'busy'
    specialization: str  # 'lucas_lehmer', 'miller_rabin', 'general'

@dataclass
class ComputationTask:
    """Computation task to be sent to remote nodes"""
    task_id: str
    exponent: int
    mersenne_number_hash: str
    mersenne_number: str
    test_type: str
    priority: int
    created_at: datetime
    timeout: int = 300  # 5 minutes

@dataclass
class TaskResult:
    """Result from remote computation"""
    task_id: str
    node_id: str
    is_prime: bool
    confidence_score: float
    test_passed: str
    computation_time: float
    verified_hash: str

class CandidateRepository:
    """Repository for storing promising candidates"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self._initialize_repository()
    
    def _initialize_repository(self):
        """Initialize candidate repository tables"""
        try:
            with self.db_manager.get_positive_cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS candidate_repository (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        exponent INTEGER UNIQUE NOT NULL,
                        mersenne_hash TEXT NOT NULL,
                        mersenne_number TEXT NOT NULL,
                        discovery_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        node_discoveries TEXT,
                        confidence_scores TEXT,
                        tests_passed TEXT,
                        status TEXT DEFAULT 'pending',
                        final_verification BOOLEAN DEFAULT FALSE
                    )
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_candidate_exponent 
                    ON candidate_repository(exponent)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_candidate_status 
                    ON candidate_repository(status)
                ''')
                
                self.logger.info("📦 Candidate repository initialized")
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing repository: {e}")
    
    def store_candidate(self, exponent: int, mersenne_number: str, 
                       node_id: str, test_passed: str, confidence: float) -> bool:
        """Store or update a promising candidate"""
        try:
            mersenne_hash = hashlib.sha256(mersenne_number.encode()).hexdigest()
            
            with self.db_manager.get_positive_cursor() as cursor:
                # Check if candidate already exists
                cursor.execute(
                    'SELECT node_discoveries, confidence_scores, tests_passed FROM candidate_repository WHERE exponent = ?',
                    (exponent,)
                )
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing candidate
                    nodes = json.loads(existing[0]) if existing[0] else []
                    scores = json.loads(existing[1]) if existing[1] else []
                    tests = json.loads(existing[2]) if existing[2] else []
                    
                    nodes.append(node_id)
                    scores.append(confidence)
                    tests.append(test_passed)
                    
                    cursor.execute('''
                        UPDATE candidate_repository 
                        SET node_discoveries = ?, confidence_scores = ?, tests_passed = ?
                        WHERE exponent = ?
                    ''', (json.dumps(nodes), json.dumps(scores), json.dumps(tests), exponent))
                    
                else:
                    # Insert new candidate
                    cursor.execute('''
                        INSERT INTO candidate_repository 
                        (exponent, mersenne_hash, mersenne_number, node_discoveries, confidence_scores, tests_passed)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        exponent, mersenne_hash, mersenne_number,
                        json.dumps([node_id]), json.dumps([confidence]), json.dumps([test_passed])
                    ))
                
                self.logger.info(f"📦 Stored candidate M{exponent} from node {node_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error storing candidate: {e}")
            return False
    
    def get_promising_candidates(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most promising candidates"""
        try:
            with self.db_manager.get_positive_cursor() as cursor:
                cursor.execute('''
                    SELECT exponent, mersenne_hash, node_discoveries, confidence_scores, 
                           tests_passed, discovery_date, status
                    FROM candidate_repository 
                    ORDER BY json_array_length(confidence_scores) DESC, discovery_date DESC
                    LIMIT ?
                ''', (limit,))
                
                candidates = []
                for row in cursor.fetchall():
                    nodes = json.loads(row[2]) if row[2] else []
                    scores = json.loads(row[3]) if row[3] else []
                    tests = json.loads(row[4]) if row[4] else []
                    
                    candidates.append({
                        'exponent': row[0],
                        'mersenne_hash': row[1],
                        'node_count': len(nodes),
                        'avg_confidence': sum(scores) / len(scores) if scores else 0,
                        'tests_passed': tests,
                        'discovery_date': row[5],
                        'status': row[6]
                    })
                
                return candidates
                
        except Exception as e:
            self.logger.error(f"❌ Error getting candidates: {e}")
            return []

class DistributedNetwork:
    """Distributed computing network manager"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self.repository = CandidateRepository(db_manager)
        
        # Network configuration
        self.remote_nodes: List[RemoteNode] = []
        self.task_queue = asyncio.Queue()
        self.result_queue = queue.Queue()
        self.active_tasks: Dict[str, ComputationTask] = {}
        
        # Statistics
        self.tasks_sent = 0
        self.tasks_completed = 0
        self.nodes_online = 0
        
        # Initialize default nodes (can be configured via API)
        self._initialize_default_nodes()
        
    def _initialize_default_nodes(self):
        """Initialize default computation nodes"""
        # These would be real distributed nodes in production
        default_nodes = [
            {
                'id': 'node_usa_01',
                'url': 'https://compute-usa-01.mersenne.network/api/compute',
                'capacity': 100,
                'specialization': 'lucas_lehmer'
            },
            {
                'id': 'node_europe_01', 
                'url': 'https://compute-europe-01.mersenne.network/api/compute',
                'capacity': 150,
                'specialization': 'miller_rabin'
            },
            {
                'id': 'node_asia_01',
                'url': 'https://compute-asia-01.mersenne.network/api/compute', 
                'capacity': 80,
                'specialization': 'general'
            },
            {
                'id': 'node_cloud_01',
                'url': 'https://compute-cloud-01.mersenne.network/api/compute',
                'capacity': 200,
                'specialization': 'lucas_lehmer'
            }
        ]
        
        for node_config in default_nodes:
            node = RemoteNode(
                id=node_config['id'],
                url=node_config['url'],
                capacity=node_config['capacity'],
                current_load=0,
                last_ping=datetime.now(),
                status='online',  # Would be checked via ping
                specialization=node_config['specialization']
            )
            self.remote_nodes.append(node)
            
        self.logger.info(f"🌐 Initialized {len(self.remote_nodes)} remote nodes")
    
    def add_remote_node(self, url: str, capacity: int = 100, specialization: str = 'general') -> str:
        """Add a new remote computation node"""
        node_id = f"node_{len(self.remote_nodes):03d}"
        
        node = RemoteNode(
            id=node_id,
            url=url,
            capacity=capacity,
            current_load=0,
            last_ping=datetime.now(),
            status='online',
            specialization=specialization
        )
        
        self.remote_nodes.append(node)
        self.logger.info(f"➕ Added remote node {node_id}: {url}")
        return node_id
    
    def create_sha256_hash(self, mersenne_number: str) -> str:
        """Create SHA-256 hash of Mersenne number"""
        return hashlib.sha256(mersenne_number.encode()).hexdigest()
    
    async def submit_computation_task(self, exponent: int, mersenne_number: str, 
                                    test_type: str = 'lucas_lehmer', priority: int = 1) -> str:
        """Submit computation task to distributed network"""
        
        # Create task with SHA-256 hash
        task_id = f"task_{int(time.time())}_{exponent}"
        mersenne_hash = self.create_sha256_hash(mersenne_number)
        
        task = ComputationTask(
            task_id=task_id,
            exponent=exponent,
            mersenne_number_hash=mersenne_hash,
            mersenne_number=mersenne_number,
            test_type=test_type,
            priority=priority,
            created_at=datetime.now()
        )
        
        await self.task_queue.put(task)
        self.active_tasks[task_id] = task
        self.tasks_sent += 1
        
        self.logger.info(f"📤 Submitted task {task_id} for M{exponent} (Hash: {mersenne_hash[:16]}...)")
        return task_id
    
    async def process_computation_tasks(self):
        """Process computation tasks and send to remote nodes"""
        while True:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Find best available node
                best_node = self._select_best_node(task.test_type)
                
                if best_node:
                    # Send task to remote node
                    await self._send_task_to_node(task, best_node)
                else:
                    # No nodes available, put task back in queue
                    await asyncio.sleep(1)
                    await self.task_queue.put(task)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"❌ Error processing tasks: {e}")
                await asyncio.sleep(1)
    
    def _select_best_node(self, test_type: str) -> Optional[RemoteNode]:
        """Select best available node for task"""
        available_nodes = [
            node for node in self.remote_nodes 
            if node.status == 'online' and node.current_load < node.capacity
        ]
        
        if not available_nodes:
            return None
        
        # Prefer nodes specialized in the test type
        specialized_nodes = [
            node for node in available_nodes 
            if node.specialization == test_type or node.specialization == 'general'
        ]
        
        target_nodes = specialized_nodes if specialized_nodes else available_nodes
        
        # Select node with lowest load
        return min(target_nodes, key=lambda n: n.current_load / n.capacity)
    
    async def _send_task_to_node(self, task: ComputationTask, node: RemoteNode):
        """Send computation task to remote node"""
        try:
            # Prepare payload with SHA-256 hash
            payload = {
                'task_id': task.task_id,
                'exponent': task.exponent,
                'mersenne_hash': task.mersenne_number_hash,
                'mersenne_number': task.mersenne_number,
                'test_type': task.test_type,
                'single_test_sufficient': True,  # Return immediately on first passed test
                'timeout': task.timeout
            }
            
            # In production, this would send to actual remote node
            # For now, simulate the request
            self.logger.info(f"🚀 Sending task {task.task_id} to {node.id}")
            self.logger.info(f"📊 Hash: {task.mersenne_number_hash}")
            self.logger.info(f"🎯 Test type: {task.test_type} (single test sufficient)")
            
            # Simulate remote computation result
            await self._simulate_remote_computation(task, node)
            
            # Update node load
            node.current_load += 1
            node.last_ping = datetime.now()
            
        except Exception as e:
            self.logger.error(f"❌ Error sending task to {node.id}: {e}")
    
    async def _simulate_remote_computation(self, task: ComputationTask, node: RemoteNode):
        """Simulate remote computation (replace with actual HTTP request in production)"""
        await asyncio.sleep(0.5)  # Simulate network delay
        
        # Simulate computation result
        import random
        is_prime = random.random() < 0.001  # Very low chance for demonstration
        confidence = random.uniform(0.85, 0.99) if is_prime else random.uniform(0.1, 0.3)
        
        result = TaskResult(
            task_id=task.task_id,
            node_id=node.id,
            is_prime=is_prime,
            confidence_score=confidence,
            test_passed='lucas_lehmer' if is_prime else 'failed_initial',
            computation_time=random.uniform(0.1, 2.0),
            verified_hash=task.mersenne_number_hash
        )
        
        # Process result
        await self._process_task_result(result)
        
        # Update node load
        node.current_load = max(0, node.current_load - 1)
    
    async def _process_task_result(self, result: TaskResult):
        """Process result from remote computation"""
        try:
            self.tasks_completed += 1
            
            # Remove from active tasks
            if result.task_id in self.active_tasks:
                task = self.active_tasks.pop(result.task_id)
                
                self.logger.info(f"📥 Received result from {result.node_id}")
                self.logger.info(f"🔍 Task {result.task_id}: {'PRIME CANDIDATE' if result.is_prime else 'NOT PRIME'}")
                self.logger.info(f"📊 Confidence: {result.confidence_score:.3f}")
                self.logger.info(f"⏱️ Computation time: {result.computation_time:.2f}s")
                
                # Store promising candidates
                if result.is_prime and result.confidence_score > 0.8:
                    self.repository.store_candidate(
                        task.exponent,
                        task.mersenne_number,
                        result.node_id,
                        result.test_passed,
                        result.confidence_score
                    )
                    
                    self.logger.info(f"🎯 PROMISING CANDIDATE M{task.exponent} stored in repository!")
                
                # Put result in queue for main thread
                self.result_queue.put(result)
                
        except Exception as e:
            self.logger.error(f"❌ Error processing result: {e}")
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status"""
        online_nodes = [node for node in self.remote_nodes if node.status == 'online']
        total_capacity = sum(node.capacity for node in online_nodes)
        current_load = sum(node.current_load for node in online_nodes)
        
        return {
            'nodes_total': len(self.remote_nodes),
            'nodes_online': len(online_nodes),
            'total_capacity': total_capacity,
            'current_load': current_load,
            'utilization': (current_load / total_capacity * 100) if total_capacity > 0 else 0,
            'tasks_sent': self.tasks_sent,
            'tasks_completed': self.tasks_completed,
            'tasks_pending': len(self.active_tasks),
            'queue_size': self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0
        }
    
    def get_node_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all nodes"""
        return [
            {
                'id': node.id,
                'url': node.url,
                'status': node.status,
                'capacity': node.capacity,
                'current_load': node.current_load,
                'utilization': (node.current_load / node.capacity * 100) if node.capacity > 0 else 0,
                'specialization': node.specialization,
                'last_ping': node.last_ping.isoformat()
            }
            for node in self.remote_nodes
        ]

# Global distributed network instance
distributed_network = None

def get_distributed_network(db_manager=None):
    """Get or create distributed network instance"""
    global distributed_network
    if distributed_network is None and db_manager:
        distributed_network = DistributedNetwork(db_manager)
    return distributed_network