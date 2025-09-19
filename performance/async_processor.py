"""
Asynchronous Processing System for Fresh Supply Chain Intelligence
High-performance async processing with queues, workers, and batch processing
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from collections import deque
import concurrent.futures
import multiprocessing as mp
from queue import Queue, Empty
import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger()

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    """Task status states"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"

@dataclass
class TaskConfig:
    """Task configuration"""
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    priority: TaskPriority = TaskPriority.NORMAL
    batch_size: int = 1
    max_batch_wait: float = 5.0

@dataclass
class Task:
    """Async task representation"""
    id: str
    func: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    config: TaskConfig = field(default_factory=TaskConfig)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0

class WorkerMetrics:
    """Worker performance metrics"""
    
    def __init__(self):
        self.tasks_processed = 0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_processing_time = 0.0
        self.avg_processing_time = 0.0
        self.current_load = 0
        self.max_load = 0
        self.start_time = datetime.now()

class AsyncWorker:
    """Asynchronous task worker"""
    
    def __init__(self, worker_id: str, max_concurrent_tasks: int = 10):
        self.worker_id = worker_id
        self.max_concurrent_tasks = max_concurrent_tasks
        self.current_tasks = set()
        self.metrics = WorkerMetrics()
        self.is_running = False
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
    async def process_task(self, task: Task) -> Task:
        """Process a single task"""
        async with self.semaphore:
            self.current_tasks.add(task.id)
            self.metrics.current_load = len(self.current_tasks)
            self.metrics.max_load = max(self.metrics.max_load, self.metrics.current_load)
            
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            
            start_time = time.time()
            
            try:
                # Execute task with timeout
                if asyncio.iscoroutinefunction(task.func):
                    result = await asyncio.wait_for(
                        task.func(*task.args, **task.kwargs),
                        timeout=task.config.timeout
                    )
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, task.func, *task.args, **task.kwargs),
                        timeout=task.config.timeout
                    )
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                
                self.metrics.tasks_completed += 1
                
                logger.debug(f"Worker {self.worker_id} completed task {task.id}")
                
            except asyncio.TimeoutError:
                task.error = f"Task timed out after {task.config.timeout} seconds"
                task.status = TaskStatus.FAILED
                self.metrics.tasks_failed += 1
                logger.error(f"Worker {self.worker_id} task {task.id} timed out")
                
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
                self.metrics.tasks_failed += 1
                logger.error(f"Worker {self.worker_id} task {task.id} failed: {e}")
                
            finally:
                processing_time = time.time() - start_time
                self.metrics.total_processing_time += processing_time
                self.metrics.tasks_processed += 1
                self.metrics.avg_processing_time = (
                    self.metrics.total_processing_time / self.metrics.tasks_processed
                )
                
                self.current_tasks.discard(task.id)
                self.metrics.current_load = len(self.current_tasks)
            
            return task
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker metrics"""
        uptime = (datetime.now() - self.metrics.start_time).total_seconds()
        
        return {
            "worker_id": self.worker_id,
            "tasks_processed": self.metrics.tasks_processed,
            "tasks_completed": self.metrics.tasks_completed,
            "tasks_failed": self.metrics.tasks_failed,
            "success_rate": (
                self.metrics.tasks_completed / max(self.metrics.tasks_processed, 1)
            ),
            "avg_processing_time": self.metrics.avg_processing_time,
            "current_load": self.metrics.current_load,
            "max_load": self.metrics.max_load,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "uptime_seconds": uptime,
            "throughput": self.metrics.tasks_processed / max(uptime, 1)
        }

class TaskQueue:
    """Priority-based task queue"""
    
    def __init__(self, redis_url: str = None):
        self.queues = {
            TaskPriority.CRITICAL: deque(),
            TaskPriority.HIGH: deque(),
            TaskPriority.NORMAL: deque(),
            TaskPriority.LOW: deque()
        }
        self.tasks = {}  # Task storage by ID
        self.redis_client = None
        self.lock = asyncio.Lock()
        
        if redis_url:
            self._initialize_redis(redis_url)
    
    async def _initialize_redis(self, redis_url: str):
        """Initialize Redis for distributed queuing"""
        try:
            self.redis_client = aioredis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("Redis queue backend initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis queue: {e}")
            self.redis_client = None
    
    async def enqueue(self, task: Task):
        """Add task to queue"""
        async with self.lock:
            self.tasks[task.id] = task
            
            if self.redis_client:
                # Store in Redis for distributed processing
                await self._enqueue_redis(task)
            else:
                # Store in memory queue
                self.queues[task.config.priority].append(task.id)
            
            logger.debug(f"Enqueued task {task.id} with priority {task.config.priority.name}")
    
    async def dequeue(self) -> Optional[Task]:
        """Get next task from queue (highest priority first)"""
        async with self.lock:
            if self.redis_client:
                return await self._dequeue_redis()
            else:
                return await self._dequeue_memory()
    
    async def _enqueue_redis(self, task: Task):
        """Enqueue task in Redis"""
        queue_name = f"task_queue:{task.config.priority.value}"
        task_data = {
            "id": task.id,
            "func_name": task.func.__name__,
            "args": task.args,
            "kwargs": task.kwargs,
            "config": {
                "max_retries": task.config.max_retries,
                "retry_delay": task.config.retry_delay,
                "timeout": task.config.timeout,
                "priority": task.config.priority.value
            },
            "created_at": task.created_at.isoformat()
        }
        
        await self.redis_client.lpush(queue_name, json.dumps(task_data))
    
    async def _dequeue_redis(self) -> Optional[Task]:
        """Dequeue task from Redis"""
        # Check queues in priority order
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
            queue_name = f"task_queue:{priority.value}"
            
            result = await self.redis_client.brpop(queue_name, timeout=1)
            if result:
                _, task_data = result
                task_dict = json.loads(task_data)
                
                # Reconstruct task (simplified - in production, use proper serialization)
                task_id = task_dict["id"]
                if task_id in self.tasks:
                    return self.tasks[task_id]
        
        return None
    
    async def _dequeue_memory(self) -> Optional[Task]:
        """Dequeue task from memory"""
        # Check queues in priority order
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
            queue = self.queues[priority]
            if queue:
                task_id = queue.popleft()
                return self.tasks.get(task_id)
        
        return None
    
    async def get_queue_size(self) -> Dict[str, int]:
        """Get queue sizes by priority"""
        if self.redis_client:
            sizes = {}
            for priority in TaskPriority:
                queue_name = f"task_queue:{priority.value}"
                size = await self.redis_client.llen(queue_name)
                sizes[priority.name] = size
            return sizes
        else:
            return {
                priority.name: len(queue)
                for priority, queue in self.queues.items()
            }
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def update_task(self, task: Task):
        """Update task in storage"""
        self.tasks[task.id] = task

class BatchProcessor:
    """Batch processing for efficient bulk operations"""
    
    def __init__(self, batch_size: int = 10, max_wait_time: float = 5.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_batches = {}
        self.lock = asyncio.Lock()
    
    async def add_to_batch(self, batch_key: str, item: Any, processor: Callable) -> Any:
        """Add item to batch for processing"""
        async with self.lock:
            if batch_key not in self.pending_batches:
                self.pending_batches[batch_key] = {
                    "items": [],
                    "processor": processor,
                    "created_at": time.time(),
                    "futures": []
                }
            
            batch = self.pending_batches[batch_key]
            future = asyncio.Future()
            
            batch["items"].append(item)
            batch["futures"].append(future)
            
            # Process batch if it's full or has been waiting too long
            should_process = (
                len(batch["items"]) >= self.batch_size or
                time.time() - batch["created_at"] > self.max_wait_time
            )
            
            if should_process:
                asyncio.create_task(self._process_batch(batch_key))
            
            return await future
    
    async def _process_batch(self, batch_key: str):
        """Process a batch of items"""
        async with self.lock:
            if batch_key not in self.pending_batches:
                return
            
            batch = self.pending_batches.pop(batch_key)
        
        try:
            # Process all items in the batch
            results = await batch["processor"](batch["items"])
            
            # Set results for all futures
            for future, result in zip(batch["futures"], results):
                if not future.done():
                    future.set_result(result)
                    
        except Exception as e:
            # Set exception for all futures
            for future in batch["futures"]:
                if not future.done():
                    future.set_exception(e)

class AsyncProcessor:
    """Main asynchronous processing system"""
    
    def __init__(self, num_workers: int = 4, redis_url: str = None):
        self.num_workers = num_workers
        self.workers = []
        self.task_queue = TaskQueue(redis_url)
        self.batch_processor = BatchProcessor()
        self.is_running = False
        self.worker_tasks = []
        self.retry_queue = deque()
        
        # Initialize workers
        for i in range(num_workers):
            worker = AsyncWorker(f"worker_{i}")
            self.workers.append(worker)
    
    async def start(self):
        """Start the async processor"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker tasks
        for worker in self.workers:
            task = asyncio.create_task(self._worker_loop(worker))
            self.worker_tasks.append(task)
        
        # Start retry processor
        retry_task = asyncio.create_task(self._retry_loop())
        self.worker_tasks.append(retry_task)
        
        logger.info(f"Started async processor with {self.num_workers} workers")
    
    async def stop(self):
        """Stop the async processor"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        logger.info("Stopped async processor")
    
    async def submit_task(self, func: Callable, *args, config: TaskConfig = None, **kwargs) -> str:
        """Submit task for async processing"""
        import uuid
        
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            config=config or TaskConfig()
        )
        
        await self.task_queue.enqueue(task)
        return task_id
    
    async def submit_batch_task(self, batch_key: str, func: Callable, items: List[Any]) -> List[Any]:
        """Submit batch task for processing"""
        return await self.batch_processor.add_to_batch(batch_key, items, func)
    
    async def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """Get task result (blocking until complete)"""
        start_time = time.time()
        
        while True:
            task = self.task_queue.get_task(task_id)
            
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            if task.status == TaskStatus.COMPLETED:
                return task.result
            elif task.status == TaskStatus.FAILED:
                raise Exception(f"Task failed: {task.error}")
            elif task.status == TaskStatus.CANCELLED:
                raise Exception("Task was cancelled")
            
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            await asyncio.sleep(0.1)
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        task = self.task_queue.get_task(task_id)
        
        if not task:
            return None
        
        return {
            "id": task.id,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "retry_count": task.retry_count,
            "error": task.error
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        task = self.task_queue.get_task(task_id)
        
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            self.task_queue.update_task(task)
            return True
        
        return False
    
    async def _worker_loop(self, worker: AsyncWorker):
        """Main worker loop"""
        while self.is_running:
            try:
                # Get next task from queue
                task = await self.task_queue.dequeue()
                
                if task and task.status == TaskStatus.PENDING:
                    # Process the task
                    processed_task = await worker.process_task(task)
                    
                    # Handle retry logic
                    if (processed_task.status == TaskStatus.FAILED and 
                        processed_task.retry_count < processed_task.config.max_retries):
                        
                        processed_task.retry_count += 1
                        processed_task.status = TaskStatus.RETRYING
                        self.retry_queue.append(processed_task)
                    
                    # Update task in storage
                    self.task_queue.update_task(processed_task)
                
                else:
                    # No tasks available, sleep briefly
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker.worker_id} error: {e}")
                await asyncio.sleep(1)
    
    async def _retry_loop(self):
        """Process retry queue"""
        while self.is_running:
            try:
                if self.retry_queue:
                    task = self.retry_queue.popleft()
                    
                    # Wait for retry delay
                    await asyncio.sleep(task.config.retry_delay)
                    
                    # Reset task status and re-enqueue
                    task.status = TaskStatus.PENDING
                    await self.task_queue.enqueue(task)
                    
                    logger.info(f"Retrying task {task.id} (attempt {task.retry_count + 1})")
                
                else:
                    await asyncio.sleep(1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retry loop error: {e}")
                await asyncio.sleep(1)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        queue_sizes = await self.task_queue.get_queue_size()
        worker_metrics = [worker.get_metrics() for worker in self.workers]
        
        # Aggregate worker metrics
        total_processed = sum(w["tasks_processed"] for w in worker_metrics)
        total_completed = sum(w["tasks_completed"] for w in worker_metrics)
        total_failed = sum(w["tasks_failed"] for w in worker_metrics)
        avg_processing_time = sum(w["avg_processing_time"] for w in worker_metrics) / len(worker_metrics)
        current_load = sum(w["current_load"] for w in worker_metrics)
        
        return {
            "system_status": "running" if self.is_running else "stopped",
            "num_workers": self.num_workers,
            "queue_sizes": queue_sizes,
            "retry_queue_size": len(self.retry_queue),
            "aggregate_metrics": {
                "total_tasks_processed": total_processed,
                "total_tasks_completed": total_completed,
                "total_tasks_failed": total_failed,
                "overall_success_rate": total_completed / max(total_processed, 1),
                "avg_processing_time": avg_processing_time,
                "current_system_load": current_load,
                "max_system_capacity": sum(w["max_concurrent_tasks"] for w in self.workers)
            },
            "worker_metrics": worker_metrics
        }

# Global async processor instance
async_processor = None

def initialize_async_processor(num_workers: int = 4, redis_url: str = None) -> AsyncProcessor:
    """Initialize global async processor"""
    global async_processor
    async_processor = AsyncProcessor(num_workers, redis_url)
    return async_processor

def get_async_processor() -> AsyncProcessor:
    """Get global async processor instance"""
    return async_processor