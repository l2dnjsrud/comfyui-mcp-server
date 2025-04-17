import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable, Union, Set
from datetime import datetime
import uuid

logger = logging.getLogger("JobQueue")

class Job:
    """Represents a single generation job in the queue"""
    
    def __init__(self, job_id: str, job_type: str, params: Dict[str, Any], priority: int = 0):
        self.job_id = job_id
        self.job_type = job_type  # 'txt2img', 'img2img', 'inpaint', etc.
        self.params = params
        self.priority = priority
        self.status = "pending"
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_time = time.time()
        self.started_time: Optional[float] = None
        self.completed_time: Optional[float] = None
        self.progress = 0
        self.details: Dict[str, Any] = {}
        self.callbacks: Set[Callable[[Dict[str, Any]], Awaitable[None]]] = set()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization"""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "params": self.params,
            "priority": self.priority,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_time": self.created_time,
            "started_time": self.started_time,
            "completed_time": self.completed_time,
            "progress": self.progress,
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Job':
        """Create job from dictionary representation"""
        job = cls(
            job_id=data["job_id"],
            job_type=data["job_type"],
            params=data["params"],
            priority=data["priority"]
        )
        job.status = data["status"]
        job.result = data.get("result")
        job.error = data.get("error")
        job.created_time = data["created_time"]
        job.started_time = data.get("started_time")
        job.completed_time = data.get("completed_time")
        job.progress = data.get("progress", 0)
        job.details = data.get("details", {})
        return job
    
    def __lt__(self, other: 'Job') -> bool:
        """Compare jobs for priority queue ordering"""
        if self.priority != other.priority:
            # Higher priority jobs come first
            return self.priority > other.priority
        else:
            # Within same priority, older jobs come first
            return self.created_time < other.created_time

class JobQueue:
    """
    Manages a queue of generation jobs with priority ordering and parallel processing.
    Provides status tracking and callback notifications for job progress.
    """
    
    def __init__(self, max_concurrent_jobs: int = 2, save_path: Optional[str] = None):
        self.queue: asyncio.PriorityQueue[Job] = asyncio.PriorityQueue()
        self.active_jobs: Dict[str, Job] = {}
        self.completed_jobs: Dict[str, Job] = {}
        self.max_concurrent_jobs = max_concurrent_jobs
        self.save_path = save_path
        self.running = False
        self.processing_task = None
        self.job_executor: Optional[Callable[[Job], Awaitable[Dict[str, Any]]]] = None
        self.progress_updater: Optional[Callable[[Job, Dict[str, Any]], Awaitable[None]]] = None
    
    async def start(self):
        """Start the job queue processor"""
        if self.running:
            return
            
        self.running = True
        self.processing_task = asyncio.create_task(self._process_queue())
        logger.info("Job queue started")
        
        # Attempt to restore queue state if save path is provided
        if self.save_path:
            await self._restore_state()
    
    async def stop(self):
        """Stop the job queue processor"""
        if not self.running:
            return
            
        self.running = False
        
        if self.processing_task:
            # Wait for processing to complete with timeout
            try:
                await asyncio.wait_for(self.processing_task, timeout=5.0)
            except asyncio.TimeoutError:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
            
        # Save queue state if save path is provided
        if self.save_path:
            await self._save_state()
            
        logger.info("Job queue stopped")
    
    def set_job_executor(self, executor: Callable[[Job], Awaitable[Dict[str, Any]]]):
        """Set the function that will execute jobs"""
        self.job_executor = executor
    
    def set_progress_updater(self, updater: Callable[[Job, Dict[str, Any]], Awaitable[None]]):
        """Set the function that will handle progress updates"""
        self.progress_updater = updater
    
    async def add_job(self, job_type: str, params: Dict[str, Any], priority: int = 0, job_id: Optional[str] = None) -> str:
        """
        Add a new job to the queue
        
        Args:
            job_type: Type of job ('txt2img', 'img2img', etc.)
            params: Job parameters
            priority: Job priority (higher numbers = higher priority)
            job_id: Optional custom job ID
            
        Returns:
            Job ID for tracking
        """
        # Generate job ID if not provided
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        # Create job
        job = Job(job_id, job_type, params, priority)
        
        # Add to queue
        await self.queue.put(job)
        
        logger.info(f"Added job {job_id} to queue with priority {priority}")
        
        return job_id
    
    async def register_callback(self, job_id: str, callback: Callable[[Dict[str, Any]], Awaitable[None]]) -> bool:
        """
        Register a callback for job status updates
        
        Args:
            job_id: ID of the job to track
            callback: Async function to call with job status updates
            
        Returns:
            True if registration successful, False otherwise
        """
        # Check if job exists in active or completed jobs
        if job_id in self.active_jobs:
            self.active_jobs[job_id].callbacks.add(callback)
            
            # Send current status immediately
            await callback(self.active_jobs[job_id].to_dict())
            return True
            
        elif job_id in self.completed_jobs:
            # For completed jobs, just send the final status once
            await callback(self.completed_jobs[job_id].to_dict())
            return True
            
        # Job not found, check if it's in the queue
        jobs_in_queue = []
        temp_queue = asyncio.PriorityQueue()
        
        found = False
        
        # Search through queue (this is inefficient but necessary)
        while not self.queue.empty():
            job = await self.queue.get()
            jobs_in_queue.append(job)
            
            if job.job_id == job_id:
                job.callbacks.add(callback)
                # Send current status immediately
                await callback(job.to_dict())
                found = True
        
        # Restore queue
        for job in jobs_in_queue:
            await temp_queue.put(job)
            
        self.queue = temp_queue
        
        return found
    
    async def unregister_callback(self, job_id: str, callback: Callable[[Dict[str, Any]], Awaitable[None]]) -> bool:
        """
        Unregister a callback for job status updates
        
        Args:
            job_id: ID of the job
            callback: Callback to remove
            
        Returns:
            True if unregistration successful, False otherwise
        """
        if job_id in self.active_jobs:
            self.active_jobs[job_id].callbacks.discard(callback)
            return True
            
        return False
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a job
        
        Args:
            job_id: ID of the job
            
        Returns:
            Job status dictionary or None if job not found
        """
        if job_id in self.active_jobs:
            return self.active_jobs[job_id].to_dict()
            
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id].to_dict()
            
        # Check if job is in queue
        jobs_in_queue = []
        temp_queue = asyncio.PriorityQueue()
        
        job_status = None
        
        # Search through queue
        while not self.queue.empty():
            job = await self.queue.get()
            jobs_in_queue.append(job)
            
            if job.job_id == job_id:
                job_status = job.to_dict()
        
        # Restore queue
        for job in jobs_in_queue:
            await temp_queue.put(job)
            
        self.queue = temp_queue
        
        return job_status
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job
        
        Args:
            job_id: ID of the job to cancel
            
        Returns:
            True if cancellation successful, False otherwise
        """
        # If job is active, we can't cancel it
        if job_id in self.active_jobs:
            return False
            
        # If job is completed, no need to cancel
        if job_id in self.completed_jobs:
            return False
            
        # Check if job is in queue
        jobs_in_queue = []
        temp_queue = asyncio.PriorityQueue()
        
        found = False
        
        # Search through queue
        while not self.queue.empty():
            job = await self.queue.get()
            
            if job.job_id == job_id:
                # Mark as cancelled and add to completed jobs
                job.status = "cancelled"
                job.completed_time = time.time()
                self.completed_jobs[job_id] = job
                
                # Notify callbacks
                for callback in job.callbacks:
                    try:
                        await callback(job.to_dict())
                    except Exception as e:
                        logger.error(f"Error in job callback: {e}")
                
                found = True
            else:
                # Keep other jobs in queue
                jobs_in_queue.append(job)
        
        # Restore queue without the cancelled job
        for job in jobs_in_queue:
            await temp_queue.put(job)
            
        self.queue = temp_queue
        
        if found:
            logger.info(f"Cancelled job {job_id}")
            
        return found
    
    async def update_job_progress(self, job_id: str, progress: int, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update progress of an active job
        
        Args:
            job_id: ID of the job
            progress: Progress percentage (0-100)
            details: Optional additional details
            
        Returns:
            True if update successful, False otherwise
        """
        if job_id not in self.active_jobs:
            return False
            
        job = self.active_jobs[job_id]
        job.progress = progress
        
        if details:
            job.details.update(details)
            
        # Notify callbacks
        for callback in job.callbacks:
            try:
                await callback(job.to_dict())
            except Exception as e:
                logger.error(f"Error in job callback: {e}")
                job.callbacks.discard(callback)
                
        return True
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current status of the entire queue
        
        Returns:
            Dictionary with queue status information
        """
        # Count jobs in queue
        queue_size = self.queue.qsize()
        
        # Get active jobs count
        active_jobs_count = len(self.active_jobs)
        
        # Get completed jobs count (limit to recent ones)
        completed_jobs_count = len(self.completed_jobs)
        
        # Calculate estimated wait time (very rough estimate)
        avg_job_time = 30  # Default assumption: 30 seconds per job
        
        # If we have completed jobs, calculate average job time
        if completed_jobs_count > 0:
            total_time = 0
            count = 0
            
            for job in self.completed_jobs.values():
                if job.completed_time and job.started_time:
                    total_time += (job.completed_time - job.started_time)
                    count += 1
                    
            if count > 0:
                avg_job_time = total_time / count
        
        # Estimate wait time based on queue size and active jobs
        estimated_wait_time = (queue_size / max(1, self.max_concurrent_jobs)) * avg_job_time
        
        return {
            "queue_size": queue_size,
            "active_jobs": active_jobs_count,
            "completed_jobs": completed_jobs_count,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "estimated_wait_time": estimated_wait_time,
            "running": self.running
        }
    
    async def clear_completed_jobs(self, max_age_seconds: int = 3600) -> int:
        """
        Clear old completed jobs to prevent memory leaks
        
        Args:
            max_age_seconds: Maximum age of completed jobs to keep
            
        Returns:
            Number of jobs cleared
        """
        current_time = time.time()
        jobs_to_remove = []
        
        for job_id, job in self.completed_jobs.items():
            if job.completed_time and (current_time - job.completed_time) > max_age_seconds:
                jobs_to_remove.append(job_id)
                
        for job_id in jobs_to_remove:
            del self.completed_jobs[job_id]
            
        if jobs_to_remove:
            logger.info(f"Cleared {len(jobs_to_remove)} completed jobs")
            
        return len(jobs_to_remove)
    
    async def _process_queue(self):
        """Process jobs from the queue"""
        while self.running:
            try:
                # Check if we can process more jobs
                if len(self.active_jobs) >= self.max_concurrent_jobs:
                    # Wait a bit and check again
                    await asyncio.sleep(0.5)
                    continue
                
                # Try to get a job from the queue with timeout
                try:
                    job = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # No jobs in queue, wait and try again
                    await asyncio.sleep(0.5)
                    continue
                
                # Mark job as active
                job.status = "processing"
                job.started_time = time.time()
                self.active_jobs[job.job_id] = job
                
                # Start job processing in background
                asyncio.create_task(self._execute_job(job))
                
            except asyncio.CancelledError:
                # Queue processing cancelled
                break
                
            except Exception as e:
                logger.error(f"Error processing job queue: {e}")
                # Wait a bit before continuing
                await asyncio.sleep(1.0)
    
    async def _execute_job(self, job: Job):
        """Execute a single job"""
        try:
            logger.info(f"Executing job {job.job_id} of type {job.job_type}")
            
            # Notify callbacks about job start
            for callback in job.callbacks:
                try:
                    await callback(job.to_dict())
                except Exception as e:
                    logger.error(f"Error in job callback: {e}")
                    job.callbacks.discard(callback)
            
            # Execute job if executor is set
            if self.job_executor:
                result = await self.job_executor(job)
                
                # Update job with result
                job.result = result
                job.status = "completed"
            else:
                # No executor set
                job.error = "No job executor configured"
                job.status = "error"
                
        except Exception as e:
            logger.error(f"Error executing job {job.job_id}: {e}")
            job.error = str(e)
            job.status = "error"
            
        finally:
            # Mark job as completed
            job.completed_time = time.time()
            
            # Move from active to completed jobs
            del self.active_jobs[job.job_id]
            self.completed_jobs[job.job_id] = job
            
            # Notify callbacks about job completion
            for callback in job.callbacks:
                try:
                    await callback(job.to_dict())
                except Exception as e:
                    logger.error(f"Error in job callback: {e}")
                    job.callbacks.discard(callback)
                    
            logger.info(f"Completed job {job.job_id} with status {job.status}")
            
            # Save queue state if save path is provided
            if self.save_path:
                await self._save_state()
    
    async def _save_state(self):
        """Save queue state to disk"""
        if not self.save_path:
            return
            
        try:
            # Convert all jobs to dictionaries
            active_jobs = {job_id: job.to_dict() for job_id, job in self.active_jobs.items()}
            completed_jobs = {job_id: job.to_dict() for job_id, job in self.completed_jobs.items()}
            
            # Convert queue to list (this will empty the queue)
            queue_jobs = []
            temp_queue = asyncio.PriorityQueue()
            
            while not self.queue.empty():
                job = await self.queue.get()
                queue_jobs.append(job.to_dict())
                await temp_queue.put(job)
                
            self.queue = temp_queue
            
            # Create state object
            state = {
                "timestamp": time.time(),
                "active_jobs": active_jobs,
                "completed_jobs": completed_jobs,
                "queue_jobs": queue_jobs
            }
            
            # Save to file
            with open(self.save_path, "w") as f:
                json.dump(state, f)
                
            logger.debug(f"Saved queue state to {self.save_path}")
                
        except Exception as e:
            logger.error(f"Error saving queue state: {e}")
    
    async def _restore_state(self):
        """Restore queue state from disk"""
        if not self.save_path:
            return
            
        try:
            import os
            if not os.path.exists(self.save_path):
                logger.info(f"No saved state found at {self.save_path}")
                return
                
            with open(self.save_path, "r") as f:
                state = json.load(f)
                
            # Restore completed jobs (limit to recent ones)
            for job_id, job_data in state.get("completed_jobs", {}).items():
                completed_time = job_data.get("completed_time", 0)
                if time.time() - completed_time < 3600:  # Only restore jobs less than an hour old
                    self.completed_jobs[job_id] = Job.from_dict(job_data)
                    
            # Restore queue jobs
            for job_data in state.get("queue_jobs", []):
                # Skip old pending jobs
                created_time = job_data.get("created_time", 0)
                if time.time() - created_time < 3600 * 6:  # Only restore jobs less than 6 hours old
                    job = Job.from_dict(job_data)
                    await self.queue.put(job)
                    
            # Don't restore active jobs - they should be resubmitted
                
            logger.info(f"Restored queue state from {self.save_path}")
                
        except Exception as e:
            logger.error(f"Error restoring queue state: {e}")
