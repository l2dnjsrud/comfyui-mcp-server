import aiosqlite
import json
import logging
import os
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger("DatabaseManager")

class DatabaseManager:
    """Base class for database operations"""
    
    async def initialize(self):
        """Initialize the database"""
        raise NotImplementedError("Subclasses must implement initialize()")
    
    async def save_job(self, job_data: Dict[str, Any]) -> str:
        """Save a job to the database and return job ID"""
        raise NotImplementedError("Subclasses must implement save_job()")
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID"""
        raise NotImplementedError("Subclasses must implement get_job()")
    
    async def update_job_status(self, job_id: str, status: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """Update job status"""
        raise NotImplementedError("Subclasses must implement update_job_status()")
    
    async def get_recent_jobs(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get recent jobs"""
        raise NotImplementedError("Subclasses must implement get_recent_jobs()")
    
    async def get_gallery_images(self, page: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
        """Get images for gallery"""
        raise NotImplementedError("Subclasses must implement get_gallery_images()")
    
    async def get_gallery_count(self) -> int:
        """Get total count of gallery images"""
        raise NotImplementedError("Subclasses must implement get_gallery_count()")
    
    async def close(self):
        """Close database connection"""
        raise NotImplementedError("Subclasses must implement close()")

class SQLiteDatabaseManager(DatabaseManager):
    """SQLite implementation of DatabaseManager"""
    
    def __init__(self, db_path: str = "mcp_server.db"):
        self.db_path = db_path
        self.conn = None
    
    async def initialize(self):
        """Initialize the SQLite database"""
        # Create database directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        
        # Connect to database
        self.conn = await aiosqlite.connect(self.db_path)
        
        # Enable foreign keys
        await self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Create tables if they don't exist
        await self.conn.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT UNIQUE NOT NULL,
                mode TEXT NOT NULL,
                prompt TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                workflow_id TEXT NOT NULL,
                model TEXT,
                image_url TEXT,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                parameters TEXT
            )
        ''')
        
        await self.conn.execute('''
            CREATE TABLE IF NOT EXISTS gallery (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                image_url TEXT NOT NULL,
                prompt TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                model TEXT,
                timestamp TEXT NOT NULL,
                featured INTEGER DEFAULT 0,
                FOREIGN KEY (job_id) REFERENCES jobs(job_id)
            )
        ''')
        
        # Create indices for performance
        await self.conn.execute('CREATE INDEX IF NOT EXISTS idx_jobs_timestamp ON jobs(timestamp)')
        await self.conn.execute('CREATE INDEX IF NOT EXISTS idx_gallery_timestamp ON gallery(timestamp)')
        await self.conn.execute('CREATE INDEX IF NOT EXISTS idx_gallery_featured ON gallery(featured)')
        
        await self.conn.commit()
        logger.info(f"Database initialized: {self.db_path}")
    
    async def save_job(self, job_data: Dict[str, Any]) -> str:
        """Save a job to the database and return job ID"""
        if not self.conn:
            await self.initialize()
        
        # Generate job ID if not provided
        job_id = job_data.get("job_id", f"job_{int(job_data['timestamp'].replace(':', '').replace('-', '').replace('T', '').replace('.', ''))}")
        
        # Convert parameters to JSON string
        parameters = json.dumps(job_data.get("parameters", {}))
        
        # Insert into jobs table
        await self.conn.execute('''
            INSERT INTO jobs 
            (job_id, mode, prompt, width, height, workflow_id, model, image_url, status, timestamp, parameters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_id,
            job_data.get("mode", "txt2img"),
            job_data.get("prompt", ""),
            job_data.get("width", 512),
            job_data.get("height", 512),
            job_data.get("workflow_id", "basic_api_test"),
            job_data.get("model"),
            job_data.get("image_url"),
            job_data.get("status", "pending"),
            job_data.get("timestamp"),
            parameters
        ))
        
        # If completed and has image URL, add to gallery
        if job_data.get("status") == "completed" and job_data.get("image_url"):
            await self.conn.execute('''
                INSERT INTO gallery
                (job_id, image_url, prompt, width, height, model, timestamp, featured)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            ''', (
                job_id,
                job_data.get("image_url"),
                job_data.get("prompt", ""),
                job_data.get("width", 512),
                job_data.get("height", 512),
                job_data.get("model"),
                job_data.get("timestamp")
            ))
        
        await self.conn.commit()
        return job_id
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a job by ID"""
        if not self.conn:
            await self.initialize()
        
        async with self.conn.execute('''
            SELECT * FROM jobs WHERE job_id = ?
        ''', (job_id,)) as cursor:
            row = await cursor.fetchone()
            
            if row:
                column_names = [description[0] for description in cursor.description]
                job = dict(zip(column_names, row))
                
                # Parse parameters JSON
                if job.get("parameters"):
                    try:
                        job["parameters"] = json.loads(job["parameters"])
                    except json.JSONDecodeError:
                        job["parameters"] = {}
                
                return job
                
            return None
    
    async def update_job_status(self, job_id: str, status: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """Update job status"""
        if not self.conn:
            await self.initialize()
        
        # Get current job data
        job = await self.get_job(job_id)
        if not job:
            return False
        
        # Update parameters with details if provided
        if details:
            parameters = job.get("parameters", {})
            parameters.update(details)
            parameters_json = json.dumps(parameters)
        else:
            parameters_json = job.get("parameters", "{}")
            if isinstance(parameters_json, dict):
                parameters_json = json.dumps(parameters_json)
        
        # Update job status
        await self.conn.execute('''
            UPDATE jobs SET status = ?, parameters = ? WHERE job_id = ?
        ''', (status, parameters_json, job_id))
        
        # If completed and has image URL, add to gallery
        if status == "completed" and job.get("image_url"):
            # Check if already in gallery
            async with self.conn.execute('''
                SELECT COUNT(*) FROM gallery WHERE job_id = ?
            ''', (job_id,)) as cursor:
                count = await cursor.fetchone()
                
                if count and count[0] == 0:
                    # Add to gallery
                    await self.conn.execute('''
                        INSERT INTO gallery
                        (job_id, image_url, prompt, width, height, model, timestamp, featured)
                        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                    ''', (
                        job_id,
                        job.get("image_url"),
                        job.get("prompt", ""),
                        job.get("width", 512),
                        job.get("height", 512),
                        job.get("model"),
                        job.get("timestamp")
                    ))
        
        await self.conn.commit()
        return True
    
    async def get_recent_jobs(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get recent jobs"""
        if not self.conn:
            await self.initialize()
        
        jobs = []
        async with self.conn.execute('''
            SELECT * FROM jobs ORDER BY timestamp DESC LIMIT ? OFFSET ?
        ''', (limit, offset)) as cursor:
            column_names = [description[0] for description in cursor.description]
            
            async for row in cursor:
                job = dict(zip(column_names, row))
                
                # Parse parameters JSON
                if job.get("parameters"):
                    try:
                        job["parameters"] = json.loads(job["parameters"])
                    except json.JSONDecodeError:
                        job["parameters"] = {}
                
                jobs.append(job)
                
        return jobs
    
    async def get_gallery_images(self, page: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
        """Get images for gallery"""
        if not self.conn:
            await self.initialize()
        
        # Calculate offset
        offset = (page - 1) * limit
        
        images = []
        
        # First get featured images
        async with self.conn.execute('''
            SELECT * FROM gallery WHERE featured = 1 ORDER BY timestamp DESC LIMIT ? OFFSET ?
        ''', (limit, offset)) as cursor:
            column_names = [description[0] for description in cursor.description]
            
            async for row in cursor:
                image = dict(zip(column_names, row))
                images.append(image)
                
        # If we don't have enough featured images, get regular images
        if len(images) < limit:
            remaining = limit - len(images)
            featured_ids = [img["id"] for img in images]
            
            id_exclusion = ""
            params = (remaining, offset)
            
            if featured_ids:
                placeholders = ",".join("?" for _ in featured_ids)
                id_exclusion = f" AND id NOT IN ({placeholders})"
                params = (*featured_ids, remaining, offset)
            
            async with self.conn.execute(f'''
                SELECT * FROM gallery WHERE featured = 0{id_exclusion} ORDER BY timestamp DESC LIMIT ? OFFSET ?
            ''', params) as cursor:
                column_names = [description[0] for description in cursor.description]
                
                async for row in cursor:
                    image = dict(zip(column_names, row))
                    images.append(image)
                    
        return images
    
    async def get_gallery_count(self) -> int:
        """Get total count of gallery images"""
        if not self.conn:
            await self.initialize()
        
        async with self.conn.execute('''
            SELECT COUNT(*) FROM gallery
        ''') as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0
            
    async def set_gallery_featured(self, image_id: int, featured: bool = True) -> bool:
        """Set an image as featured in the gallery"""
        if not self.conn:
            await self.initialize()
        
        await self.conn.execute('''
            UPDATE gallery SET featured = ? WHERE id = ?
        ''', (1 if featured else 0, image_id))
        
        await self.conn.commit()
        return True
    
    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            self.conn = None