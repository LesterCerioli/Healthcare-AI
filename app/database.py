import json
import os
import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Optional, Dict, Any, List, Tuple
from app.config import config
import contextlib
import uuid
from datetime import date, datetime, timedelta
from uuid import UUID

class AsyncDatabase:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.connection_string = config.DATABASE_URL  # Use a mesma connection string
    
    async def connect(self):
        """Create async connection pool"""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                dsn=self.connection_string,
                min_size=1,
                max_size=20,
                command_timeout=60,
                statement_cache_size=0  # ← ADICIONE ESTA LINHA
            )
    
    @contextlib.asynccontextmanager
    async def get_connection(self):
        """Async context manager to handle database connections"""
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as connection:
            try:
                yield connection
            finally:
                pass
    
    async def close(self):
        """Close async connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None
# Global database instance

async_db = AsyncDatabase()