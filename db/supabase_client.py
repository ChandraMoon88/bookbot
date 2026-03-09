"""
db/supabase_client.py
---------------------
Supabase connection clients.
Provides both the Supabase SDK client and a SQLAlchemy engine
for complex queries and transactions.
"""

import os
from supabase import create_client, Client
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

SUPABASE_URL     = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY     = os.environ.get("SUPABASE_ANON_KEY", "")       # public reads
SUPABASE_SRV_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "") # server writes

# Supabase client for CRUD operations (uses service role key server-side)
supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SRV_KEY or SUPABASE_KEY,
)

# SQLAlchemy engine for complex queries, transactions, and analytics
DATABASE_URL = os.environ.get("SUPABASE_DATABASE_URL", "")

_engine = None


def get_engine():
    global _engine
    if _engine is None and DATABASE_URL:
        _engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=300,
        )
    return _engine
