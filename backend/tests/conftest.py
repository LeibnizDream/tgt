"""
Set credentials before any module import so routers.auth doesn't raise ValueError.
These must be at module level (not inside a fixture) because auth.py reads
os.environ at import time.
"""
import os

os.environ.setdefault("TENANT_ID", "test-tenant-id")
os.environ.setdefault("CLIENT_ID", "test-client-id")
os.environ.setdefault("CLIENT_SECRET", "test-client-secret")
