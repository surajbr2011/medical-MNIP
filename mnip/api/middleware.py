import time
import logging
from fastapi import Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from mnip.config import settings

logger = logging.getLogger("mnip")
logging.basicConfig(level=logging.INFO)

class LoggingAndSecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Optional: Basic validation of auth header if present
        auth_header = request.headers.get("Authorization")
        if auth_header and not auth_header.startswith("Bearer "):
            # Let it proceed or validate if JWT token verification is required.
            # In a production app, we would decode settings.JWT_SECRET.
            pass
            
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"Method: {request.method} Path: {request.url.path} "
            f"Status: {response.status_code} Latency: {process_time:.4f}s"
        )
        response.headers["X-Process-Time"] = str(process_time)
        return response

def setup_middlewares(app):
    # Setup CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom logging and security headers middleware
    app.add_middleware(LoggingAndSecurityMiddleware)
