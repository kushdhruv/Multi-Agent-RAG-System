from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import run as run_router
import os

# Create the FastAPI application instance
app = FastAPI(
    title="Hackathon Retrieval System API",
    description="An advanced multi-agent RAG system for document question answering.",
    version="1.0.0"
)

# --- Add CORS Middleware ---
# This allows your frontend (even on a different domain like ngrok) to communicate
# with your backend API without being blocked by browser security policies.
origins = ["*"] # For a hackathon, allowing all origins is fine.

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Add Static File Serving for the Frontend ---
# Define the path to the static directory, assuming it's in the project root
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")

# Mount the static directory to serve files like index.html, css, js
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
async def read_index():
    """
    Serves the main index.html file as the root page for the frontend.
    """
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# --- Include the API Router ---
# This incorporates the /ingest and /run endpoints from your run.py file.
app.include_router(
    run_router.router, 
    prefix="/api/v1",
    tags=["Submissions"]
)

# A simple health check endpoint to confirm the API is running
@app.get("/health", tags=["Health Check"])
def health_check():
    """
    A simple health check endpoint to confirm the API is running.
    """
    return {"status": "ok", "message": "Welcome to the Retrieval System API!"}
