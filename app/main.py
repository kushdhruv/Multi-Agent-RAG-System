from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.endpoints import run as run_router
from fastapi.middleware.cors import CORSMiddleware
import os

# Create the FastAPI application instance
app = FastAPI(
    title="Hackathon Retrieval System API",
    description="An advanced multi-agent RAG system for document question answering.",
    version="1.0.0"
)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# --- Add this section to serve the frontend ---

# Define the path to the static directory, assuming it's in the project root
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")

# Mount the static directory to serve files like CSS, JS, images
# This allows the HTML file to load its assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
async def read_index():
    """
    Serves the main index.html file as the root page for the frontend.
    """
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# --- End of frontend section ---


# Include the router for the /hackrx/run endpoint
# This keeps the API endpoint logic organized in its own module
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
    This is separate from the root '/' which now serves the frontend.
    """
    return {"status": "ok", "message": "Welcome to the Retrieval System API!"}
