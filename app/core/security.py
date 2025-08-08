from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

# Initialize the HTTPBearer scheme, which looks for the "Authorization" header
# with a "Bearer" token.
bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """
    A FastAPI dependency that verifies the provided Bearer token.
    
    This function is injected into endpoints that require authentication. It compares
    the token from the request's 'Authorization' header with the one stored in
    our application settings.

    Args:
        credentials: The HTTPAuthorizationCredentials object automatically extracted
                     by FastAPI from the request header.

    Raises:
        HTTPException: If the token is invalid or missing, an HTTP 403 Forbidden
                       error is raised, preventing access to the endpoint.
    """
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != settings.HACKATHON_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing authentication token.",
        )
    return credentials
