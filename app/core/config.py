from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Manages application settings loaded from a .env file.
    Pydantic's BaseSettings provides type validation for configuration variables.
    """
    # Load settings from the .env file in the project's root directory
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

    # API Keys for external services
    GOOGLE_API_KEY: str
    PINECONE_API_KEY: str
    
    # Bearer token is now optional, allowing it to be provided via the UI
    HACKATHON_BEARER_TOKEN: Optional[str] = None

# Create a single, globally accessible instance of the settings
settings = Settings()
