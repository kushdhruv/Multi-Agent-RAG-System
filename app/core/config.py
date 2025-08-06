from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Manages application settings loaded from a.env file.
    Pydantic's BaseSettings provides type validation for configuration variables.
    """
    # Load settings from a.env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8', extra='ignore')

    # OpenAI API Key for LLM calls
    OPENAI_API_KEY: str

    # Bearer token for securing the API endpoint
    HACKATHON_BEARER_TOKEN: str
    
    # --- NEW: Pinecone API Key ---
    # API Key for connecting to the Pinecone vector database
    PINECONE_API_KEY: str
    GOOGLE_API_KEY: str # Add this line

# Create a single, globally accessible instance of the settings
settings = Settings()                                                       