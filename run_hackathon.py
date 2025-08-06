import httpx
import json
import os
from dotenv import load_dotenv

# Load environment variables from.env file
load_dotenv()

# --- Configuration ---
# The URL of your running FastAPI application
API_BASE_URL = "http://localhost:8000/api/v1"
# The specific endpoint for the submission
ENDPOINT_URL = f"{API_BASE_URL}/hackrx/run"
# Your bearer token from the.env file
BEARER_TOKEN = os.getenv("HACKATHON_BEARER_TOKEN")

# The payload as specified in the hackathon documentation
REQUEST_PAYLOAD = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        # Example question(s); replace with actual questions as needed
        "What is the policy coverage period?"
    ]
}

def run_test():
    """
    Sends a POST request to the API with the hackathon payload and prints the response.
    """
    if not BEARER_TOKEN:
        print("Error: HACKATHON_BEARER_TOKEN not found in.env file.")
        return

    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    print("--- Sending Request to API ---")
    print(f"URL: {ENDPOINT_URL}")
    print("Payload:")
    print(json.dumps(REQUEST_PAYLOAD, indent=2))
    print("-----------------------------")

    try:
        # Use a longer timeout as the first run might involve model downloads
        with httpx.Client(timeout=300.0) as client:
            response = client.post(ENDPOINT_URL, headers=headers, json=REQUEST_PAYLOAD)

        print("\n--- Received Response from API ---")
        print(f"Status Code: {response.status_code}")
        
        # Check if the request was successful
        if response.status_code == 200:
            print("Response Body (Formatted):")
            # Pretty-print the JSON response
            response_json = response.json()
            print(json.dumps(response_json, indent=2))
        else:
            # Print error details if something went wrong
            print("Error Response:")
            try:
                print(json.dumps(response.json(), indent=2))
            except json.JSONDecodeError:
                print(response.text)

    except httpx.ConnectError as e:
        print(f"\nConnection Error: Could not connect to the API at {API_BASE_URL}.")
        print("Please ensure the FastAPI server is running with the command:")
        print("uvicorn app.main:app --reload")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    run_test()