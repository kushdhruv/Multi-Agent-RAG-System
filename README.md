# Multi-Agent Document Q&A System

An advanced, multi-agent Retrieval-Augmented Generation (RAG) system designed to answer complex questions about documents with high accuracy and reliability. Features a sophisticated agent-based workflow, scalable vector database, and interactive user interface.

## ✨ Key Features

- **Multi-Agent Architecture**: Deploys specialized AI agents (Decomposer, HyDE, Synthesizer, Critic, Refiner) to ensure well-researched, context-grounded, and fact-checked answers
- **Optimized Workflow**: Efficient agent pipeline that reduces latency and cost while maintaining accuracy
- **Scalable Vector Search**: Utilizes Pinecone as a serverless vector database for fast, scalable document indexing and retrieval
- **High-Performance Backend**: Built with FastAPI, providing high-speed, asynchronous API with automatic interactive documentation
- **Modern LLMs**: Powered by Google's Gemini 1.5 Flash for fast, intelligent, and cost-effective agent operations
- **State-of-the-Art Retrieval**: Two-stage search process using bi-encoder for initial retrieval and cross-encoder for reranking
- **Custom Frontend**: Clean, responsive user interface built with HTML and Tailwind CSS

## 🏗️ System Architecture

The project's core is its multi-agent workflow, where each step is handled by a specialized agent:

1. **Decomposer Agent** (`one_decomposer_agent.py`): Breaks complex questions into simpler, atomic sub-questions
2. **HyDE Agent** (`two_hyde_agent.py`): Generates "hypothetical" answers for each sub-question to improve vector search queries
3. **Retrieval Service** (`three_retrieval_service.py`): Handles document processing, embedding, and two-stage retrieval with Pinecone
4. **Synthesizer Agent** (`four_synthesizer_agent.py`): Creates draft answers based on retrieved context
5. **Critic Agent** (`five_critic_agent.py`): Critiques answers to prevent hallucinations and ensure factual accuracy
6. **Refiner Agent** (`six_refiner_agent.py`): Performs final polish on answers when needed

This structured process ensures comprehensive, fact-checked answers against the original document.

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | FastAPI, Uvicorn | High-performance, asynchronous web framework |
| **LLM** | Google Gemini 1.5 Flash | Powering all intelligent agent operations |
| **Vector DB** | Pinecone | Scalable, cloud-based vector search |
| **Embeddings** | sentence-transformers | Creating vector representations of text |
| **PDF Processing** | PyMuPDF | Fast and accurate text extraction from PDFs |
| **Text Splitting** | langchain.text_splitter | Intelligent, token-aware text chunking |
| **Frontend** | HTML, Tailwind CSS, Vanilla JavaScript | Clean, responsive, and interactive user interface |

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A Pinecone API Key
- A Google AI API Key

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Create the environment
   python -m venv venv

   # Activate it
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Create the environment file**:
   Create a file named `.env` in the root of the project and add your API keys:
   ```env
   # Your Google AI API Key for Gemini
   GOOGLE_API_KEY="YOUR_GOOGLE_AI_KEY"

   # Your Pinecone API Key
   PINECONE_API_KEY="YOUR_PINECONE_KEY"

   # Hackathon API Bearer Token for securing the endpoint
   HACKATHON_BEARER_TOKEN="7b0bbf4d2af55715d67327967709878f6fcead8b44f13cab4d0953d0ea1e1d4e"
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

You will need two terminals running simultaneously.

**Terminal 1: Start the FastAPI Server**
```bash
uvicorn app.main:app --reload
```

Wait for the server to start. You should see Uvicorn running on `http://127.0.0.1:8000`.

**Terminal 2: (Optional) Run the Test Script**
```bash
python run_hackathon.py
```

### Using the Frontend

With the server running, open your web browser and navigate to:
```
http://127.0.0.1:8000
```

You will see the custom user interface where you can input a document URL and questions to test the system interactively.

## 🔌 API Endpoint

### Primary API Endpoint

- **URL**: `/api/v1/hackrx/run`
- **Method**: `POST`
- **Auth**: `Authorization: Bearer <your_token>`

### Request Body
```json
{
  "documents": "string (URL to a PDF document)",
  "questions": [
    "string (A list of questions)"
  ]
}
```

### Success Response (200 OK)
```json
{
  "answers": [
    "string (A list of answers corresponding to the questions)"
  ]
}
```

## 📁 Project Structure

```
BajajHack/
├── app/
│   ├── api/
│   │   └── endpoints/
│   │       └── run.py
│   ├── core/
│   │   ├── config.py
│   │   └── security.py
│   ├── schemas/
│   │   └── models.py
│   ├── services/
│   │   ├── one_decomposer_agent.py
│   │   ├── two_hyde_agent.py
│   │   ├── three_retrieval_service.py
│   │   ├── four_synthesizer_agent.py
│   │   ├── five_critic_agent.py
│   │   └── six_refiner_agent.py
│   ├── __init__.py
│   └── main.py
├── requirements.txt
├── run_hackathon.py
└── README.md
```

## 🔧 Configuration

The system uses environment variables for configuration. Key settings include:

- `GOOGLE_API_KEY`: Your Google AI API key for Gemini access
- `PINECONE_API_KEY`: Your Pinecone API key for vector database
- `HACKATHON_BEARER_TOKEN`: Security token for API authentication

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini for providing the LLM capabilities
- Pinecone for the vector database infrastructure
- FastAPI for the high-performance web framework
- The open-source community for the various libraries used in this project