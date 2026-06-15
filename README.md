# 🥗 Nutricare Disorder Agent

An advanced **Retrieval-Augmented Generation (RAG)** chatbot that provides evidence-based information about nutritional disorders. Built as an AI Agent using LangGraph with a production-ready data pipeline.

**Portfolio Project** | AI Agent Specialization

---

## ✨ Features

- **Agentic RAG Architecture**: LangGraph-based workflow with query expansion and self-correction loops
- **Advanced Document Processing**: LlamaParse for intelligent PDF extraction with table support
- **Semantic Search**: ChromaDB vector store with HuggingFace embeddings for accurate retrieval
- **Free-Tier Friendly**: Optimized for Groq's free-tier API with automatic rate limiting
- **Production Ready**: Fully containerized, deployment-ready for HuggingFace Spaces
- **Safety Guardrails**: Llama Guard integration for user input filtering
- **Memory Management**: Mem0 integration for persistent conversation context

---

## 🏗️ Architecture

```
User Query
    ↓
[Safety Filter] - Llama Guard checks for harmful content
    ↓
[Query Expansion] - LangGraph expands query into multiple search terms
    ↓
[Semantic Retrieval] - ChromaDB + HF embeddings retrieves relevant documents
    ↓
[LLM Response] - Groq LLM generates answer using retrieved context
    ↓
[Self-Correction] - Agent validates and corrects response if needed
    ↓
Final Response
```

### Technology Stack
- **LLM**: Groq API (llama-3-70b-8192) - free tier
- **Embeddings**: HuggingFace (nomic-ai/nomic-embed-text-v1.5) - local processing
- **Vector DB**: ChromaDB - persistent storage
- **Workflow**: LangGraph - agentic orchestration
- **Parsing**: LlamaParse - intelligent document extraction
- **UI**: Streamlit - conversational interface
- **Memory**: Mem0 - conversation persistence

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- 4GB RAM (8GB recommended)
- Groq API key (free at https://console.groq.com)
- LlamaParse API key (free at https://cloud.llamaindex.ai)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nutricare-disorder-agent.git
cd nutricare-disorder-agent

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your API keys
```

### Running the Pipeline

```bash
# 1. Test configuration (30 seconds)
python main.py --mock --build

# 2. Process documents (15-30 minutes)
python main.py --data

# 3. Build agent workflow (1 minute)
python main.py --build

# 4. Run the Streamlit app
streamlit run pipelines/streamlit_app.py
```

For detailed setup, see [QUICKSTART.md](QUICKSTART.md)

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [QUICKSTART.md](QUICKSTART.md) | 5-minute setup guide with examples |
| [SETUP_FIXES.md](SETUP_FIXES.md) | Detailed explanation of migration from Great Learning to Groq |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Solutions for 20+ common issues |
| [FIX_SUMMARY.md](FIX_SUMMARY.md) | Technical summary of all fixes applied |

---

## 🔑 Key Configuration

```bash
# API Endpoints (Groq OpenAI-compatible)
OPENAI_API_BASE=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_xxxxxxxxxxxxxxx

# LLM Model (latest Groq model)
OPENAI_MODEL=llama-3-70b-8192

# Embeddings (local HuggingFace, no rate limits)
OPENAI_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

# Safety Filter
LLAMA_KEY=llx_xxxxxxxxxxxxxxx
LLAMA_MODEL=mixtral-8x7b-32768
```

See [env.example](env.example) for complete template.

---

## 📊 Project Structure

```
nutricare-disorder-agent/
├── main.py                          # Application entry point & CLI parameter parser
├── requirements.txt                 # Pinned python framework dependencies
├── .env                             # Environment configuration (git-ignored)
│
├── models/                          # Core Intelligence & Class Wrappers
│   ├── openai.py                    # LLM orchestration & HuggingFace embedding configuration
│   ├── chroma.py                    # Vector store setups and collection initialization
│   ├── llama.py                     # Safety guardrails and Llama Guard compliance validation
│   └── nutrition_bot.py             # Agent conversational interface & Mem0 transactional logic
│
├── pipelines/                       # State Graph Components & Workflows
│   ├── data_processor.py            # Ingestion pipeline (Parsing, Chunking, Indexing)
│   ├── agent.py                     # LangGraph workflow compile map and self-correction routing
│   ├── huggingface.py               # HuggingFace Spaces container configuration logic
│   └── streamlit_app.py             # Web Interface UI
│
├── storages/                        # Synthetic Data Generation
│   ├── question_generator.py        # Normal text chunk hypothetical Q generator
│   └── table_question_generator.py  # Matrix/Table structure question extraction
│
├── src/                             # Core System Utilities
│   ├── config.py                    # Dynamic environment parser and variable mapper
│   ├── doc_handler.py               # File system paths & raw PDF data IO helpers
│   ├── eda.py                       # Exploratory analysis checking text densities
│   └── utils.py                     # Global helper variables and formatting utilities
│
└── data/
    └── nutritional-medical-reference/
        └── nutritional-disorders.pdf    # Source clinical data corpus 
```

---

## 💡 How It Works

### 1. Data Processing Pipeline
- **PDF Extraction**: LlamaParse intelligently extracts text and tables
- **Semantic Chunking**: Documents split by meaning, not size
- **Hypothetical Questions**: LLM generates Q&A pairs for better retrieval
- **Vectorization**: HuggingFace embeddings create dense vectors
- **Storage**: ChromaDB persists vectors for fast retrieval

### 2. Agentic RAG Workflow
- **Query Understanding**: Expands single query into multiple search variations
- **Semantic Search**: Retrieves top-k relevant document chunks
- **Context Assembly**: Combines retrieved content intelligently
- **LLM Generation**: Groq LLM generates informed response
- **Validation Loop**: Self-correction agent verifies answer quality

### 3. Deployment
- **HuggingFace Spaces**: Fully containerized, ready-to-deploy
- **Streamlit UI**: Conversational interface for users
- **Vector DB**: Persists in Docker volume for consistency

---

## 🎯 Use Cases

- **Educational**: Understand nutritional deficiencies and treatments
- **Medical Reference**: Extract information from nutritional guidelines
- **Research**: Test RAG and agentic LLM architectures
- **Portfolio**: Demonstrate end-to-end AI product development

---

## 🔧 Troubleshooting

### Common Issues

**Rate Limit Errors?**
→ Automatic handling built-in; See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**Model Download Taking Long?**
→ First run expects 10-15 min for 400MB embedding model; subsequent runs are instant

**API Authentication Failing?**
→ Check your API keys in `.env`; see [SETUP_FIXES.md](SETUP_FIXES.md)

For more solutions, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 📈 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| First Setup | 30-45 min | Includes model download |
| Document Processing | 20-30 min | 200+ page PDF with throttling |
| Query Response Time | 5-10 sec | Including retrieval and generation |
| Vector Store Size | ~2GB | Persisted in `./db3/` |
| API Rate Limit | 30 req/min | Free tier Groq |

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- ✅ Retrieval-Augmented Generation (RAG) systems
- ✅ Agent-based workflows (LangGraph)
- ✅ Vector databases and semantic search
- ✅ LLM API integration and rate limiting
- ✅ Document parsing and chunking strategies
- ✅ Production deployment (containerization)
- ✅ Error handling and graceful degradation
- ✅ Full-stack AI application development

---

## 🚀 Deployment

### Local Deployment
```bash
python main.py --run
# Opens http://localhost:8501
```

### HuggingFace Spaces Deployment
```bash
# Update .env with HF credentials
HF_REPO_ID=your-username/nutricare-agent
HF_TOKEN=hf_xxxxx

# Deploy
python main.py --deploy
```

### Docker
```bash
docker build -f Dockerfile.local -t nutricare-agent .
docker run -p 8501:8501 --env-file .env nutricare-agent
```

---

## 📝 License

This project was developed as part of the University of Texas / McCombs AI Agent specialization.

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional data sources integration
- Multi-language support
- Enhanced visualization
- Performance optimization
- Additional safety checks

---

## 📞 Support

- 📖 **Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
- 🔧 **Troubleshooting**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- 🏗️ **Architecture**: See [SETUP_FIXES.md](SETUP_FIXES.md)
- 🐛 **Debugging**: Run with `--log` flag for verbose output

---

## 🎯 What's Next?

1. ✅ Process your nutritional data
2. ✅ Test queries via Streamlit UI
3. ✅ Deploy to HuggingFace Spaces
4. ✅ Share in your portfolio

**[Start with QUICKSTART.md →](QUICKSTART.md)**

---

**Built with ❤️ using LangChain, Groq, and HuggingFace**
**Perfect for AI/ML job applications** 🚀
