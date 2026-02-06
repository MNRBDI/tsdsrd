# TSD RIB Multimodal RAG System

A vision-language model (VLM) based multimodal Retrieval-Augmented Generation (RAG) system for industrial safety Risk Improvement Benchmark (RIB) analysis.

## Features

- **VLM-Based Object Detection**: Replaces traditional object detection with pure VLM analysis using Qwen3-VL-32B
- **Two-Stage Detection Pipeline**:
  - Stage 1: Image description and category detection
  - Stage 2: Focused subsection scoring against relevant RIB categories
- **47 RIB Subsections**: Comprehensive coverage across 5 categories
  - PERILS (10 subsections)
  - ELECTRICAL (7 subsections)
  - HOUSEKEEPING (8 subsections)
  - HUMAN_ELEMENT (9 subsections)
  - PROCESS (16 subsections)
- **Priority Weighting**: Critical safety items receive higher scores (1.0-3.0x multipliers)
- **Multimodal RAG Pipeline**: Combines visual detection, semantic search, and LLM reasoning
- **Interactive UI**: Streamlit interface for easy querying

## Architecture

```
User Query/Image → VLM Detection → Semantic Search → Context Retrieval → LLM Recommendation
                         ↓                                    ↓
                   Top 5 RIB                          Database with
                   Subsections                        Observations
```

## System Components

### 1. VLM Detection Server (`json_prompt.py`)
- FastAPI server on port 8010
- Two-stage detection process
- Returns top 5 ranked RIB subsections with scores
- Compatible with existing RAG pipeline

### 2. Multimodal RAG System (`importance!/multimodal_rag_observation_basic.py`)
- VLLM integration (Qwen3-VL-32B on port 8000)
- PostgreSQL with pgvector for semantic search
- Embedding model: BAAI/bge-large-en-v1.5
- Three-stage pipeline:
  1. Image description generation
  2. Semantic search for matching observations
  3. Recommendation generation with context

### 3. Streamlit UI (`importance!/ui_app.py`)
- Interactive web interface
- Displays detection results, semantic matches, and recommendations
- Shows top 5 RIB subsections with rankings

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- PostgreSQL with pgvector extension
- Docker (optional, for containerized deployment)

### Setup

1. Clone the repository:
```bash
git clone git@github.com:MNRBDI/tsdsrd.git
cd tsdsrd
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your Hugging Face token
```

5. Set up PostgreSQL database:
```bash
# Create database
createdb tsdsrd

# Load RIB observations
python observation_to_postgres.py
```

## Usage

### 1. Start VLLM Server

```bash
# Using Docker Compose
docker-compose up vllm-server

# Or manually
vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

### 2. Start VLM Detection Server

```bash
source ~/VLM/.venv/bin/activate
python json_prompt.py
```

Server will start on `http://localhost:8010`

### 3. Start Streamlit UI

```bash
cd front-end
streamlit run ui_app.py
```

Access UI at `http://localhost:8501`

### 4. Test Detection API

```bash
curl -X POST "http://localhost:8010/detect-top5" \
  -F "file=@path/to/image.jpg" | jq .
```

### 5. Standalone Testing

```bash
python json_prompt.py test path/to/image.jpg
```

## API Endpoints

### VLM Detection Server

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /detect-top5` - Detect top 5 RIB subsections from image

**Request:**
```bash
POST /detect-top5
Content-Type: multipart/form-data
file: <image_file>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "tier1_detection_count": 5,
    "broad_categories": ["ELECTRICAL", "HOUSEKEEPING"],
    "primary_rib_subsection": "2.5 - Exposed Wiring",
    "top5_rib_subsections": [
      {
        "rank": 1,
        "section": "2.5 - Exposed Wiring",
        "score": 132.0,
        "detection_count": 7
      }
    ],
    "image_description": "...",
    "detected_objects": [...]
  }
}
```

## Configuration

### Environment Variables

See `.env.example` for all available configuration options:

- `HF_TOKEN` - Hugging Face API token
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` - Database configuration

### Priority Weights

Critical safety items have higher priority weights in `json_prompt.py`:

- **RIB_2.4** (Open electrical panel): 3.0x
- **RIB_2.3** (Combustible at board): 2.5x
- **RIB_2.6** (Burn marks): 2.5x
- **RIB_2.5** (Exposed wiring): 2.2x
- **RIB_1.6** (Lightning protection): 1.8x

## Project Structure

```
.
├── json_prompt.py                 # VLM detection server
├── importance!/
│   ├── multimodal_rag_observation_basic.py  # Main RAG system
│   └── ui_app.py                  # Streamlit UI
├── object_detection/
│   ├── OWLv2_tiers_improv.py     # Legacy OWLv2 detector
│   └── api_server.py             # OWLv2 API server
├── observation_chunks(ni betul).json  # RIB observations database
├── docker-compose.yml            # Container orchestration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Development

### Running Tests

```bash
# Test VLM detection
python json_prompt.py test RIB_images/page34_img1.png

# Test RAG pipeline
python importance!/multimodal_rag_observation_basic.py
```

### Adding New RIB Subsections

1. Add subsection to `observation_chunks(ni betul).json`
2. Add vocabulary to `json_prompt.py` in `tier2_rib_vocabulary`
3. Set appropriate `priority_weight` (1.0-3.0)
4. Restart VLM detection server

## Performance

- **VLM Detection**: ~10-30 seconds per image (two-stage)
- **Semantic Search**: <1 second
- **LLM Generation**: ~5-15 seconds (depends on max_tokens)
- **Total Pipeline**: ~20-50 seconds per query

## Troubleshooting

### Timeout Errors

If you encounter timeout errors:
- Check VLLM server is running: `curl http://localhost:8000/v1/models`
- Increase timeout in `json_prompt.py` (lines with `timeout=` parameter)
- Reduce image size or complexity
- Ensure GPU has sufficient VRAM

### CUDA Warnings

The warnings about CUDA capability 12.1 vs PyTorch max 12.0 can be safely ignored - the system will still work.

### Database Connection Issues

Ensure PostgreSQL is running and credentials in `.env` are correct:
```bash
psql -U amir -d tsdsrd -c "SELECT COUNT(*) FROM observation_chunks;"
```

## License

[Add your license here]

## Contributors

[Add contributors here]

## Acknowledgments

- Qwen3-VL team for the vision-language model
- VLLM project for efficient inference
- Hugging Face for model hosting
