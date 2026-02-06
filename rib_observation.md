# TSD RIB Observation Search System

This system loads Risk Improvement Benchmark (RIB) chunks into PostgreSQL and enables semantic search based on observations using vector embeddings.

## Overview

The system consists of two main components:

1. **Data Loader** (`load_rib_chunks_to_postgres.py`) - Loads RIB chunks from JSON into PostgreSQL with observation embeddings
2. **Semantic Search** (`search_rib_observations.py`) - Searches observations using natural language queries

## Database Schema

### Table: `tsd_rib`
Main table storing RIB observations and recommendations.

```sql
CREATE TABLE tsd_rib (
    id VARCHAR(50) PRIMARY KEY,
    section_number VARCHAR(20) NOT NULL,
    title TEXT NOT NULL,
    category VARCHAR(50) NOT NULL,
    observation TEXT,
    recommendation TEXT,
    regulation TEXT,
    full_text TEXT,
    observation_embedding vector(1024),  -- Embedding of observation for search
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Table: `tsd_metadata`
Metadata about each chunk.

```sql
CREATE TABLE tsd_metadata (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(50) UNIQUE NOT NULL REFERENCES tsd_rib(id),
    section VARCHAR(20),
    title TEXT,
    category VARCHAR(50),
    observation_length INTEGER,
    recommendation_length INTEGER,
    regulation_length INTEGER,
    has_observation BOOLEAN DEFAULT FALSE,
    has_recommendation BOOLEAN DEFAULT FALSE,
    has_regulation BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Prerequisites

### 1. Install Required Packages

```bash
pip install psycopg2-binary sentence-transformers torch tqdm
```

### 2. Install PostgreSQL pgvector Extension

```bash
# Ubuntu/Debian
sudo apt install postgresql-16-pgvector

# Or compile from source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

Enable in your database:
```sql
CREATE EXTENSION vector;
```

### 3. Prepare Your Data

Ensure you have your RIB chunks JSON file (e.g., `rib_chunks_fixed_v3.json`) with the following structure:

```json
[
  {
    "id": "Process_5.13",
    "section_number": "5.13",
    "title": "PIPELINE FOR GAS SUPPLY FOR COMMERCIAL BUILDING / APARTMENT",
    "category": "Process",
    "observation": "The apartment has been using...",
    "recommendation": "Over time, gas pipelines...",
    "regulation": "1. NFPA 54 – National Fuel Gas Code...",
    "full_text": "Section 5.13: PIPELINE...",
    "metadata": {
      "section": "5.13",
      "title": "PIPELINE FOR GAS SUPPLY...",
      "category": "Process"
    }
  }
]
```

## Usage

### Step 1: Load Data into Database

Edit database configuration in `load_rib_chunks_to_postgres.py`:

```python
db_config = {
    'host': 'localhost',
    'database': 'tsdsrd',
    'user': 'amir',
    'password': 'amir123',
    'port': 5432
}

# Specify your JSON file path
json_file = 'rib_chunks_fixed_v3.json'
```

Run the loader:

```bash
python load_rib_chunks_to_postgres.py
```

This will:
1. Create the `tsd_rib` and `tsd_metadata` tables
2. Generate embeddings for all observations
3. Insert chunks into the database
4. Create vector indexes for fast similarity search
5. Verify the data was loaded correctly

### Step 2: Search Observations

Run the search script:

```bash
python search_rib_observations.py
```

This provides:
- Example searches with common queries
- Interactive search mode where you can enter custom queries

#### Programmatic Usage

```python
from search_rib_observations import RIBSemanticSearch

# Initialize
db_config = {
    'host': 'localhost',
    'database': 'tsdsrd',
    'user': 'amir',
    'password': 'amir123',
    'port': 5432
}

search = RIBSemanticSearch(db_config)

# Search for observations
results = search.search_observations(
    query="gas pipeline corrosion",
    limit=5,
    category_filter="Process",  # Optional
    similarity_threshold=0.5    # Optional: 0-1 similarity score
)

# Display results
search.print_results(results, show_full_text=True)
```

## Example Queries

Here are some example queries you can try:

```python
# General safety queries
"How to prevent gas leaks?"
"Electrical safety procedures"
"Fire prevention measures"

# Specific technical queries
"Pipeline maintenance schedule"
"Corrosion protection methods"
"Emergency shutdown procedures"

# Category-specific queries (with filter)
query = "inspection requirements"
category_filter = "Electrical"
```

## Query Performance

The system uses HNSW (Hierarchical Navigable Small World) indexing for vector similarity search, providing:
- Fast query times (< 10ms for most queries)
- High recall accuracy
- Efficient for large datasets

## Embedding Model

Default model: **BAAI/bge-large-en-v1.5** (1024 dimensions)

This model provides:
- High quality semantic understanding
- Good performance on technical/regulatory text
- Normalized embeddings for cosine similarity

Alternative models you can use:
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, fastest)
- `BAAI/bge-base-en-v1.5` (768 dim, balanced)
- `intfloat/e5-large-v2` (1024 dim, excellent quality)

To change the model, edit both scripts:
```python
model_name = "BAAI/bge-large-en-v1.5"  # Change this
```

**Important:** Use the same model for loading and searching!

## SQL Examples

### Direct SQL Queries

```sql
-- Get all observations from a category
SELECT section_number, title, observation
FROM tsd_rib
WHERE category = 'Process'
  AND observation IS NOT NULL;

-- Search by section number
SELECT * FROM tsd_rib
WHERE section_number LIKE '5.%';

-- Get chunks with recommendations
SELECT r.section_number, r.title, r.recommendation
FROM tsd_rib r
JOIN tsd_metadata m ON r.id = m.chunk_id
WHERE m.has_recommendation = TRUE;

-- Semantic search (you need query embedding first)
SELECT id, title, observation,
       1 - (observation_embedding <=> '[your_embedding]'::vector) as similarity
FROM tsd_rib
WHERE observation IS NOT NULL
ORDER BY observation_embedding <=> '[your_embedding]'::vector
LIMIT 10;
```

## Troubleshooting

### Issue: "extension vector does not exist"
**Solution:** Install pgvector extension and run `CREATE EXTENSION vector;`

### Issue: Embeddings are not normalized
**Solution:** Check that `normalize_embeddings=True` in the encoder

### Issue: Slow queries
**Solution:** Ensure HNSW index is created:
```sql
CREATE INDEX idx_tsd_rib_observation_embedding 
ON tsd_rib USING hnsw (observation_embedding vector_cosine_ops);
```

### Issue: Out of memory during loading
**Solution:** Reduce batch size:
```python
loader.load_all_chunks(json_file, batch_size=16)  # Default is 32
```

## Database Maintenance

### Reindex vectors (if needed)
```sql
REINDEX INDEX idx_tsd_rib_observation_embedding;
```

### Check index usage
```sql
EXPLAIN ANALYZE
SELECT id, title
FROM tsd_rib
ORDER BY observation_embedding <=> '[some_embedding]'::vector
LIMIT 10;
```

### Backup data
```bash
pg_dump -U amir -d tsdsrd -t tsd_rib -t tsd_metadata > rib_backup.sql
```

## Integration with RAG System

To integrate with a RAG (Retrieval-Augmented Generation) system:

```python
# 1. Get relevant observations
search = RIBSemanticSearch(db_config)
results = search.search_observations(
    query=user_question,
    limit=5,
    similarity_threshold=0.6
)

# 2. Format context for LLM
context = ""
for result in results:
    context += f"Section {result['section_number']}: {result['title']}\n"
    context += f"Observation: {result['observation']}\n"
    context += f"Recommendation: {result['recommendation']}\n\n"

# 3. Send to LLM
prompt = f"""Based on the following RIB observations:

{context}

Answer this question: {user_question}
"""

# Send to your LLM (e.g., OpenAI, Anthropic, etc.)
```

## Performance Metrics

With ~50-100 chunks:
- Loading time: ~30 seconds (including embedding generation)
- Query time: < 10ms per query
- Index size: ~5-10MB

## License

This code is provided as-is for your TSD RIB project.