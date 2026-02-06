# load_rib_chunks_to_postgres.py

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import execute_batch, Json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

class TSDRIBLoader:
    def __init__(self, db_config: Dict[str, str], model_name: str = None):
        """
        Initialize the loader with database and HuggingFace model configurations
        
        Args:
            db_config: Dictionary with keys: host, database, user, password, port
            model_name: HuggingFace model name for embeddings
        """
        self.db_config = db_config
        self.conn = None
        
        # Choose embedding model
        if model_name is None:
            # Default to a good general-purpose model
            model_name = "BAAI/bge-large-en-v1.5"  # 1024 dimensions, high quality
        
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        
        # Load the model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✓ Model loaded on {device}")
        print(f"✓ Embedding dimension: {self.embedding_dim}")
        
    def connect_db(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                port=self.db_config.get('port', 5432)
            )
            print("✓ Connected to PostgreSQL database")
            return self.conn
        except Exception as e:
            print(f"✗ Error connecting to database: {e}")
            raise
    
    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")
    
    def create_tables(self):
        """Create the tsd_rib and tsd_metadata tables"""
        self.connect_db()
        cursor = self.conn.cursor()
        
        try:
            print("Creating tables...")
            
            # Enable pgvector extension if not already enabled
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Drop existing tables if they exist (optional - comment out if you want to preserve data)
            # cursor.execute("DROP TABLE IF EXISTS tsd_metadata CASCADE;")
            # cursor.execute("DROP TABLE IF EXISTS tsd_rib CASCADE;")
            
            # Create tsd_rib table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS tsd_rib (
                    id VARCHAR(50) PRIMARY KEY,
                    section_number VARCHAR(20) NOT NULL,
                    title TEXT NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    observation TEXT,
                    recommendation TEXT,
                    regulation TEXT,
                    full_text TEXT,
                    observation_embedding vector({self.embedding_dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create tsd_metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tsd_metadata (
                    id SERIAL PRIMARY KEY,
                    chunk_id VARCHAR(50) UNIQUE NOT NULL REFERENCES tsd_rib(id) ON DELETE CASCADE,
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
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tsd_rib_category 
                ON tsd_rib(category);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tsd_rib_section 
                ON tsd_rib(section_number);
            """)
            
            # Create vector index for similarity search (using HNSW for better performance)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tsd_rib_observation_embedding 
                ON tsd_rib USING hnsw (observation_embedding vector_cosine_ops);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tsd_metadata_chunk_id 
                ON tsd_metadata(chunk_id);
            """)
            
            self.conn.commit()
            print("✓ Tables created successfully")
            
        except Exception as e:
            print(f"✗ Error creating tables: {e}")
            self.conn.rollback()
            raise
        finally:
            cursor.close()
            self.close_db()
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using HuggingFace model
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        if not text or not text.strip():
            # Return zero vector if text is empty
            return [0.0] * self.embedding_dim
        
        try:
            # For BGE models, add instruction prefix for better retrieval
            if "bge" in self.model_name.lower():
                text = f"Represent this observation for retrieval: {text}"
            
            # Generate embedding
            embedding = self.model.encode(
                text,
                normalize_embeddings=True,  # Normalize for cosine similarity
                show_progress_bar=False
            )
            
            return embedding.tolist()
        except Exception as e:
            print(f"  ✗ Failed to generate embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch (more efficient)
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            
        Returns:
            List of embeddings
        """
        try:
            # Handle empty texts
            processed_texts = []
            for text in texts:
                if text and text.strip():
                    # For BGE models, add instruction prefix
                    if "bge" in self.model_name.lower():
                        processed_texts.append(f"Represent this observation for retrieval: {text}")
                    else:
                        processed_texts.append(text)
                else:
                    processed_texts.append("")  # Empty text placeholder
            
            # Generate embeddings in batch
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            print(f"  ✗ Failed to generate batch embeddings: {e}")
            raise
    
    def load_chunks_from_json(self, json_file: str) -> List[Dict[str, Any]]:
        """
        Load chunks from JSON file
        
        Args:
            json_file: Path to JSON file containing chunks
            
        Returns:
            List of chunk dictionaries
        """
        print(f"Loading chunks from {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"✓ Loaded {len(chunks)} chunks")
        return chunks
    
    def insert_chunk_with_embedding(self, chunk: Dict[str, Any], embedding: List[float], cursor):
        """
        Insert a single chunk with its embedding into the database
        
        Args:
            chunk: Chunk dictionary
            embedding: Pre-computed embedding for the observation
            cursor: Database cursor
        """
        # Extract metadata
        metadata = chunk.get('metadata', {})
        
        # Insert into tsd_rib table
        cursor.execute("""
            INSERT INTO tsd_rib (
                id, section_number, title, category,
                observation, recommendation, regulation, full_text,
                observation_embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                section_number = EXCLUDED.section_number,
                title = EXCLUDED.title,
                category = EXCLUDED.category,
                observation = EXCLUDED.observation,
                recommendation = EXCLUDED.recommendation,
                regulation = EXCLUDED.regulation,
                full_text = EXCLUDED.full_text,
                observation_embedding = EXCLUDED.observation_embedding,
                updated_at = CURRENT_TIMESTAMP
        """, (
            chunk['id'],
            chunk['section_number'],
            chunk['title'],
            chunk['category'],
            chunk.get('observation', ''),
            chunk.get('recommendation', ''),
            chunk.get('regulation', ''),
            chunk.get('full_text', ''),
            embedding
        ))
        
        # Calculate lengths and boolean flags
        obs_text = chunk.get('observation', '')
        rec_text = chunk.get('recommendation', '')
        reg_text = chunk.get('regulation', '')
        
        # Insert into tsd_metadata table
        cursor.execute("""
            INSERT INTO tsd_metadata (
                chunk_id, section, title, category,
                observation_length, recommendation_length, regulation_length,
                has_observation, has_recommendation, has_regulation
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                section = EXCLUDED.section,
                title = EXCLUDED.title,
                category = EXCLUDED.category,
                observation_length = EXCLUDED.observation_length,
                recommendation_length = EXCLUDED.recommendation_length,
                regulation_length = EXCLUDED.regulation_length,
                has_observation = EXCLUDED.has_observation,
                has_recommendation = EXCLUDED.has_recommendation,
                has_regulation = EXCLUDED.has_regulation
        """, (
            chunk['id'],
            chunk['section_number'],
            chunk['title'],
            chunk['category'],
            len(obs_text),
            len(rec_text),
            len(reg_text),
            bool(obs_text.strip()),
            bool(rec_text.strip()),
            bool(reg_text.strip())
        ))
    
    def load_all_chunks(self, json_file: str, batch_size: int = 32):
        """
        Load all chunks from JSON file and insert them into the database
        Uses batch embedding generation for efficiency
        
        Args:
            json_file: Path to JSON file containing chunks
            batch_size: Number of chunks to embed at once
        """
        chunks = self.load_chunks_from_json(json_file)
        
        if not chunks:
            print("✗ No chunks found to load")
            return
        
        print(f"\nGenerating embeddings for {len(chunks)} observations...")
        
        # Extract all observation texts for embedding
        observation_texts = [chunk.get('observation', '') for chunk in chunks]
        
        # Generate all embeddings in batches (much faster!)
        embeddings = self.generate_embeddings_batch(observation_texts, batch_size=batch_size)
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        
        self.connect_db()
        cursor = self.conn.cursor()
        
        try:
            print(f"\nInserting chunks into database...")
            
            for i, (chunk, embedding) in enumerate(tqdm(zip(chunks, embeddings),
                                                         total=len(chunks),
                                                         desc="Inserting chunks")):
                try:
                    self.insert_chunk_with_embedding(chunk, embedding, cursor)
                    
                    # Commit every 10 chunks
                    if (i + 1) % 10 == 0:
                        self.conn.commit()
                
                except Exception as e:
                    print(f"  ✗ Error inserting chunk {chunk.get('id', 'unknown')}: {e}")
                    self.conn.rollback()
                    continue
            
            # Final commit
            self.conn.commit()
            print(f"\n✓ Successfully loaded {len(chunks)} chunks into database")
            
        except Exception as e:
            print(f"✗ Error during bulk load: {e}")
            self.conn.rollback()
            raise
        finally:
            cursor.close()
            self.close_db()
    
    def verify_data(self):
        """Verify that data was loaded correctly"""
        self.connect_db()
        cursor = self.conn.cursor()
        
        try:
            # Count chunks
            cursor.execute("SELECT COUNT(*) FROM tsd_rib")
            chunk_count = cursor.fetchone()[0]
            print(f"\nTotal chunks in database: {chunk_count}")
            
            # Count chunks with embeddings
            cursor.execute("SELECT COUNT(*) FROM tsd_rib WHERE observation_embedding IS NOT NULL")
            embedding_count = cursor.fetchone()[0]
            print(f"Chunks with observation embeddings: {embedding_count}")
            
            # Count chunks with observations
            cursor.execute("SELECT COUNT(*) FROM tsd_metadata WHERE has_observation = TRUE")
            obs_count = cursor.fetchone()[0]
            print(f"Chunks with observations: {obs_count}")
            
            # Show category distribution
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM tsd_rib
                GROUP BY category
                ORDER BY count DESC
            """)
            print("\nCategory distribution:")
            for row in cursor.fetchall():
                print(f"  {row[0]}: {row[1]}")
            
            # Show sample chunk with embedding info
            cursor.execute("""
                SELECT r.id, r.title, r.section_number,
                       m.observation_length, m.recommendation_length, m.regulation_length,
                       vector_dims(r.observation_embedding) as embedding_dim
                FROM tsd_rib r
                JOIN tsd_metadata m ON r.id = m.chunk_id
                WHERE r.observation_embedding IS NOT NULL
                  AND m.has_observation = TRUE
                LIMIT 3
            """)
            print(f"\nSample chunks:")
            for row in cursor.fetchall():
                print(f"\n  ID: {row[0]}")
                print(f"  Section: {row[2]}")
                print(f"  Title: {row[1][:60]}...")
                print(f"  Observation length: {row[3]} chars")
                print(f"  Recommendation length: {row[4]} chars")
                print(f"  Regulation length: {row[5]} chars")
                print(f"  Embedding dimensions: {row[6]}")
            
            # Check if embeddings are normalized (for cosine similarity)
            cursor.execute("""
                SELECT id,
                       (observation_embedding <#> observation_embedding) as magnitude
                FROM tsd_rib
                WHERE observation_embedding IS NOT NULL
                  AND observation_embedding <> ARRAY_FILL(0::float, ARRAY[vector_dims(observation_embedding)])
                LIMIT 5
            """)
            print("\nSample embedding magnitudes (should be ~1.0 if normalized):")
            for row in cursor.fetchall():
                print(f"  {row[0]}: {row[1]:.4f}")
            
            # Test a sample similarity search
            print("\n--- Testing Similarity Search ---")
            cursor.execute("""
                SELECT id, title, section_number
                FROM tsd_rib
                WHERE observation IS NOT NULL 
                  AND observation != ''
                LIMIT 1
            """)
            sample = cursor.fetchone()
            
            if sample:
                sample_id = sample[0]
                print(f"Finding similar chunks to: {sample[1][:60]}...")
                
                cursor.execute("""
                    SELECT r1.id, r1.title, r1.section_number,
                           (r1.observation_embedding <=> r2.observation_embedding) as distance
                    FROM tsd_rib r1
                    CROSS JOIN tsd_rib r2
                    WHERE r2.id = %s
                      AND r1.id != %s
                      AND r1.observation_embedding IS NOT NULL
                    ORDER BY distance
                    LIMIT 3
                """, (sample_id, sample_id))
                
                print("\nTop 3 most similar chunks:")
                for row in cursor.fetchall():
                    print(f"  {row[2]}: {row[1][:60]}... (distance: {row[3]:.4f})")
            
        except Exception as e:
            print(f"✗ Error during verification: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cursor.close()
            self.close_db()


def main():
    """Main execution function"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'tsdsrd',
        'user': 'amir',
        'password': 'amir123',
        'port': 5432
    }
    
    # Choose your embedding model
    # Options (from fastest/smallest to slowest/best):
    # 1. "sentence-transformers/all-MiniLM-L6-v2" - 384 dim, very fast
    # 2. "BAAI/bge-base-en-v1.5" - 768 dim, good balance
    # 3. "BAAI/bge-large-en-v1.5" - 1024 dim, high quality (RECOMMENDED)
    # 4. "intfloat/e5-large-v2" - 1024 dim, excellent quality
    
    model_name = "BAAI/bge-large-en-v1.5"
    
    # Path to your chunks JSON file
    json_file = '/home/amir/Desktop/MRE TSD/observation_chunks.json'
    
    # Initialize loader
    loader = TSDRIBLoader(db_config, model_name=model_name)
    
    # Create tables
    print("Step 1: Creating database tables...\n")
    loader.create_tables()
    
    # Load chunks and generate embeddings
    print("\nStep 2: Loading chunks and generating embeddings...\n")
    loader.load_all_chunks(json_file, batch_size=32)
    
    # Verify the data
    print("\nStep 3: Verifying loaded data...")
    loader.verify_data()
    
    print("\n✓ Process completed successfully!")
    print("\nYou can now use semantic search on observations with queries like:")
    print("  SELECT id, title, observation")
    print("  FROM tsd_rib")
    print("  ORDER BY observation_embedding <=> '[your_query_embedding]'")
    print("  LIMIT 5;")


if __name__ == "__main__":
    main()