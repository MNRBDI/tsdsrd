# load_chunks_to_postgres.py

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import execute_batch, Json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

class ChunkEmbeddingLoader:
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
            # Other good options:
            # "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions, fast
            # "sentence-transformers/all-mpnet-base-v2"  # 768 dimensions, balanced
            # "BAAI/bge-base-en-v1.5"  # 768 dimensions, good quality
            # "intfloat/e5-large-v2"  # 1024 dimensions, excellent
        
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
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using HuggingFace model
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            # For BGE models, add instruction prefix for better retrieval
            if "bge" in self.model_name.lower():
                text = f"Represent this document for retrieval: {text}"
            
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
            # For BGE models, add instruction prefix
            if "bge" in self.model_name.lower():
                texts = [f"Represent this document for retrieval: {text}" for text in texts]
            
            # Generate embeddings in batch
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            print(f"  ✗ Failed to generate batch embeddings: {e}")
            raise
    
    def load_chunks_from_json(self, chunks_dir: str) -> List[Dict[str, Any]]:
        """
        Load all chunk JSON files from directory
        
        Args:
            chunks_dir: Path to directory containing chunk JSON files
            
        Returns:
            List of chunk dictionaries
        """
        chunks_path = Path(chunks_dir)
        
        # Check if all_chunks.json exists
        all_chunks_file = chunks_path / 'rib_chunks_complete.json'
        if all_chunks_file.exists():
            print(f"Loading chunks from {all_chunks_file}")
            with open(all_chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            print(f"✓ Loaded {len(chunks)} chunks from all_chunks.json")
            return chunks
        
        # Otherwise load individual files
        print(f"Loading individual chunk files from {chunks_dir}")
        chunks = []
        json_files = list(chunks_path.glob('*.json'))
        
        for json_file in json_files:
            if json_file.name != 'all_chunks.json':
                with open(json_file, 'r', encoding='utf-8') as f:
                    chunk = json.load(f)
                    chunks.append(chunk)
        
        print(f"✓ Loaded {len(chunks)} chunks from individual files")
        return chunks
    
    def insert_chunk_with_embedding(self, chunk: Dict[str, Any], embedding: List[float], cursor):
        """
        Insert a single chunk with its embedding into the database
        
        Args:
            chunk: Chunk dictionary
            embedding: Pre-computed embedding
            cursor: Database cursor
        """
        # Insert into rib_chunks table
        cursor.execute("""
            INSERT INTO rib_chunks (id, category, title, content, token_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                category = EXCLUDED.category,
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                token_count = EXCLUDED.token_count,
                embedding = EXCLUDED.embedding
        """, (
            chunk['id'],
            chunk['category'],
            chunk['title'],
            chunk['content'],
            chunk['token_count'],
            embedding
        ))
        
        # Insert into chunk_metadata table
        metadata = chunk['metadata']
        cursor.execute("""
            INSERT INTO chunk_metadata (
                chunk_id, section_number, section_title, category,
                regulations, keywords, risk_type
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
                section_number = EXCLUDED.section_number,
                section_title = EXCLUDED.section_title,
                category = EXCLUDED.category,
                regulations = EXCLUDED.regulations,
                keywords = EXCLUDED.keywords,
                risk_type = EXCLUDED.risk_type
        """, (
            chunk['id'],
            metadata['section_number'],
            metadata['section_title'],
            metadata['category'],
            Json(metadata['regulations']),
            Json(metadata['keywords']),
            metadata['risk_type']
        ))
    
    def load_all_chunks(self, chunks_dir: str, batch_size: int = 32):
        """
        Load all chunks from JSON files and insert them into the database
        Uses batch embedding generation for efficiency
        
        Args:
            chunks_dir: Path to directory containing chunk JSON files
            batch_size: Number of chunks to embed at once
        """
        chunks = self.load_chunks_from_json(chunks_dir)
        
        if not chunks:
            print("✗ No chunks found to load")
            return
        
        print(f"\nGenerating embeddings for {len(chunks)} chunks...")
        
        # Extract all content texts
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate all embeddings in batches (much faster!)
        embeddings = self.generate_embeddings_batch(texts, batch_size=batch_size)
        
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
                    print(f"  ✗ Error inserting chunk {chunk['id']}: {e}")
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
            cursor.execute("SELECT COUNT(*) FROM rib_chunks")
            chunk_count = cursor.fetchone()[0]
            print(f"\nTotal chunks in database: {chunk_count}")
        
        # Count chunks with embeddings
            cursor.execute("SELECT COUNT(*) FROM rib_chunks WHERE embedding IS NOT NULL")
            embedding_count = cursor.fetchone()[0]
            print(f"Chunks with embeddings: {embedding_count}")
        
        # Show category distribution
            cursor.execute("""
                SELECT category, COUNT(*) as count 
                FROM rib_chunks 
                GROUP BY category 
                ORDER BY count DESC
                """)
            print("\nCategory distribution:")
            for row in cursor.fetchall():
                print(f"  {row[0]}: {row[1]}")
        
        # Show sample chunk with embedding info
        # FIX: Use vector_dims() instead of array_length()
            cursor.execute("""
                SELECT id, title, token_count, 
                vector_dims(embedding) as embedding_dim
                FROM rib_chunks 
                WHERE embedding IS NOT NULL
                LIMIT 1
            """)
            sample = cursor.fetchone()
            if sample:
                print(f"\nSample chunk:")
                print(f"  ID: {sample[0]}")
                print(f"  Title: {sample[1]}")
                print(f"  Token count: {sample[2]}")
                print(f"  Embedding dimensions: {sample[3]}")
        
        # Additional verification: Check if embeddings are normalized
            cursor.execute("""
                SELECT id, 
                   (embedding <#> embedding) as magnitude
            FROM rib_chunks 
            WHERE embedding IS NOT NULL
            LIMIT 5
            """)
            print("\nSample embedding magnitudes (should be ~1.0 if normalized):")
            for row in cursor.fetchall():
                print(f"  {row[0]}: {row[1]:.4f}")
    
        except Exception as e:
            print(f"✗ Error during verification: {e}")
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
    
    # Path to chunks directory
    chunks_dir = 'rib_chunks'
    
    # Initialize loader
    loader = ChunkEmbeddingLoader(db_config, model_name=model_name)
    
    # Load chunks and generate embeddings
    print("Starting to load chunks and generate embeddings...\n")
    loader.load_all_chunks(chunks_dir, batch_size=32)
    
    # Verify the data
    print("\nVerifying loaded data...")
    loader.verify_data()
    
    print("\n✓ Process completed successfully!")


if __name__ == "__main__":
    main()