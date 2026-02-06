import re
import json
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from dataclasses import dataclass, asdict

@dataclass
class ChunkMetadata:
    section_number: str
    section_title: str
    category: str
    regulations: List[str]
    keywords: List[str]
    risk_type: str = ""

@dataclass
class DocumentChunk:
    id: str
    category: str
    title: str
    content: str
    metadata: ChunkMetadata
    token_count: int = 0

class RIBDocumentChunker:
    def __init__(self):
        self.categories = {
            "1.0": "Perils",
            "2.0": "Electrical",
            "3.0": "Housekeeping",
            "4.0": "Human Element",
            "5.0": "Process",
            "6.0": "References",
            "7.0": "Regulations"
        }
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def identify_category(self, section_number: str) -> str:
        """Identify category based on section number"""
        main_section = section_number.split('.')[0] + '.0'
        return self.categories.get(main_section, "General")
    
    def extract_regulations(self, regulation_text: str) -> List[str]:
        """Extract regulation names from regulation section"""
        regulations = []
        
        # Common regulation patterns
        patterns = [
            # r'Act \d+[:\s]+([^\n]+)',
            r'^Regulation / Guideline\s*\d+[:\s]*\d*[:\s]*\d*\s*[-–]\s*([^\n]+)',
            r'MS\s*\d+[:\s]*\d*[:\s]*\d*\s*[-–]\s*([^\n]+)',
            r'NFPA\s*\d+\s*[-–]\s*([^\n]+)',
            r'API\s*\w+\s*[-–]\s*([^\n]+)',
            r'IEC\s*\d+[-\d]*\s*[-–]\s*([^\n]+)',
            r'IEEE\s*\d+\.?\d*\s*[-–]\s*([^\n]+)',
            r'OSHA\s*[\d\s\.CFR]+\s*[-–]?\s*([^\n]+)',
            r'Section\s+\d+[:\s]+([^\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, regulation_text, re.IGNORECASE)
            for match in matches:
                reg_name = match.group(0).strip()
                if len(reg_name) > 10:  # Filter out very short matches
                    regulations.append(reg_name)
        
        return list(set(regulations))[:5]  # Limit to top 5 unique regulations
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Common important terms in risk management
        keyword_patterns = [
            r'\b(fire|explosion|hazard|risk|safety|emergency)\b',
            r'\b(inspection|maintenance|testing|monitoring)\b',
            r'\b(electrical|mechanical|structural|chemical)\b',
            r'\b(corrosion|leakage|damage|failure)\b',
            r'\b(compliance|regulation|standard|code)\b',
        ]
        
        keywords = set()
        text_lower = text.lower()
        
        for pattern in keyword_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                keywords.add(match.group(0).lower())
        
        return list(keywords)[:10]  # Limit to 10 keywords
    
    def split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """Split document into main sections"""
        sections = []
        
        # Pattern to match section numbers like "1.1", "2.3", "5.12", etc.
        section_pattern = r'\n(\d+\.\d+)\s+([A-Z][A-Z\s,/\-()]+)\n'
        
        matches = list(re.finditer(section_pattern, text))
        
        for i, match in enumerate(matches):
            section_number = match.group(1)
            section_title = match.group(2).strip()
            
            # Get content between this section and the next
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_content = text[start_pos:end_pos].strip()
            
            sections.append({
                'section_number': section_number,
                'section_title': section_title,
                'content': section_content
            })
        
        return sections
    
    def parse_section_components(self, content: str) -> Dict[str, str]:
        """Parse a section into Observation, Recommendation, and Regulation components"""
        components = {
            'observation': '',
            'recommendation': '',
            'regulation': ''
        }
        
        # Extract Observation
        obs_match = re.search(r'Observation\s*(.*?)(?=Recommendation|$)', content, re.DOTALL | re.IGNORECASE)
        if obs_match:
            components['observation'] = obs_match.group(1).strip()
        
        # Extract Recommendation
        rec_match = re.search(r'Recommendation\s*(.*?)(?=Regulation|$)', content, re.DOTALL | re.IGNORECASE)
        if rec_match:
            components['recommendation'] = rec_match.group(1).strip()
        
        # Extract Regulation/Guideline
        reg_match = re.search(r'Regulation\s*/\s*Guideline\s*(.*?)$', content, re.DOTALL | re.IGNORECASE)
        if reg_match:
            components['regulation'] = reg_match.group(1).strip()
        
        return components
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def create_chunk(self, section: Dict[str, Any]) -> DocumentChunk:
        """Create a DocumentChunk from a section"""
        section_number = section['section_number']
        section_title = section['section_title']
        content = section['content']
        
        # Parse components
        components = self.parse_section_components(content)
        
        # Build formatted content
        formatted_content = f"""Section {section_number}: {section_title}

OBSERVATION:
{components['observation']}

RECOMMENDATION:
{components['recommendation']}

REGULATION/GUIDELINE:
{components['regulation']}"""
        
        # Extract regulations and keywords
        regulations = self.extract_regulations(components['regulation'])
        keywords = self.extract_keywords(content)
        
        # Determine risk type
        risk_type = self.determine_risk_type(section_title, content)
        
        # Create metadata
        metadata = ChunkMetadata(
            section_number=section_number,
            section_title=section_title,
            category=self.identify_category(section_number),
            regulations=regulations,
            keywords=keywords,
            risk_type=risk_type
        )
        
        # Create chunk
        chunk = DocumentChunk(
            id=section_number,
            category=metadata.category,
            title=section_title,
            content=formatted_content,
            metadata=metadata,
            token_count=self.estimate_tokens(formatted_content)
        )
        
        return chunk
    
    def determine_risk_type(self, title: str, content: str) -> str:
        """Determine the type of risk based on title and content"""
        title_lower = title.lower()
        content_lower = content.lower()
        
        risk_types = {
            'fire': ['fire', 'combustible', 'flammable', 'ignition', 'explosion'],
            'electrical': ['electrical', 'electric', 'circuit', 'wiring', 'transformer'],
            'structural': ['structural', 'subsidence', 'foundation', 'building', 'collapse'],
            'environmental': ['environmental', 'pollution', 'drainage', 'flood', 'erosion'],
            'mechanical': ['mechanical', 'machinery', 'equipment', 'crane', 'lifting'],
            'chemical': ['chemical', 'hazardous', 'toxic', 'corrosive'],
            'safety': ['safety', 'injury', 'accident', 'evacuation', 'emergency']
        }
        
        combined_text = title_lower + ' ' + content_lower
        
        for risk_type, keywords in risk_types.items():
            if any(keyword in combined_text for keyword in keywords):
                return risk_type
        
        return 'general'
    
    def create_summary_chunk(self, text: str) -> DocumentChunk:
        """Create a summary chunk with scope and objectives"""
        # Extract scope and objective sections
        scope_match = re.search(r'Scope\s*(.*?)(?=Objective|$)', text, re.DOTALL | re.IGNORECASE)
        obj_match = re.search(r'Objective\s*(.*?)(?=List of RIB|$)', text, re.DOTALL | re.IGNORECASE)
        
        scope = scope_match.group(1).strip() if scope_match else ""
        objective = obj_match.group(1).strip() if obj_match else ""
        
        content = f"""RISK IMPROVEMENT BENCHMARK (RIB) - DOCUMENT SUMMARY

SCOPE:
{scope}

OBJECTIVE:
{objective}"""
        
        metadata = ChunkMetadata(
            section_number="0.0",
            section_title="Document Summary",
            category="Summary",
            regulations=[],
            keywords=['scope', 'objective', 'risk improvement', 'benchmark'],
            risk_type='overview'
        )
        
        return DocumentChunk(
            id="0.0",
            category="Summary",
            title="Document Summary",
            content=content,
            metadata=metadata,
            token_count=self.estimate_tokens(content)
        )
    
    def create_category_index_chunks(self) -> List[DocumentChunk]:
        """Create index chunks for main categories"""
        index_chunks = []
        
        category_descriptions = {
            "Perils": "Natural and environmental hazards including subsidence, flooding, lightning, and windstorms",
            "Electrical": "Electrical safety issues including inspections, wiring, circuit breakers, and transformers",
            "Housekeeping": "Workplace organization and storage safety including material handling and containment",
            "Human Element": "Human-related safety factors including evacuation, smoking policies, and hot work permits",
            "Process": "Process-specific safety measures including equipment maintenance and operational procedures"
        }
        
        for section_num, category in self.categories.items():
            if category in category_descriptions:
                content = f"""CATEGORY INDEX: {category}

Description:
{category_descriptions[category]}

This category contains multiple risk improvement recommendations related to {category.lower()} hazards and controls."""
                
                metadata = ChunkMetadata(
                    section_number=section_num,
                    section_title=f"{category} Index",
                    category=category,
                    regulations=[],
                    keywords=[category.lower(), 'index', 'category'],
                    risk_type='index'
                )
                
                chunk = DocumentChunk(
                    id=f"{section_num}_index",
                    category=category,
                    title=f"{category} Index",
                    content=content,
                    metadata=metadata,
                    token_count=self.estimate_tokens(content)
                )
                
                index_chunks.append(chunk)
        
        return index_chunks
    
    def chunk_document(self, pdf_path: str) -> List[DocumentChunk]:
        """Main method to chunk the entire document"""
        print("Extracting text from PDF...")
        text = self.extract_text_from_pdf(pdf_path)
        
        print("Creating summary chunk...")
        chunks = [self.create_summary_chunk(text)]
        
        print("Creating category index chunks...")
        chunks.extend(self.create_category_index_chunks())
        
        print("Splitting document into sections...")
        sections = self.split_into_sections(text)
        
        print(f"Found {len(sections)} sections. Creating chunks...")
        for section in sections:
            try:
                chunk = self.create_chunk(section)
                chunks.append(chunk)
                print(f"  ✓ Created chunk: {chunk.id} - {chunk.title} ({chunk.token_count} tokens)")
            except Exception as e:
                print(f"  ✗ Error processing section {section.get('section_number', 'unknown')}: {e}")
        
        return chunks
    
    def save_chunks(self, chunks: List[DocumentChunk], output_dir: str = "chunks"):
        """Save chunks to JSON files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save individual chunks
        for chunk in chunks:
            chunk_dict = {
                'id': chunk.id,
                'category': chunk.category,
                'title': chunk.title,
                'content': chunk.content,
                'token_count': chunk.token_count,
                'metadata': asdict(chunk.metadata)
            }
            
            filename = f"{chunk.id.replace('.', '_')}.json"
            with open(output_path / filename, 'w', encoding='utf-8') as f:
                json.dump(chunk_dict, f, indent=2, ensure_ascii=False)
        
        # Save all chunks in one file
        all_chunks = [{
            'id': chunk.id,
            'category': chunk.category,
            'title': chunk.title,
            'content': chunk.content,
            'token_count': chunk.token_count,
            'metadata': asdict(chunk.metadata)
        } for chunk in chunks]
        
        with open(output_path / 'all_chunks.json', 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved {len(chunks)} chunks to '{output_dir}' directory")
        
        # Print statistics
        self.print_statistics(chunks)
    
    def print_statistics(self, chunks: List[DocumentChunk]):
        """Print statistics about the chunks"""
        print("\n" + "="*60)
        print("CHUNKING STATISTICS")
        print("="*60)
        
        total_chunks = len(chunks)
        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_tokens = total_tokens / total_chunks if total_chunks > 0 else 0
        
        print(f"Total chunks: {total_chunks}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Average tokens per chunk: {avg_tokens:.0f}")
        
        # Category breakdown
        print("\nChunks by category:")
        category_counts = {}
        for chunk in chunks:
            category_counts[chunk.category] = category_counts.get(chunk.category, 0) + 1
        
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")
        
        # Token distribution
        print("\nToken distribution:")
        token_ranges = {
            '0-500': 0,
            '501-1000': 0,
            '1001-1500': 0,
            '1501+': 0
        }
        
        for chunk in chunks:
            if chunk.token_count <= 500:
                token_ranges['0-500'] += 1
            elif chunk.token_count <= 1000:
                token_ranges['501-1000'] += 1
            elif chunk.token_count <= 1500:
                token_ranges['1001-1500'] += 1
            else:
                token_ranges['1501+'] += 1
        
        for range_name, count in token_ranges.items():
            print(f"  {range_name} tokens: {count}")
        
        print("="*60 + "\n")


def main():
    """Main execution function"""
    # Initialize chunker
    chunker = RIBDocumentChunker()
    
    # Path to your PDF file
    pdf_path = "/home/amir/Desktop/MRE TSD/5. Risk Improvement Benchmark 1.pdf"  # replace with your PDF path
    
    # Chunk the document
    print("Starting document chunking process...\n")
    chunks = chunker.chunk_document(pdf_path)
    
    # Save chunks
    chunker.save_chunks(chunks, output_dir="rib_chunks")
    
    print("\n✓ Document chunking completed successfully!")
    
    # Optional: Display first chunk as example
    if chunks:
        print("\nExample chunk (first one):")
        print("-" * 60)
        print(f"ID: {chunks[0].id}")
        print(f"Category: {chunks[0].category}")
        print(f"Title: {chunks[0].title}")
        print(f"Token Count: {chunks[0].token_count}")
        print(f"Regulations: {chunks[0].metadata.regulations}")
        print(f"Keywords: {chunks[0].metadata.keywords}")
        print(f"\nContent preview (first 500 chars):")
        print(chunks[0].content[:500] + "...")


if __name__ == "__main__":
    main()