# fixed_chunking_v3.py

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import pdfplumber
import json

@dataclass
class ObservationChunk:
    """Represents a single observation chunk with metadata"""
    section_number: str
    title: str
    observation: str
    recommendation: str
    regulation: str
    full_text: str
    category: str

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
    
    def identify_category(self, section_number: str) -> str:
        """Identify category based on section number"""
        main_section = section_number.split('.')[0] + '.0'
        return self.categories.get(main_section, "General")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from PDF file using pdfplumber"""
        all_text = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    all_text.append(text)
                else:
                    print(f"⚠️ No text found on page {page_num}")
        
        return "\n\n".join(all_text)
    
    def extract_section_content(self, text: str, start_marker: str, end_marker: Optional[str] = None) -> str:
        """
        Extract text between two markers with flexible matching.
        
        Args:
            text: Full text to search
            start_marker: Starting marker (e.g., "Observation")
            end_marker: Ending marker (e.g., "Recommendation"), None for end of text
            
        Returns:
            Extracted text between markers
        """
        # Create flexible pattern for start marker
        start_pattern = start_marker.replace(' ', r'\s*')
        
        if end_marker:
            # Create flexible pattern for end marker
            end_pattern = end_marker.replace(' ', r'\s*')
            # Match content between markers
            pattern = rf'{start_pattern}\s*\n(.*?)(?:\n\s*{end_pattern}|\Z)'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        else:
            # Match until end of text
            pattern = rf'{start_pattern}\s*\n(.*?)$'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def extract_title_before_observation(self, text: str, section_number: str) -> str:
        """
        Extract the complete title (bold/uppercase text) between section number and 'Observation'.
        This handles multi-line titles.
        
        Args:
            text: The section text
            section_number: The section number (e.g., "5.13")
            
        Returns:
            The complete title
        """
        # Check if this section even has an "Observation" keyword
        if not re.search(r'\bObservation\b', text, re.IGNORECASE):
            return None
        
        # Pattern to find text between section number and "Observation"
        # Look for text that comes after the section number and before "Observation"
        pattern = rf'{re.escape(section_number)}\s+(.*?)(?=\s*\bObservation\b)'
        
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            title_text = match.group(1).strip()
            
            # Clean up the title - remove excessive whitespace and newlines
            # Join multiple lines with space
            title_lines = [line.strip() for line in title_text.split('\n') if line.strip()]
            
            # Filter to keep only lines that are likely part of the title
            # (uppercase letters, numbers, slashes, hyphens, parentheses, commas, ampersands)
            title_parts = []
            for line in title_lines:
                # Check if line looks like a title (mostly uppercase and special chars)
                # Allow some flexibility for titles
                if re.match(r'^[A-Z0-9\s/\-(),&]+$', line):
                    title_parts.append(line)
                elif len(line) > 0 and line[0].isupper():
                    # If it starts with uppercase, it might be part of title
                    # Check if it has mostly uppercase or special formatting
                    upper_count = sum(1 for c in line if c.isupper())
                    if upper_count / len(line) > 0.5:  # More than 50% uppercase
                        title_parts.append(line)
                    else:
                        break
                else:
                    # Stop if we hit a line that's not title-like
                    break
            
            if title_parts:
                # Join with space and clean up extra spaces
                complete_title = ' '.join(title_parts)
                complete_title = re.sub(r'\s+', ' ', complete_title).strip()
                return complete_title
        
        return None
    
    def split_into_sections(self, text: str) -> List[Dict[str, any]]:
        """Split document into sections with improved title extraction"""
        sections = []
        
        # Pattern to match section numbers like "1.1", "2.3", "5.13", etc.
        # Just find the section number, we'll extract title differently
        section_pattern = r'\n(\d+\.\d+)\s+'
        
        matches = list(re.finditer(section_pattern, text))
        
        print(f"Found {len(matches)} potential section matches")
        
        processed_count = 0
        skipped_count = 0
        
        for i, match in enumerate(matches):
            section_number = match.group(1).strip()
            
            # Get content from section number to next section (or end)
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_full_text = text[start_pos:end_pos].strip()
            
            # Check if this section has an "Observation" - if not, skip it
            if not re.search(r'\bObservation\b', section_full_text, re.IGNORECASE):
                skipped_count += 1
                continue
            
            # Extract title from the text between section number and "Observation"
            section_title = self.extract_title_before_observation(section_full_text, section_number)
            
            if not section_title:
                print(f"  ⚠️ Could not extract title for section {section_number}, skipping")
                skipped_count += 1
                continue
            
            sections.append({
                'section_number': section_number,
                'section_title': section_title,
                'content': section_full_text
            })
            processed_count += 1
        
        print(f"Processed {processed_count} valid sections, skipped {skipped_count} sections")
        
        return sections
    
    def parse_section_components(self, content: str) -> Dict[str, str]:
        """Parse a section into Observation, Recommendation, and Regulation components"""
        components = {
            'observation': '',
            'recommendation': '',
            'regulation': ''
        }
        
        # Extract Observation
        observation_text = self.extract_section_content(content, "Observation", "Recommendation")
        if observation_text:
            components['observation'] = observation_text
        
        # Extract Recommendation
        recommendation_text = self.extract_section_content(content, "Recommendation", "Regulation")
        if recommendation_text:
            components['recommendation'] = recommendation_text
        
        # Extract Regulation/Guideline - try multiple variations
        regulation_text = (
            self.extract_section_content(content, "Regulation / Guideline", None) or
            self.extract_section_content(content, "Regulation/Guideline", None) or
            self.extract_section_content(content, "Regulation", None) or
            self.extract_section_content(content, "Guideline", None)
        )
        if regulation_text:
            components['regulation'] = regulation_text
        
        return components
    
    def create_chunk(self, section: Dict[str, any]) -> ObservationChunk:
        """Create an ObservationChunk from a section"""
        section_number = section['section_number']
        section_title = section['section_title']
        content = section['content']
        
        # Parse components
        components = self.parse_section_components(content)
        
        # Build full text
        full_text = f"Section {section_number}: {section_title}\n\n"
        
        if components['observation']:
            full_text += f"OBSERVATION:\n{components['observation']}\n\n"
        
        if components['recommendation']:
            full_text += f"RECOMMENDATION:\n{components['recommendation']}\n\n"
        
        if components['regulation']:
            full_text += f"REGULATION/GUIDELINE:\n{components['regulation']}\n\n"
        
        # Create chunk
        chunk = ObservationChunk(
            section_number=section_number,
            title=section_title,
            observation=components['observation'],
            recommendation=components['recommendation'],
            regulation=components['regulation'],
            full_text=full_text.strip(),
            category=self.identify_category(section_number)
        )
        
        return chunk
    
    def create_summary_chunk(self, text: str) -> ObservationChunk:
        """Create a summary chunk with scope and objectives"""
        # Extract scope and objective sections
        scope_match = re.search(r'Scope\s*(.*?)(?=Objective|$)', text, re.DOTALL | re.IGNORECASE)
        obj_match = re.search(r'Objective\s*(.*?)(?=List of RIB|$)', text, re.DOTALL | re.IGNORECASE)
        
        scope = scope_match.group(1).strip() if scope_match else ""
        objective = obj_match.group(1).strip() if obj_match else ""
        
        full_text = f"""RISK IMPROVEMENT BENCHMARK (RIB) - DOCUMENT SUMMARY

SCOPE:
{scope}

OBJECTIVE:
{objective}"""
        
        return ObservationChunk(
            section_number="0.0",
            title="Document Summary",
            observation="",
            recommendation="",
            regulation="",
            full_text=full_text,
            category="Summary"
        )
    
    def create_category_index_chunks(self) -> List[ObservationChunk]:
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
                full_text = f"""CATEGORY INDEX: {category}

Description:
{category_descriptions[category]}

This category contains multiple risk improvement recommendations related to {category.lower()} hazards and controls."""
                
                chunk = ObservationChunk(
                    section_number=section_num,
                    title=f"{category} Index",
                    observation="",
                    recommendation="",
                    regulation="",
                    full_text=full_text,
                    category=category
                )
                
                index_chunks.append(chunk)
        
        return index_chunks
    
    def chunk_document(self, pdf_path: str) -> List[ObservationChunk]:
        """Main method to chunk the entire document"""
        print("📄 Extracting text from PDF...")
        text = self.extract_text_from_pdf(pdf_path)
        print(f"✅ Extracted {len(text)} characters\n")
        
        print("Creating summary chunk...")
        chunks = [self.create_summary_chunk(text)]
        
        print("Creating category index chunks...")
        index_chunks = self.create_category_index_chunks()
        chunks.extend(index_chunks)
        print(f"✓ Created {len(index_chunks)} category index chunks\n")
        
        print("Splitting document into sections...")
        sections = self.split_into_sections(text)
        
        print(f"\nCreating chunks from {len(sections)} valid sections...")
        for section in sections:
            try:
                chunk = self.create_chunk(section)
                chunks.append(chunk)
                print(f"  ✓ {chunk.section_number} - {chunk.title[:60]}{'...' if len(chunk.title) > 60 else ''}")
            except Exception as e:
                print(f"  ✗ Error processing section {section.get('section_number', 'unknown')}: {e}")
        
        return chunks
    
    def save_chunks_to_json(self, chunks: List[ObservationChunk], output_file: str = "rib_chunks_fixed.json"):
        """Save chunks to JSON file"""
        chunk_dicts = []
        
        for chunk in chunks:
            chunk_dict = {
                'id': f"{chunk.category}_{chunk.section_number}",
                'section_number': chunk.section_number,
                'title': chunk.title,
                'category': chunk.category,
                'observation': chunk.observation,
                'recommendation': chunk.recommendation,
                'regulation': chunk.regulation,
                'full_text': chunk.full_text,
                'metadata': {
                    'section': chunk.section_number,
                    'title': chunk.title,
                    'category': chunk.category,
                }
            }
            chunk_dicts.append(chunk_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved {len(chunks)} chunks to '{output_file}'")
    
    def print_statistics(self, chunks: List[ObservationChunk]):
        """Print statistics about the chunks"""
        print("\n" + "="*80)
        print("CHUNKING STATISTICS")
        print("="*80)
        
        total_chunks = len(chunks)
        chunks_with_obs = sum(1 for c in chunks if c.observation)
        chunks_with_rec = sum(1 for c in chunks if c.recommendation)
        chunks_with_reg = sum(1 for c in chunks if c.regulation)
        
        print(f"Total chunks: {total_chunks}")
        print(f"Chunks with observation: {chunks_with_obs}")
        print(f"Chunks with recommendation: {chunks_with_rec}")
        print(f"Chunks with regulation: {chunks_with_reg}")
        
        # Category breakdown
        print("\nChunks by category:")
        category_counts = {}
        for chunk in chunks:
            category_counts[chunk.category] = category_counts.get(chunk.category, 0) + 1
        
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")
        
        # Check for missing sections in main categories
        print("\nSection coverage check:")
        all_sections = [c.section_number for c in chunks if c.section_number != "0.0" and not c.section_number.endswith(".0")]
        
        # Check each category
        for main_num in range(1, 6):  # 1.0 to 5.0
            category_sections = [s for s in all_sections if s.startswith(f"{main_num}.")]
            if category_sections:
                print(f"  Category {main_num}.0: {len(category_sections)} sections")
                # Sort and show range
                sorted_sections = sorted(category_sections, key=lambda x: float(x))
                print(f"    Range: {sorted_sections[0]} to {sorted_sections[-1]}")
        
        print("="*80 + "\n")


def main():
    """Main execution function"""
    
    # Initialize chunker
    chunker = RIBDocumentChunker()
    
    # Path to your PDF file
    pdf_path = "/home/amir/Desktop/MRE TSD/5. Risk Improvement Benchmark 1.pdf"
    
    # Chunk the document
    print("Starting document chunking process...\n")
    chunks = chunker.chunk_document(pdf_path)
    
    # Print statistics
    chunker.print_statistics(chunks)
    
    # Save to JSON
    chunker.save_chunks_to_json(chunks, "rib_chunks_fixed_v3.json")
    
    print("\n✓ Document chunking completed successfully!")
    
    # Show first few chunks as examples
    print("\n" + "="*80)
    print("SAMPLE CHUNKS")
    print("="*80)
    
    # Skip summary and category chunks, show actual content chunks
    content_chunks = [c for c in chunks if c.observation or c.recommendation]
    
    for i, chunk in enumerate(content_chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(f"ID: {chunk.category}_{chunk.section_number}")
        print(f"Section: {chunk.section_number} - {chunk.title}")
        print(f"Category: {chunk.category}")
        print(f"Observation length: {len(chunk.observation)} chars")
        print(f"Recommendation length: {len(chunk.recommendation)} chars")
        print(f"Regulation length: {len(chunk.regulation)} chars")
        
        if chunk.title:
            print(f"\nTitle:")
            print(chunk.title)


if __name__ == "__main__":
    main()