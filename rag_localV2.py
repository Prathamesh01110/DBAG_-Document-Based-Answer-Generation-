import os
import re
import json
import time
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

# === SETTINGS ===
DOCUMENT_PATH = "SJCEM_Document.txt"
MODEL_NAME = "all-MiniLM-L6-v2"
LLAMA_SERVER_URL = "http://localhost:8080"

class AdvancedChunker:
    """Advanced chunker without NLTK dependency"""
    
    def __init__(self):
        self.min_chunk_size = 100
        self.max_chunk_size = 300
        self.overlap_size = 50
    
    def simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting without NLTK"""
        # Split on sentence endings followed by whitespace and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def identify_sections(self, text: str) -> List[Dict]:
        """Identify document sections"""
        lines = text.split('\n')
        sections = []
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers (lines with === or ending with : or all caps)
            is_header = (
                '===' in line or 
                line.endswith(':') or 
                (line.isupper() and len(line.split()) > 1) or
                re.match(r'^[A-Z][^.!?]*(?:Details?|Information|List|Members?|Faculty|Staff|Departments?|Courses?)s?:?\s*$', line, re.IGNORECASE)
            )
            
            if is_header:
                # Save previous section
                if current_section and current_content:
                    sections.append({
                        'header': current_section,
                        'content': '\n'.join(current_content),
                        'full_text': f"{current_section}\n" + '\n'.join(current_content)
                    })
                
                # Start new section
                current_section = line.replace('===', '').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            sections.append({
                'header': current_section,
                'content': '\n'.join(current_content),
                'full_text': f"{current_section}\n" + '\n'.join(current_content)
            })
        
        return sections
    
    def create_smart_chunks(self, text: str) -> List[Dict]:
        """Create intelligent chunks"""
        sections = self.identify_sections(text)
        
        if not sections:
            # Fallback: split by paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            sections = [{'header': f'Section {i+1}', 'content': p, 'full_text': p} 
                       for i, p in enumerate(paragraphs)]
        
        chunks = []
        
        for section in sections:
            content = section['content']
            header = section['header']
            words = content.split()
            
            if len(words) <= self.max_chunk_size:
                # Section fits in one chunk
                chunks.append({
                    'text': section['full_text'],
                    'type': 'complete_section',
                    'header': header,
                    'word_count': len(words)
                })
            else:
                # Split large sections with overlap
                sentences = self.simple_sentence_split(content)
                
                current_chunk = []
                current_words = 0
                
                for sentence in sentences:
                    sentence_words = len(sentence.split())
                    
                    if current_words + sentence_words > self.max_chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = f"{header}\n" + ' '.join(current_chunk)
                        chunks.append({
                            'text': chunk_text,
                            'type': 'section_chunk',
                            'header': header,
                            'word_count': current_words
                        })
                        
                        # Start new chunk with overlap
                        if len(current_chunk) > 1:
                            overlap_sentences = current_chunk[-1:]  # Keep last sentence
                            current_chunk = overlap_sentences
                            current_words = len(' '.join(overlap_sentences).split())
                        else:
                            current_chunk = []
                            current_words = 0
                    
                    current_chunk.append(sentence)
                    current_words += sentence_words
                
                # Add final chunk
                if current_chunk:
                    chunk_text = f"{header}\n" + ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'type': 'section_chunk',
                        'header': header,
                        'word_count': current_words
                    })
        
        return chunks

class StreamingRAGSystem:
    """Complete RAG system with streaming and metrics"""
    
    def __init__(self):
        print("🔄 Loading embedding model...")
        start_time = time.time()
        self.model = SentenceTransformer(MODEL_NAME)
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        self.chunker = AdvancedChunker()
        self.chunks = []
        self.embeddings = None
        self.document_loaded = False
    
    def load_document(self, file_path: str):
        """Load and process document"""
        print(f"📄 Loading document: {file_path}")
        start_time = time.time()
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        print("🔧 Creating intelligent chunks...")
        chunk_start = time.time()
        self.chunks = self.chunker.create_smart_chunks(text)
        chunk_time = time.time() - chunk_start
        
        print(f"📊 Created {len(self.chunks)} chunks in {chunk_time:.2f}s")
        
        # Show chunk info
        for i, chunk in enumerate(self.chunks):
            header = chunk.get('header', 'No header')[:30]
            print(f"  Chunk {i+1}: {chunk['type']} - {header}... ({chunk['word_count']} words)")
        
        print("🧮 Generating embeddings...")
        embed_start = time.time()
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        self.embeddings = self.model.encode(chunk_texts, show_progress_bar=False)
        embed_time = time.time() - embed_start
        
        total_time = time.time() - start_time
        print(f"✅ Document processed in {total_time:.2f}s (chunks: {chunk_time:.2f}s, embeddings: {embed_time:.2f}s)")
        self.document_loaded = True
    
    def retrieve_context(self, question: str, top_k: int = 4) -> Tuple[str, List[Tuple[str, float]]]:
        """Retrieve relevant context with metrics"""
        if not self.document_loaded:
            return "", []
        
        start_time = time.time()
        
        # Encode question
        q_embedding = self.model.encode([question])
        
        # Calculate similarities
        similarities = cosine_similarity(q_embedding, self.embeddings)[0]
        
        # Get top chunks
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Build context
        context_parts = []
        chunk_info = []
        
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = similarities[idx]
            
            context_parts.append(f"=== {chunk.get('header', 'Section')} ===\n{chunk['text']}")
            chunk_info.append((chunk.get('header', f'Chunk {idx}'), score))
        
        context = '\n\n'.join(context_parts)
        retrieval_time = time.time() - start_time
        
        print(f"🔍 Context retrieved in {retrieval_time:.3f}s")
        return context, chunk_info
    
    def stream_response(self, question: str, context: str) -> None:
        """Stream response word by word with metrics"""
        
        prompt = f"""Document Context:
{context}

Question: {question}

Based on the document context above, provide a complete and accurate answer. Include ALL relevant information from the context.

Answer:"""

        payload = {
            "prompt": prompt,
            "n_predict": 400,
            "temperature": 0.1,
            "top_p": 0.95,
            "stream": True,
            "stop": ["Question:", "Document Context:", "Based on the above"]
        }
        
        print(f"🚀 Sending request to LLM...")
        request_start = time.time()
        
        try:
            response = requests.post(
                f"{LLAMA_SERVER_URL}/completion",
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"❌ Error: {response.status_code}")
                return
            
            # Measure time to first token
            first_token_time = None
            token_count = 0
            response_start = time.time()
            
            print(f"\n--- Answer ---")
            
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        chunk_str = chunk.decode('utf-8')
                        
                        # Handle Server-Sent Events format
                        if chunk_str.startswith('data: '):
                            data_str = chunk_str[6:]
                            
                            if data_str.strip() == '[DONE]':
                                break
                            
                            # Parse JSON
                            try:
                                data = json.loads(data_str)
                                content = data.get('content', '')
                                
                                if content:
                                    # Record first token time
                                    if first_token_time is None:
                                        first_token_time = time.time() - request_start
                                        print(f"\n⚡ First token in {first_token_time:.3f}s")
                                        print("--- Streaming Response ---")
                                    
                                    print(content, end='', flush=True)
                                    token_count += len(content.split())
                            
                            except json.JSONDecodeError:
                                # Handle plain text response
                                if data_str.strip() and first_token_time is None:
                                    first_token_time = time.time() - request_start
                                    print(f"\n⚡ First token in {first_token_time:.3f}s")
                                    print("--- Streaming Response ---")
                                
                                if data_str.strip():
                                    print(data_str, end='', flush=True)
                                    token_count += len(data_str.split())
                    
                    except Exception as e:
                        continue
            
            # Final metrics
            total_response_time = time.time() - response_start
            if token_count > 0:
                tokens_per_sec = token_count / total_response_time
                print(f"\n\n📊 Metrics:")
                print(f"   • Time to first token: {first_token_time:.3f}s")
                print(f"   • Total response time: {total_response_time:.2f}s")
                print(f"   • Tokens generated: ~{token_count}")
                print(f"   • Speed: {tokens_per_sec:.1f} tokens/sec")
        
        except Exception as e:
            print(f"❌ Streaming failed: {e}")
            # Fallback to non-streaming
            self.fallback_response(payload)
    
    def fallback_response(self, payload: dict):
        """Fallback non-streaming response"""
        print("🔄 Falling back to non-streaming...")
        
        payload['stream'] = False
        try:
            response = requests.post(f"{LLAMA_SERVER_URL}/completion", json=payload)
            if response.status_code == 200:
                result = response.json()
                content = result.get('content', '')
                print(f"\n--- Answer ---\n{content}")
        except Exception as e:
            print(f"❌ Fallback also failed: {e}")
    
    def answer_question(self, question: str, show_debug: bool = False):
        """Complete pipeline with metrics"""
        if not self.document_loaded:
            print("❌ No document loaded!")
            return
        
        print(f"\n" + "="*60)
        print(f"🤔 Question: {question}")
        print("="*60)
        
        # Retrieve context
        context, chunk_info = self.retrieve_context(question)
        
        if show_debug:
            print(f"\n🔍 Retrieved chunks:")
            for i, (header, score) in enumerate(chunk_info, 1):
                print(f"   {i}. {header} (score: {score:.3f})")
        
        # Stream response
        self.stream_response(question, context)

def main():
    """Main function"""
    
    if not os.path.exists(DOCUMENT_PATH):
        print(f"❌ Error: {DOCUMENT_PATH} not found!")
        return
    
    # Initialize system
    print("🚀 Starting Advanced RAG System with Streaming")
    print("="*60)
    
    rag = StreamingRAGSystem()
    rag.load_document(DOCUMENT_PATH)
    
    print(f"\n✅ System ready!")
    print("Commands:")
    print("  • Type your question to get streamed answer")
    print("  • Type 'debug' to toggle debug mode") 
    print("  • Type 'stats' to show system stats")
    print("  • Type 'quit' to exit")
    print("="*60)
    
    debug_mode = False
    
    while True:
        try:
            question = input(f"\n{'[DEBUG] ' if debug_mode else ''}Question: ").strip()
            
            if question.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif question.lower() == 'debug':
                debug_mode = not debug_mode
                print(f"🐛 Debug mode: {'ON' if debug_mode else 'OFF'}")
                continue
            elif question.lower() == 'stats':
                print(f"📊 System Statistics:")
                print(f"   • Chunks: {len(rag.chunks)}")
                print(f"   • Embedding dimensions: {rag.embeddings.shape if rag.embeddings is not None else 'N/A'}")
                print(f"   • Model: {MODEL_NAME}")
                continue
            elif not question:
                continue
            
            # Answer question with timing
            total_start = time.time()
            rag.answer_question(question, show_debug=debug_mode)
            total_time = time.time() - total_start
            print(f"\n⏱️  Total time: {total_time:.2f}s")
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()