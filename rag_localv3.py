#!/usr/bin/env python3
"""
Offline AI Assistant - Speech-to-Speech Pipeline with Piper TTS Only
Optimized for AMD R5 5600G CPU-only system with 16GB RAM
"""

import os
import sys
import time
import threading
import queue
import json
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import logging
import signal
import wave
from typing import Optional, Generator

# Add project paths
PROJECT_DIR = Path(__file__).parent
sys.path.append(str(PROJECT_DIR / "fastwhisper"))

# Import components
from faster_whisper import WhisperModel
import torch

# Import Piper TTS
try:
    from piper import PiperVoice, SynthesisConfig
    PIPER_AVAILABLE = True
except ImportError:
    print("⚠️  Piper TTS not found. Install with: pip install piper-tts")
    PIPER_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OfflineAIAssistant:
    def __init__(self):
        self.running = False
        self.is_speaking = False
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.tts_stream = None  # Track TTS audio stream
        self.tts_thread = None  # Track TTS thread
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_duration = 1.0  # seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Voice activity detection threshold
        self.silence_threshold = 0.001
        self.min_speech_duration = 0.5
        
        # Debug tracking
        self.debug_counter = 0
        self.last_silence_debug = time.time()
        
        # Model paths
        self.models_dir = PROJECT_DIR / "models"
        self.whisper_model_size = "base"
        
        # Piper TTS settings (ONLY TTS - no OpenVoice)
        self.piper_voice_path = PROJECT_DIR / "models" / "piper" / "en_US-lessac-medium.onnx"
        self.piper_voice = None
        
        # Initialize components
        self.whisper_model = None
        
        # Conversation context
        self.conversation_history = []
        self.max_history = 10
        
        # llama.cpp server settings
        self.llm_url = "http://127.0.0.1:8080"
        
        logger.info("Initializing Offline AI Assistant with Piper TTS ONLY...")
        
    def initialize_models(self):
        """Initialize Whisper and Piper TTS models"""
        try:
            # Initialize Whisper STT
            logger.info("Loading Whisper model...")
            self.whisper_model = WhisperModel(
                self.whisper_model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=4
            )
            logger.info("✅ Whisper model loaded successfully")
            
            # Initialize ONLY Piper TTS
            logger.info("Loading Piper TTS model...")
            self.initialize_piper_tts()
            
            # Test llama.cpp connection
            self.test_llm_connection()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def initialize_piper_tts(self):
        """Initialize ONLY Piper TTS (no OpenVoice)"""
        try:
            if not PIPER_AVAILABLE:
                raise ImportError("Piper TTS not available - install with: pip install piper-tts")
            
            # Check if Piper voice model exists
            if not self.piper_voice_path.exists():
                logger.error(f"❌ Piper voice not found at: {self.piper_voice_path}")
                logger.info("Download with:")
                logger.info("mkdir -p models/piper")
                logger.info("wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx -O models/piper/en_US-lessac-medium.onnx")
                raise FileNotFoundError("Piper voice model not found")
            
            # Load Piper voice
            logger.info(f"🎤 Loading Piper voice: {self.piper_voice_path.name}")
            self.piper_voice = PiperVoice.load(str(self.piper_voice_path))
            
            # Configure for fast, real-time synthesis
            self.synthesis_config = SynthesisConfig(
                volume=0.9,
                length_scale=0.9,  # Slightly faster speech
                noise_scale=0.3,   # Less noise for faster processing
                noise_w_scale=0.3,
                normalize_audio=True,
            )
            
            logger.info("✅ Piper TTS initialized successfully (FAST MODE)")
            
        except Exception as e:
            logger.error(f"❌ Piper TTS failed: {e}")
            logger.info("Falling back to espeak...")
            self.piper_voice = None
    
    def test_llm_connection(self):
        """Test connection to llama.cpp server"""
        try:
            response = requests.get(f"{self.llm_url}/health", timeout=5)
            logger.info("✅ llama.cpp server connected")
        except Exception as e:
            logger.error(f"❌ Cannot connect to llama.cpp server at {self.llm_url}")
            logger.error("Start server: ./start_llama_server.sh")
            raise ConnectionError("llama.cpp server not available")
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if status:
            logger.warning(f"Audio input status: {status}")
        
        # Convert to mono if stereo
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata.flatten()
        
        # Check for voice activity
        energy = np.mean(np.square(audio_data))
        
        # Debug counter
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 0
            
        # Print audio level every 50 callbacks
        if self.debug_counter % 50 == 0:
            print(f"🎤 Audio: {energy:.6f} (threshold: {self.silence_threshold})")
        
        # Add audio to queue
        self.audio_queue.put(audio_data.copy())
        
        # Voice detection
        if energy > self.silence_threshold:
            if self.is_speaking:
                # Barge-in: Stop TTS
                logger.info("🛑 Barge-in detected!")
                self.stop_tts_playback()
            
            if self.debug_counter % 10 == 0:
                print(f"🗣️  Voice: {energy:.6f}")
    
    def stop_tts_playback(self):
        """Stop TTS playback immediately"""
        self.is_speaking = False
        
        # Stop audio stream
        if self.tts_stream:
            try:
                self.tts_stream.close()
            except:
                pass
            self.tts_stream = None
        
        # Stop sounddevice
        try:
            sd.stop()
        except:
            pass
    
    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Fast transcription with Whisper"""
        try:
            print("🎯 Transcribing...")
            
            # Save temporary audio
            temp_audio = "/tmp/temp_audio.wav"
            sf.write(temp_audio, audio_data, self.sample_rate)
            
            # Fast transcription
            segments, info = self.whisper_model.transcribe(
                temp_audio,
                language="en",
                task="transcribe",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300)  # Faster VAD
            )
            
            # Combine segments
            text = " ".join([segment.text.strip() for segment in segments])
            
            # Cleanup
            os.remove(temp_audio)
            
            if text.strip():
                print(f"📝 USER: '{text.strip()}'")
                return text.strip()
            else:
                print("❌ No speech detected")
                return None
            
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            return None
    
    def generate_response(self, user_input: str) -> str:
        """Fast response generation"""
        try:
            print(f"🧠 Generating response...")
            
            # Build context
            context = self.build_conversation_context(user_input)
            
            # Fast LLM request
            payload = {
                "prompt": context,
                "max_tokens": 100,  # Shorter for faster response
                "temperature": 0.7,
                "top_p": 0.9,
                "stop": ["User:", "\n\n"],
                "stream": False
            }
            
            response = requests.post(
                f"{self.llm_url}/completion",
                json=payload,
                timeout=15  # Faster timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get("content", "").strip()
                
                print(f"🤖 AI: '{ai_response}'")
                
                # Update history
                self.conversation_history.append(("User", user_input))
                self.conversation_history.append(("Assistant", ai_response))
                
                # Keep history short
                if len(self.conversation_history) > self.max_history * 2:
                    self.conversation_history = self.conversation_history[-self.max_history * 2:]
                
                return ai_response
            else:
                return "I'm having trouble right now."
                
        except Exception as e:
            logger.error(f"❌ Response error: {e}")
            return "Sorry, I couldn't process that."
    
    def build_conversation_context(self, user_input: str) -> str:
        """Build fast conversation context"""
        system_prompt = "You are a helpful AI assistant. Keep responses short and conversational, 1-2 sentences max unless more detail is requested."
        
        context = f"System: {system_prompt}\n\n"
        
        # Add recent history only
        for role, message in self.conversation_history[-4:]:  # Last 2 exchanges
            context += f"{role}: {message}\n"
        
        context += f"User: {user_input}\nAssistant:"
        
        return context
    
    def text_to_speech_fast(self, text: str):
        """Ultra-fast TTS with Piper streaming"""
        try:
            print(f"🔊 TTS: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            self.is_speaking = True
            
            if self.piper_voice:
                # Use Piper in a separate thread for non-blocking
                self.tts_thread = threading.Thread(
                    target=self.piper_stream_audio, 
                    args=(text,), 
                    daemon=True
                )
                self.tts_thread.start()
            else:
                # Fast espeak fallback
                threading.Thread(
                    target=lambda: os.system(f'espeak -s 180 -v en-us "{text}" 2>/dev/null'),
                    daemon=True
                ).start()
            
        except Exception as e:
            logger.error(f"❌ TTS error: {e}")
            self.is_speaking = False
    
    def bytes_to_numpy_int16(self, b: bytes, channels: int):
        """Convert bytes to numpy array"""
        arr = np.frombuffer(b, dtype=np.int16)
        if channels > 1:
            arr = arr.reshape(-1, channels)
        return arr
    
    def piper_stream_audio(self, text: str):
        """Stream Piper audio in real-time"""
        try:
            print("🎵 Streaming Piper TTS...")
            
            # Generate audio stream
            gen = self.piper_voice.synthesize(text, syn_config=self.synthesis_config)
            
            try:
                first_chunk = next(gen)
            except StopIteration:
                print("❌ No audio generated")
                self.is_speaking = False
                return
            
            # Get audio format
            sample_rate = first_chunk.sample_rate
            channels = first_chunk.sample_channels
            
            # Stream audio with minimal latency
            with sd.OutputStream(
                samplerate=sample_rate, 
                channels=channels, 
                dtype='int16', 
                blocksize=0,  # Minimal latency
                latency='low'  # Low latency mode
            ) as stream:
                
                self.tts_stream = stream
                
                try:
                    # Play first chunk immediately
                    arr = self.bytes_to_numpy_int16(first_chunk.audio_int16_bytes, channels)
                    stream.write(arr)
                    
                    # Stream remaining chunks
                    for chunk in gen:
                        if not self.is_speaking:  # Check for interruption
                            break
                            
                        arr = self.bytes_to_numpy_int16(chunk.audio_int16_bytes, channels)
                        stream.write(arr)
                    
                    print("✅ TTS complete")
                    
                except Exception as e:
                    print(f"❌ Streaming error: {e}")
                    
            self.tts_stream = None
            self.is_speaking = False
            
        except Exception as e:
            print(f"❌ Piper error: {e}")
            self.is_speaking = False
    
    def process_audio_stream(self):
        """Process audio with faster response"""
        audio_buffer = []
        silence_counter = 0
        max_silence = int(1.5 / self.chunk_duration)  # Faster - 1.5 seconds
        last_activity_time = time.time()
        
        print("🎧 Audio processing started (FAST MODE)")
        
        while self.running:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                energy = np.mean(np.square(audio_chunk))
                current_time = time.time()
                
                if energy > self.silence_threshold:
                    # Voice detected
                    audio_buffer.extend(audio_chunk)
                    silence_counter = 0
                    last_activity_time = current_time
                else:
                    # Silence detected
                    if len(audio_buffer) > 0:
                        silence_counter += 1
                        
                        # Faster processing - less silence needed
                        if silence_counter >= max_silence:
                            duration = len(audio_buffer) / self.sample_rate
                            
                            if duration >= self.min_speech_duration:
                                audio_array = np.array(audio_buffer, dtype=np.float32)
                                # Process immediately in same thread for speed
                                threading.Thread(
                                    target=self.process_speech_fast, 
                                    args=(audio_array,), 
                                    daemon=True
                                ).start()
                            
                            # Reset
                            audio_buffer = []
                            silence_counter = 0
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Audio processing error: {e}")
    
    def process_speech_fast(self, audio_data: np.ndarray):
        """Fast speech processing pipeline"""
        print("\n🚀 FAST PROCESSING...")
        
        # Fast transcription
        text = self.transcribe_audio(audio_data)
        if not text:
            return
            
        # Fast response generation
        response = self.generate_response(text)
        
        # Fast TTS
        self.text_to_speech_fast(response)
        
        print("🎤 Ready for next input...\n")
    
    def start(self):
        """Start the assistant"""
        try:
            logger.info("🚀 Starting FAST AI Assistant (Piper TTS Only)...")
            
            # Initialize models
            self.initialize_models()
            
            self.running = True
            
            # Start audio processing
            audio_thread = threading.Thread(target=self.process_audio_stream, daemon=True)
            audio_thread.start()
            
            print("\n" + "="*60)
            print("🤖 FAST AI ASSISTANT READY")
            print("="*60)
            print("⚡ Features:")
            print("   - Piper TTS streaming (no OpenVoice)")
            print("   - Faster response times")
            print("   - Real-time audio generation")
            print("   - Low latency mode")
            print("="*60 + "\n")
            
            with sd.InputStream(
                callback=self.audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=np.float32,
                latency='low'  # Low latency audio input
            ):
                while self.running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            logger.info("Stopping assistant...")
            self.stop()
        except Exception as e:
            logger.error(f"Error starting assistant: {e}")
            raise
    
    def stop(self):
        """Stop the assistant"""
        self.running = False
        self.is_speaking = False
        if self.tts_stream:
            try:
                self.tts_stream.close()
            except:
                pass
        logger.info("✅ AI Assistant stopped")

def signal_handler(signum, frame):
    """Handle Ctrl+C"""
    logger.info("Shutting down...")
    sys.exit(0)

def main():
    """Main function"""
    signal.signal(signal.SIGINT, signal_handler)
    
    if not (PROJECT_DIR / "models").exists():
        logger.error("Models directory not found")
        sys.exit(1)
    
    assistant = OfflineAIAssistant()
    
    try:
        assistant.start()
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        logger.error("Requirements:")
        logger.error("1. llama.cpp server running")
        logger.error("2. Piper voice model downloaded")
        logger.error("3. pip install piper-tts")
        sys.exit(1)

if __name__ == "__main__":
    main()