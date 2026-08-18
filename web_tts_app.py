import os
import requests
import json
import io
import threading
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from flask import Flask, render_template, request, jsonify, send_file, Response
import tempfile
import uuid
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class AzureTTSClient:
    def __init__(self, endpoint: str, api_key: str):
        """Initialize the Azure TTS client"""
        self.endpoint = endpoint
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        
    def chunk_text(self, text: str, max_chars: int = 4000) -> List[str]:
        """Split text into chunks that respect sentence boundaries"""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            # If a single sentence is too long, split by words
            if len(sentence) > max_chars:
                words = sentence.split()
                temp_chunk = ""
                
                for word in words:
                    if len(temp_chunk + " " + word) <= max_chars:
                        temp_chunk += (" " + word) if temp_chunk else word
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                
                if temp_chunk:
                    if len(current_chunk + " " + temp_chunk) <= max_chars:
                        current_chunk += (" " + temp_chunk) if current_chunk else temp_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = temp_chunk
            else:
                # Check if adding this sentence exceeds the limit
                if len(current_chunk + " " + sentence) <= max_chars:
                    current_chunk += (" " + sentence) if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def text_to_speech(self, text: str, voice: str = "alloy") -> bytes:
        """Convert text to speech using Azure OpenAI TTS API"""
        payload = {
            "model": "gpt-4o-mini-tts",
            "input": text,
            "voice": voice
        }
        
        try:
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            raise Exception(f"TTS API request failed: {str(e)}")
    
    def convert_text_to_audio_data(self, text: str, voice: str = "alloy", max_workers: int = 3) -> List[bytes]:
        """Convert text to speech and return list of audio data as bytes"""
        chunks = self.chunk_text(text)
        print(f"Text split into {len(chunks)} chunks")
        
        # Submit all chunks for processing in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunks
            future_to_index = {
                executor.submit(self.text_to_speech, chunk, voice): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results in order
            results = [None] * len(chunks)
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    audio_data = future.result()
                    results[index] = audio_data
                    print(f"Chunk {index + 1}/{len(chunks)} processed")
                except Exception as e:
                    print(f"Error processing chunk {index + 1}: {e}")
                    results[index] = None
        
        # Filter out None results
        audio_chunks = [audio for audio in results if audio is not None]
        return audio_chunks

# Flask Web Application
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize TTS client with environment variables
azure_endpoint = os.getenv('AZURE_TTS_ENDPOINT')
azure_api_key = os.getenv('AZURE_API_KEY')

tts_client = None
if azure_endpoint and azure_api_key:
    try:
        tts_client = AzureTTSClient(azure_endpoint, azure_api_key)
        print("✅ TTS Client initialized from environment variables")
    except Exception as e:
        print(f"❌ Failed to initialize TTS client: {e}")
else:
    print("⚠️  Azure credentials not found in environment variables")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/initialize', methods=['POST'])
def initialize_client():
    """Initialize TTS client"""
    global tts_client
    
    data = request.get_json()
    endpoint = data.get('endpoint', '').strip()
    api_key = data.get('api_key', '').strip()
    
    if not endpoint or not api_key:
        return jsonify({'error': 'Please provide both endpoint and API key'}), 400
    
    try:
        tts_client = AzureTTSClient(endpoint, api_key)
        return jsonify({'success': True, 'message': 'TTS Client initialized successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to initialize TTS client: {str(e)}'}), 500

@app.route('/api/convert', methods=['POST'])
def convert_text():
    """Convert text to speech and return audio data as base64"""
    global tts_client
    
    if not tts_client:
        return jsonify({'error': 'TTS client not initialized'}), 400
    
    data = request.get_json()
    text = data.get('text', '').strip()
    voice = data.get('voice', 'alloy')
    
    if not text:
        return jsonify({'error': 'Please provide text to convert'}), 400
    
    try:
        # Convert text to audio data
        audio_chunks = tts_client.convert_text_to_audio_data(text, voice)
        
        if not audio_chunks:
            return jsonify({'error': 'Failed to generate audio'}), 500
        
        # Convert audio chunks to base64 for web delivery
        audio_data_list = []
        for i, audio_data in enumerate(audio_chunks):
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            audio_data_list.append({
                'index': i,
                'data': audio_b64,
                'type': 'audio/mpeg'
            })
        
        return jsonify({
            'success': True, 
            'message': 'Audio conversion completed',
            'audio_chunks': audio_data_list,
            'total_chunks': len(audio_chunks)
        })
    except Exception as e:
        return jsonify({'error': f'TTS conversion failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("Starting Azure TTS Web Application...")
    print("Access the application at: http://localhost:5000")
    print("Or from your phone using your computer's IP address: http://[YOUR-IP]:5000")
    
    # Print configuration info
    if azure_endpoint:
        print(f"Azure TTS Endpoint: {azure_endpoint}")
    if azure_api_key:
        print(f"Azure API Key: {'*' * (len(azure_api_key) - 8) + azure_api_key[-8:]}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
