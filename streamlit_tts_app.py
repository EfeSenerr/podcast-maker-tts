import streamlit as st
import requests
import base64
import html
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import io
import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from markitdown import MarkItDown
from PyPDF2 import PdfReader
from docx import Document
import tempfile
import azure.cognitiveservices.speech as speechsdk
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI

# Load environment variables from .env file for  
load_dotenv()

# Podcast transformation prompts
PODCAST_PROMPTS = {
"direct": None,  # No LLM transformation
"podcast-detailed": """You are a skilled podcast script writer. Transform the following text into an engaging, conversational podcast script. The transformed text will be later directly converted to speech, so ensure it flows naturally when spoken.

IMPORTANT RULES:
1. Cover ALL important points from the original text - this is NOT a summary
2. Use natural, conversational language as if explaining to a friend
3. Add transitions between topics to maintain flow
4. Include brief explanations for complex terms
5. Maintain the original meaning and all key details
6. Structure it for audio consumption (no visual references)
7. Use engaging phrases to keep listeners interested
8. The output should be comprehensive and detailed
9. MOST IMPORTANT: Stick to the language and tone of the original text

Original text to transform:
{text}

Generate the complete podcast script:""",

    "podcast-summary": """You are a skilled podcast script writer. Create a concise but comprehensive summary podcast script from the following text. The transformed text will be later directly converted to speech, so ensure it flows naturally when spoken.

IMPORTANT RULES:
1. Capture the MAIN points and key takeaways
2. Use natural, conversational language
3. Keep it brief but informative (aim for 30-40% of original length)
4. Structure it for easy listening
5. Highlight the most important insights
6. Use engaging language to maintain interest
7. MOST IMPORTANT: Stick to the language and tone of the original text

Original text to summarize:
{text}

Generate the summary podcast script:""",

    "podcast-educational": """You are an educational content creator. Transform the following text into an easy-to-understand educational podcast script. The transformed text will be later directly converted to speech, so ensure it flows naturally when spoken.

IMPORTANT RULES:
1. Cover ALL concepts from the original text
2. Explain complex ideas in simple terms
3. Use analogies and examples where helpful
4. Structure content logically for learning
5. Add brief recaps of key points
6. Make it engaging and accessible for all audiences
7. Include all important details - this is NOT a summary
8. MOST IMPORTANT: Stick to the language and tone of the original text

Original text to transform:
{text}

Generate the educational podcast script:""",

    "character-repair": """Repair only character, encoding, spelling, punctuation, and spacing problems in the following text.

STRICT RULES:
1. Preserve the original language, wording, meaning, tone, paragraph structure, and order.
2. Do not summarize, translate, rewrite, rephrase, expand, or remove content.
3. Fix mojibake, replacement characters, broken Unicode characters, incorrect smart quotes, and accidental spacing around punctuation.
4. Restore language-specific characters only when the intended character is clear from context.
5. Correct obvious misspellings, including transposed, missing, repeated, or extra letters.
6. Correct malformed word forms only when the intended word is unambiguous from context.
7. Make the smallest possible correction. Do not replace valid words merely to improve style, fluency, or grammar.
8. If a suspected character or word problem is ambiguous, leave it unchanged.
9. Return only the corrected text without commentary, headings, or quotation marks.

Original text:
{text}

Corrected text:"""
}


def get_config_value(name: str, default: str = "") -> str:
    """Read configuration from Streamlit secrets, then environment variables."""
    try:
        value = st.secrets.get(name, "")
    except (FileNotFoundError, KeyError):
        value = ""
    return str(value or os.getenv(name, default)).strip()


def azure_resource_endpoint(endpoint: str) -> str:
    """Return the resource origin expected by Azure SDK clients."""
    parsed = urlparse(endpoint.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Endpoint must be a complete HTTPS URL.")
    return f"{parsed.scheme}://{parsed.netloc}"


def azure_openai_v1_endpoint(endpoint: str) -> str:
    """Normalize Azure OpenAI and Foundry resource URLs for the v1 API."""
    base_endpoint = endpoint.strip().split("/openai/deployments", 1)[0].rstrip("/")
    if not base_endpoint:
        raise ValueError("LLM endpoint is required.")
    if base_endpoint.endswith("/openai/v1"):
        return f"{base_endpoint}/"
    return f"{base_endpoint}/openai/v1/"


def resolve_text_input(manual_text: str, uploaded_text: str) -> str:
    """Prefer successfully parsed uploads; otherwise retain manually entered text."""
    return uploaded_text if uploaded_text.strip() else manual_text


class AzureOpenAITTSClient:
    def __init__(self, endpoint: str, api_key: str, model: str, audio_format: str = "mp3"):
        """Initialize the Azure TTS client"""
        endpoint = endpoint.strip()
        if "/audio/speech" in endpoint:
            self.endpoint = endpoint
        else:
            self.endpoint = (
                f"{azure_resource_endpoint(endpoint)}/openai/v1/audio/speech"
                "?api-version=preview"
            )
        self.api_key = api_key
        self.model = model
        self.audio_format = audio_format
        
    def chunk_text(self, text: str, max_chars: int = 6000) -> List[str]:
        """Split text into chunks that respect sentence boundaries
        
        GPT-4o mini TTS has a limit of 2000 tokens per request.
        Using ~6000 characters provides a safe margin since tokens are 
        roughly 3-4 characters each (2000 tokens ≈ 6000-8000 chars).
        We use 6000 characters to stay safely within the token limit.
        """
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
            "model": self.model,
            "input": text,
            "voice": voice,
            "response_format": self.audio_format
        }
        
        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "Content-Type": "application/json",
                    "api-key": self.api_key
                },
                json=payload,
                timeout=30
            )
            if not response.ok:
                try:
                    error_detail = response.json().get("error", {}).get("message", response.text)
                except requests.exceptions.JSONDecodeError:
                    error_detail = response.text
                raise RuntimeError(
                    f"Azure OpenAI returned HTTP {response.status_code}: "
                    f"{str(error_detail)[:500]}"
                )
            return response.content
        except requests.exceptions.RequestException as e:
            raise Exception(f"TTS API request failed: {str(e)}")
    
    def convert_text_to_audio_data(self, text: str, voice: str = "alloy", max_workers: int = 3) -> bytes:
        """Convert text to speech and return combined audio data as bytes"""
        chunks = self.chunk_text(text)
        
        if len(chunks) > 1:
            st.info(f"Processing text in {len(chunks)} parts for optimal quality...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        completed_chunks = 0
        
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
                    completed_chunks += 1
                    progress_bar.progress(completed_chunks / len(chunks))
                    st.write(f"✅ Part {index + 1}/{len(chunks)} completed")
                except Exception as e:
                    st.error(f"❌ Error processing part {index + 1}: {e}")
                    results[index] = None
        
        progress_bar.empty()
        
        failed_chunks = [index + 1 for index, audio in enumerate(results) if audio is None]
        if failed_chunks:
            raise RuntimeError(
                "Audio generation failed for part(s): "
                + ", ".join(str(index) for index in failed_chunks)
            )
        audio_chunks = results
        
        # Combine all chunks into a single audio file
        if len(audio_chunks) > 1:
            st.info("Combining audio parts into seamless speech...")
            combined_audio = self.combine_audio_chunks(audio_chunks)
        else:
            combined_audio = audio_chunks[0]
        
        return combined_audio
    
    def combine_audio_chunks(self, audio_chunks: List[bytes]) -> bytes:
        """Combine multiple audio chunks into a single seamless audio file"""
        if not audio_chunks:
            return b""
        
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        # For MP3 files, we can simply concatenate the bytes
        # This works because MP3 is designed to be streamable
        combined_audio = b"".join(audio_chunks)
        return combined_audio


class AzureSpeechTTSClient(AzureOpenAITTSClient):
    def __init__(self, endpoint: str, api_key: str, audio_format: str = "mp3"):
        """Initialize Azure AI Speech for neural and MAI voices."""
        self.speech_config = speechsdk.SpeechConfig(
            subscription=api_key,
            endpoint=azure_resource_endpoint(endpoint)
        )
        self.speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3
        )
        self.audio_format = audio_format

    def text_to_speech(self, text: str, voice: str) -> bytes:
        """Convert text to speech with the Azure Speech SDK."""
        locale_parts = voice.split("-", 2)[:2]
        language = "-".join(locale_parts) if len(locale_parts) == 2 else "en-US"
        ssml = (
            '<speak version="1.0" '
            'xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="{html.escape(language)}">'
            f'<voice name="{html.escape(voice)}">{html.escape(text)}</voice>'
            "</speak>"
        )
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=None
        )
        result = synthesizer.speak_ssml_async(ssml).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return bytes(result.audio_data)

        cancellation = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
        detail = cancellation.error_details or str(cancellation.reason)
        raise RuntimeError(f"Azure Speech synthesis failed: {detail}")


class FileParser:
    """Helper class to parse different file formats"""
    
    @staticmethod
    def parse_txt(file) -> str:
        """Parse TXT file"""
        try:
            return file.read().decode('utf-8')
        except UnicodeDecodeError:
            # Try different encodings
            file.seek(0)
            try:
                return file.read().decode('latin-1')
            except:
                raise Exception("Unable to decode text file. Please ensure it's a valid text file.")
    
    @staticmethod
    def parse_pdf(file) -> str:
        """Parse PDF file using pypdf2"""
        try:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Failed to parse PDF: {str(e)}")
    
    @staticmethod
    def parse_docx(file) -> str:
        """Parse DOCX file using python-docx"""
        try:
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Failed to parse DOCX: {str(e)}")
    
    @staticmethod
    def parse_with_markitdown(file, file_extension: str) -> str:
        """Parse file using MarkItDown library"""
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            # Parse with MarkItDown
            md = MarkItDown()
            result = md.convert(tmp_path)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return result.text_content if hasattr(result, 'text_content') else str(result)
        except Exception as e:
            raise Exception(f"Failed to parse with MarkItDown: {str(e)}")
    
    @staticmethod
    def parse_file(uploaded_file) -> str:
        """Main method to parse uploaded files"""
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Try MarkItDown first as it's more robust
        try:
            return FileParser.parse_with_markitdown(uploaded_file, file_extension)
        except Exception as markitdown_error:
            st.warning(f"MarkItDown parsing failed, trying alternative method...")
            
            # Fallback to specific parsers
            if file_extension == '.txt':
                return FileParser.parse_txt(uploaded_file)
            elif file_extension == '.pdf':
                return FileParser.parse_pdf(uploaded_file)
            elif file_extension in ['.docx', '.doc']:
                return FileParser.parse_docx(uploaded_file)
            else:
                raise Exception(f"Unsupported file format: {file_extension}")

class TextTransformer:
    """Transform text with the Azure OpenAI v1 Responses API."""

    def __init__(self, endpoint: str, api_key: str = "", auth_method: str = "api_key"):
        """Initialize API-key or Microsoft Entra ID authentication."""
        if auth_method == "default_azure_credential":
            client_api_key = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://ai.azure.com/.default"
            )
        elif auth_method == "api_key":
            if not api_key:
                raise ValueError("An LLM API key is required for API-key authentication.")
            client_api_key = api_key
        else:
            raise ValueError(f"Unsupported LLM authentication method: {auth_method}")

        self.client = OpenAI(
            base_url=azure_openai_v1_endpoint(endpoint),
            api_key=client_api_key
        )
    
    def transform_text(self, text: str, style: str, model: str = "gpt-4o-mini") -> str:
        """Transform text using the selected podcast style"""
        if style == "direct" or style not in PODCAST_PROMPTS:
            return text
        
        prompt_template = PODCAST_PROMPTS[style]
        if not prompt_template:
            return text
        
        prompt = prompt_template.format(text=text)
        
        try:
            response = self.client.responses.create(
                model=model,
                instructions=(
                    "Follow the selected transformation instructions exactly. "
                    "Return only the transformed text."
                ),
                input=prompt,
                max_output_tokens=16000
            )
            if not response.output_text:
                raise RuntimeError("The model returned no text.")
            return response.output_text
        except Exception as e:
            raise Exception(f"LLM transformation failed: {str(e)}")

def main():
    st.set_page_config(
        page_title="Podcast Maker 🎧",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎵 Podcast Maker 🎧")
    st.markdown("""Welcome to the Podcast Maker! This application uses Azure OpenAI or Azure AI Speech to convert your text into natural-sounding speech.
                You can paste any text here, and the application will automatically process it to create seamless, high-quality audio. Long texts are handled intelligently behind the scenes for optimal results.""")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        provider_labels = {
            "azure_openai": "Azure OpenAI (gpt-4o-mini-tts)",
            "azure_speech": "Azure AI Speech (Neural / MAI Voice)"
        }
        configured_provider = get_config_value("TTS_PROVIDER", "azure_openai")
        if configured_provider not in provider_labels:
            configured_provider = "azure_openai"
        tts_provider = st.selectbox(
            "TTS Provider",
            options=list(provider_labels),
            index=list(provider_labels).index(configured_provider),
            format_func=lambda value: provider_labels[value]
        )

        endpoint_name = (
            "AZURE_SPEECH_ENDPOINT" if tts_provider == "azure_speech"
            else "AZURE_TTS_ENDPOINT"
        )
        key_name = "AZURE_SPEECH_KEY" if tts_provider == "azure_speech" else "AZURE_API_KEY"
        configured_api_key = get_config_value(key_name)

        with st.expander("🔐 API configuration", expanded=True):
            endpoint = st.text_input(
                "Endpoint",
                value=get_config_value(endpoint_name),
                key=f"{tts_provider}_endpoint",
                help=(
                    "For Azure Speech, use https://RESOURCE.cognitiveservices.azure.com. "
                    "For Azure OpenAI, use the resource base URL or full audio/speech URL."
                )
            ).strip()
            # Remove values retained by sessions created before configured secrets
            # were separated from browser-visible widget state.
            st.session_state.pop(f"{tts_provider}_api_key", None)
            api_key_override = st.text_input(
                "Temporary API key override (optional)",
                value="",
                key=f"{tts_provider}_api_key_override_v2",
                type="password",
                help=(
                    "Leave blank to use the server-side configured secret. "
                    "A value entered here is used only for this browser session."
                )
            ).strip()
            api_key = api_key_override or configured_api_key

            if tts_provider == "azure_openai":
                tts_model = st.text_input(
                    "TTS deployment name",
                    value=get_config_value("AZURE_TTS_MODEL", "gpt-4o-mini-tts"),
                    help="This must be your Azure deployment name, not necessarily the base model name."
                ).strip()
                speech_voice = None
            else:
                tts_model = None
                speech_voice = st.text_input(
                    "Speech voice name",
                    value=get_config_value(
                        "AZURE_SPEECH_VOICE",
                        "tr-TR-Elif:MAI-Voice-2-Flash"
                    ),
                    help="Example MAI voice: tr-TR-Elif:MAI-Voice-2-Flash"
                ).strip()

        configured_llm_api_key = get_config_value("AZURE_LLM_API_KEY")
        with st.expander("🤖 Optional podcast transformation"):
            llm_endpoint = st.text_input(
                "LLM endpoint",
                value=get_config_value("AZURE_LLM_ENDPOINT"),
                help=(
                    "Azure OpenAI or Foundry v1 URL, for example "
                    "https://RESOURCE.services.ai.azure.com/openai/v1."
                )
            ).strip()
            llm_auth_labels = {
                "api_key": "API key",
                "default_azure_credential": "DefaultAzureCredential (Microsoft Entra ID)"
            }
            configured_llm_auth = get_config_value("AZURE_LLM_AUTH", "api_key")
            if configured_llm_auth not in llm_auth_labels:
                configured_llm_auth = "api_key"
            llm_auth_method = st.selectbox(
                "LLM authentication",
                options=list(llm_auth_labels),
                index=list(llm_auth_labels).index(configured_llm_auth),
                format_func=lambda value: llm_auth_labels[value],
                help=(
                    "DefaultAzureCredential uses your local Azure CLI/developer login "
                    "or the deployment's managed identity."
                )
            )
            if llm_auth_method == "api_key":
                llm_api_key_override = st.text_input(
                    "Temporary LLM API key override (optional)",
                    value="",
                    key="llm_api_key_override_v2",
                    type="password",
                    help=(
                        "Leave blank to use the server-side configured secret. "
                        "A value entered here is used only for this browser session."
                    )
                ).strip()
                llm_api_key = llm_api_key_override or configured_llm_api_key
            else:
                llm_api_key = ""
            llm_model = st.text_input(
                "LLM deployment name",
                value=get_config_value("AZURE_LLM_DEPLOYMENT", "gpt-5.6-luna"),
                help="This must match an Azure OpenAI Responses-compatible deployment."
            ).strip()

        llm_auth_ready = (
            llm_auth_method == "default_azure_credential"
            or bool(llm_api_key)
        )

        # Show connection status
        if endpoint and api_key:
            st.success("🔗 TTS configuration ready")
            st.caption("Credentials are present; they are validated on conversion")
        else:
            st.error("❌ Missing Azure TTS Configuration")
            st.caption("Enter credentials above or configure Streamlit secrets")
        
        # Show LLM connection status
        if llm_endpoint and llm_auth_ready:
            st.success("🤖 Azure LLM configuration ready")
            st.caption("Credentials are validated when transformation runs")
        else:
            st.warning("⚠️ Azure LLM Not Configured")
            st.caption("Only Direct mode available")
        
        st.markdown("### 🎵 Voice & Audio Settings")
        
        if tts_provider == "azure_openai":
            voice_options = {
                "alloy": "Alloy - Balanced and natural",
                "echo": "Echo - Clear and articulate",
                "fable": "Fable - Warm and storytelling",
                "onyx": "Onyx - Deep and authoritative",
                "nova": "Nova - Bright and energetic",
                "shimmer": "Shimmer - Soft and gentle"
            }
            selected_voice = st.selectbox(
                "🎤 Voice Style",
                options=list(voice_options.keys()),
                index=0,
                format_func=lambda x: voice_options[x],
                help="Choose the voice character for text-to-speech"
            )
        else:
            voice_options = {speech_voice: speech_voice}
            selected_voice = speech_voice
            st.caption(f"🎤 Voice: {speech_voice or 'Not configured'}")
        
        # Audio format option
        audio_format = st.selectbox(
            "🎵 Audio Format",
            options=["mp3"],
            index=0,
            help="MP3 supports reliable concatenation when long text is split into parts."
        )
        
        # Auto-play setting (disabled by default)
        auto_play = st.checkbox(
            "🔊 Auto-play audio",
            value=False,
            help="Automatically start playing audio after conversion"
        )
        
        st.markdown("### 🎙️ Processing Mode")
        
        # Determine available processing modes based on LLM configuration
        if llm_endpoint and llm_auth_ready:
            available_modes = [
                "direct",
                "character-repair",
                "podcast-detailed",
                "podcast-summary",
                "podcast-educational"
            ]
        else:
            available_modes = ["direct"]
            st.caption("⚠️ Configure the LLM endpoint and authentication to enable transformations")
        
        # Processing mode selection
        processing_mode = st.selectbox(
            "📝 Content Style",
            options=available_modes,
            index=0,
            format_func=lambda x: {
                "direct": "📄 Direct - Convert text as-is",
                "character-repair": "🔤 Text Repair - Fix characters & spelling",
                "podcast-detailed": "🎙️ Podcast (Detailed) - Full content, conversational",
                "podcast-summary": "📋 Podcast (Summary) - Key points only",
                "podcast-educational": "📚 Podcast (Educational) - Easy to understand"
            }[x],
            help="Choose how to process the text before converting to speech"
        )
        
        if processing_mode != "direct":
            st.info("💡 LLM will transform your text into podcast-style content before TTS conversion")
        
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        # Add tabs for text input methods
        input_tab1, input_tab2 = st.tabs(["✍️ Type/Paste Text", "📁 Upload File"])
        
        with input_tab1:
            manual_text_input = st.text_area(
                "Enter your text to convert to speech:",
                value="""You can paste any text here.""",
                height=200,
                key="manual_text_input"
            )

        uploaded_text_input = ""
        with input_tab2:
            st.markdown("**Upload documents to convert to speech**")
            st.caption("You can upload multiple files (up to 3). They will be combined in upload order.")
            
            uploaded_files = st.file_uploader(
                "Choose file(s)",
                type=['txt', 'pdf', 'docx'],
                help="Supported formats: TXT, PDF, DOCX. Upload up to 3 files.",
                accept_multiple_files=True
            )
            
            if uploaded_files:
                # Limit to 3 files
                if len(uploaded_files) > 3:
                    st.warning("⚠️ Maximum 3 files allowed. Only the first 3 will be processed.")
                    uploaded_files = uploaded_files[:3]
                
                st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
                
                # Display file list
                for i, file in enumerate(uploaded_files):
                    st.write(f"  {i+1}. {file.name}")
                
                combined_parsed_text = ""
                
                with st.spinner("📖 Parsing files..."):
                    for i, uploaded_file in enumerate(uploaded_files):
                        try:
                            parsed_text = FileParser.parse_file(uploaded_file)
                            
                            if parsed_text:
                                st.success(f"✅ File {i+1}: Extracted {len(parsed_text):,} characters from {uploaded_file.name}")
                                
                                # Add separator between files if multiple
                                if combined_parsed_text:
                                    combined_parsed_text += "\n\n--- Next Document ---\n\n"
                                combined_parsed_text += parsed_text
                            else:
                                st.error(f"❌ No text could be extracted from {uploaded_file.name}")
                                
                        except Exception as e:
                            st.error(f"❌ Error parsing {uploaded_file.name}: {str(e)}")
                
                if combined_parsed_text:
                    st.success(f"📊 Total extracted: {len(combined_parsed_text):,} characters from {len(uploaded_files)} file(s)")
                    
                    # Show full preview with scrollable text area - using combined_parsed_text
                    with st.expander("📄 Preview extracted text (click to expand)", expanded=False):
                        # Create a unique key for each preview based on file names
                        preview_key = f"preview_{hash(''.join([f.name for f in uploaded_files]))}"
                        st.text_area(
                            "Full extracted content from all files:",
                            value=combined_parsed_text,
                            height=400,
                            disabled=True,
                            key=preview_key
                        )
                        st.caption(f"Total characters: {len(combined_parsed_text):,} | Files: {len(uploaded_files)}")
                    
                    uploaded_text_input = combined_parsed_text

        text_input = resolve_text_input(manual_text_input, uploaded_text_input)

        # Safety: Limit input to ~10 pages (about 30,000 characters)
        MAX_INPUT_CHARS = 50000  # ~10 pages (assuming 3000 chars/page)
        if len(text_input) > MAX_INPUT_CHARS:
            st.warning(f"⚠️ Input too long! Please limit your text to about 10 pages (~{MAX_INPUT_CHARS} characters). You entered {len(text_input):,} characters.")
        
        # Convert button
        if st.button("🎵 Convert to Speech", type="primary", use_container_width=True):
            if not endpoint or not api_key:
                st.error("Please provide both API endpoint and key in the sidebar.")
                return

            if tts_provider == "azure_openai" and not tts_model:
                st.error("Please provide the Azure OpenAI TTS deployment name.")
                return

            if tts_provider == "azure_speech" and not selected_voice:
                st.error("Please provide an Azure Speech voice name.")
                return

            if not text_input.strip():
                st.error("Please enter some text to convert")
                return

            if len(text_input) > MAX_INPUT_CHARS:
                st.error(f"❌ Input too long! Please limit your text to about 10 pages (~{MAX_INPUT_CHARS} characters). You entered {len(text_input):,} characters.")
                return

            try:
                final_text = text_input.strip()
                
                # Apply LLM transformation if selected
                if processing_mode != "direct":
                    if not llm_endpoint or not llm_auth_ready:
                        st.error(
                            "❌ LLM configuration is incomplete. Configure the endpoint "
                            "and selected authentication method."
                        )
                        return
                    
                    with st.spinner(f"🤖 Transforming text to {processing_mode} style..."):
                        st.info(f"Using {llm_model} to transform the text...")
                        transformer = TextTransformer(
                            llm_endpoint,
                            llm_api_key,
                            llm_auth_method
                        )
                        final_text = transformer.transform_text(final_text, processing_mode, llm_model)
                        st.success(f"✅ Text transformed! New length: {len(final_text):,} characters")
                        
                        # Show transformed text preview
                        with st.expander("📝 Preview transformed text", expanded=False):
                            st.text_area(
                                "Podcast script:",
                                value=final_text,
                                height=300,
                                disabled=True,
                                key="transformed_text_preview"
                            )
                
                # Initialize TTS client
                with st.spinner("Initializing TTS client..."):
                    if tts_provider == "azure_speech":
                        tts_client = AzureSpeechTTSClient(endpoint, api_key, audio_format)
                    else:
                        tts_client = AzureOpenAITTSClient(
                            endpoint,
                            api_key,
                            tts_model,
                            audio_format
                        )

                # Convert text to audio
                with st.spinner("Converting text to speech..."):
                    combined_audio = tts_client.convert_text_to_audio_data(final_text, selected_voice)

                if not combined_audio:
                    st.error("Failed to generate audio")
                    return

                st.success("✅ Audio conversion completed!")

                # Store combined audio in session state
                st.session_state.combined_audio = combined_audio

            except Exception as e:
                st.error(f"❌ Conversion failed: {str(e)}")
    
    with col2:
        st.header("Audio Player")
        
        if 'combined_audio' in st.session_state and st.session_state.combined_audio:
            combined_audio = st.session_state.combined_audio
            
            # Audio info panel
            with st.container():
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    audio_size_mb = len(combined_audio) / (1024 * 1024)
                    st.metric("📊 Audio Size", f"{audio_size_mb:.2f} MB")
                with col_info2:
                    st.metric("🎵 Voice", voice_options.get(selected_voice, selected_voice).split(' - ')[0])
            
            st.markdown("**🎵 Your Audio is Ready!**")
            
            # Main audio player
            st.audio(
                combined_audio, 
                format=f'audio/{audio_format}',
                start_time=0,
                autoplay=auto_play,
                loop=False
            )
            
            # Additional audio information
            show_audio_info = st.checkbox("📊 Show detailed audio info", help="Display technical audio information")
            
            if show_audio_info:
                audio_size_kb = len(combined_audio) / 1024
                st.info(f"📊 Format: {audio_format.upper()} | Voice: {selected_voice} | Size: {audio_size_kb:.1f} KB")
            
            # Download section
            st.markdown("---")
            st.markdown("**💾 Download Audio**")
            
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                st.download_button(
                    label="⬇️ Download Audio",
                    data=combined_audio,
                    file_name=f"podcast_{selected_voice}.{audio_format}",
                    mime=f"audio/{audio_format}",
                    use_container_width=True,
                    help="Download the complete audio file"
                )
            
            with col_download2:
                # Create a base64 encoded version for sharing
                audio_b64 = base64.b64encode(combined_audio).decode()
                audio_size_mb = len(combined_audio) / (1024 * 1024)
                st.write(f"📋 File size: {audio_size_mb:.2f} MB")
            
            # Playback tips
            with st.expander("� Playback Tips", expanded=False):
                st.markdown("""
                **🎵 Audio Playback:**
                - Use your browser's built-in controls for play/pause/seek
                - Right-click the audio player for additional options
                - The audio will play continuously without interruption
                - Compatible with all modern browsers and mobile devices
                
                **💾 Download Options:**
                - Click "Download Audio" to save the file locally
                - The downloaded file works with any audio player
                - Perfect for offline listening or sharing
                """)
        else:
            st.info("👆 Convert some text to see the audio player")
            st.markdown("""
            **🎵 Audio Player Features:**
            - 🎤 6 different voice styles to choose from
            - 🎵 Seamless audio playback without interruptions
            - 📊 Multiple audio format support (MP3, WAV, OGG)
            - 💾 Easy download for offline listening
            - � Mobile-optimized player controls
            - 🔊 Optional auto-play functionality
            """)
    
    # Instructions section
    with st.expander("📖 How to Use This App", expanded=False):
        st.markdown("""
        ### 🚀 Quick Start Guide
        
        1. **🔐 Configure Credentials**: Ensure your Azure OpenAI endpoint and API key are set in secrets
        2. **🎤 Choose Voice**: Select from 6 different voice personalities with unique characteristics
        3. **�️ Select Processing Mode**: Choose Direct (as-is) or Podcast style (AI-enhanced)
        4. **📝 Enter Text**: Type/paste text OR upload up to 3 documents (TXT, PDF, DOCX)
        5. **�️ Preview**: View the full extracted text before conversion
        6. **🎵 Convert**: Click "Convert to Speech" to generate high-quality audio
        7. **🎧 Listen**: Click play to listen (auto-play is disabled by default)
        8. **💾 Download**: Save the complete audio file
        
        ### 🎵 Voice Personalities
        - **Alloy**: Balanced and natural - great for general content
        - **Echo**: Clear and articulate - perfect for educational material  
        - **Fable**: Warm and storytelling - ideal for narratives and stories
        - **Onyx**: Deep and authoritative - excellent for presentations
        - **Nova**: Bright and energetic - suitable for marketing content
        - **Shimmer**: Soft and gentle - perfect for meditation or relaxation
        
        ### ✨ Advanced Features
        - ✅ **Multiple File Upload**: Upload up to 3 files (TXT, PDF, DOCX) and combine them
        - ✅ **Full Text Preview**: View the complete extracted text before conversion
        - ✅ **Podcast Transformation**: Use AI to transform text into engaging podcast scripts
        - ✅ **Multiple Styles**: Choose from Direct, Detailed Podcast, Summary, or Educational modes
        - ✅ **Smart Text Processing**: Automatically handles long texts with intelligent processing
        - ✅ **Multiple Parsers**: Uses MarkItDown with fallback parsers for robust file handling
        - ✅ **Parallel Processing**: Faster generation with multi-threaded conversion
        - ✅ **Seamless Audio**: Creates continuous, uninterrupted audio playback
        - ✅ **Mobile-Optimized**: Responsive design works perfectly on phones
        - ✅ **Multiple Audio Formats**: Choose between MP3, WAV, and OGG formats
        - ✅ **Secure Credentials**: API keys are completely hidden and secure
        - ✅ **Progress Tracking**: Real-time feedback during audio generation
        - ✅ **Easy Downloads**: Save complete audio files with one click
        
        ### 🎙️ Processing Modes
        - **📄 Direct**: Convert text exactly as-is - no modifications
        - **🎙️ Podcast (Detailed)**: Transform into conversational podcast covering ALL points
        - **📋 Podcast (Summary)**: Create a concise summary podcast with key takeaways
        - **📚 Podcast (Educational)**: Make complex content easy to understand
        
        ### 🎛️ Audio Controls
        - **Auto-Play**: Enable to automatically start playing audio after conversion
        - **MP3 Output**: Long audio can be combined reliably and downloaded as MP3
        - **Browser Controls**: Use your browser's native audio controls for full playback control
        - **Audio Info**: View technical details about your generated audio
        
        ### 📱 Mobile Usage Tips
        - All controls are touch-friendly and responsive
        - Use landscape mode for the best experience
        - Audio files work with your device's native audio controls
        - Downloads save directly to your device's download folder
        """)
    
    # Performance Tips
    with st.expander("🚀 Performance & Tips", expanded=False):
        st.markdown("""
        ### ⚡ Performance Optimization
        - **Smart Processing**: Texts are automatically processed in optimal segments for best quality
        - **Parallel Generation**: Multiple parts are generated simultaneously for faster results
        - **Seamless Combining**: Audio parts are seamlessly merged into continuous speech
        - **Memory Efficient**: Audio data is streamed efficiently without excessive memory usage
        
        ### 💡 Pro Tips
        - **File Upload**: Upload PDF documents, Word files, or text files - they'll be automatically parsed
        - **Long Documents**: The app automatically handles long texts - just paste and convert!
        - **Voice Testing**: Try different voices with the same text to find your preferred style
        - **Text Preview**: After uploading a file, you can preview the extracted text before converting
        - **Mobile Downloads**: Audio files can be saved directly to your phone for offline listening
        - **Browser Compatibility**: Works best in modern browsers with HTML5 audio support
        
        ### 🔧 Troubleshooting
        - **No Audio**: Check that your API credentials are correct and valid
        - **Slow Generation**: Large texts take longer - progress bars show real-time status
        - **Download Issues**: Ensure your browser allows downloads from this domain
        """)
    
    # Security section
    with st.expander("🔐 Security & Privacy", expanded=False):
        st.markdown("""
        ### 🛡️ Security Features
        - **Hidden API Keys**: Your credentials are masked and never displayed in logs
        - **Secure Storage**: Deployed keys can be stored in Streamlit secrets
        - **Session-Only Overrides**: Keys entered in the sidebar are not written to disk
        - **No Persistence**: Audio data is not permanently stored on servers
        - **HTTPS Ready**: Fully compatible with secure HTTPS deployments
        
        ### 🔒 Privacy Considerations
        - **Text Processing**: Your text is sent to the selected Azure service
        - **Temporary Storage**: Audio is generated and delivered directly to your browser
        - **No Tracking**: This app doesn't track or store your personal data
        - **Local Downloads**: Generated audio files are saved locally to your device
        """)
    
    # Footer with enhanced information
    st.markdown("---")
    col_footer1, col_footer2, col_footer3 = st.columns(3)
    
    with col_footer1:
        st.markdown("**🎵 Podcast Maker 🎧**")
        st.markdown("Built with ❤️ using Streamlit")
    
    with col_footer2:
        st.markdown("**🔗 Powered By**")
        st.markdown("Azure OpenAI / Azure AI Speech")
    
    with col_footer3:
        st.markdown("**🔗 Github**")
        st.markdown("[View the Repository](https://github.com/EfeSenerr/podcast-maker-tts)")
if __name__ == "__main__":
    main()
