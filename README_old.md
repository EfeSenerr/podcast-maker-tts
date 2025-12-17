# 🎵 Podcast Maker 🎧

A Python application that converts text to speech using Azure OpenAI's TTS API with parallel processing for long texts.

## Features

- **Text-to-Speech Conversion**: Uses Azure OpenAI's gpt-4o-mini-tts model
- **Smart Text Chunking**: Automatically splits long texts (>4000 chars) while respecting sentence boundaries
- **Parallel Processing**: Processes multiple chunks simultaneously for faster generation
- **Sequential Playback**: Plays audio chunks in order for natural speech flow
- **GUI Interface**: Easy-to-use graphical interface with Tkinter
- **Multiple Voices**: Supports 6 different voice options (alloy, echo, fable, onyx, nova, shimmer)
- **Real-time Control**: Start, stop, and clear functions

## Setup

1. **Activate your environment**:
   ```powershell
   conda activate xx
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure API credentials** (optional):
   - Edit `config.py` to update your endpoint and API key
   - Or set environment variables `AZURE_TTS_ENDPOINT` and `AZURE_API_KEY`

## Usage

### GUI Interface (Recommended)

Run the application:
```powershell
python azure_tts_app.py
```

1. The application will start with a GUI
2. Configure your API endpoint and key (pre-filled from your provided values)
3. Click "Initialize TTS Client"
4. Paste your text in the text area
5. Select a voice from the dropdown
6. Click "Convert & Play" to generate and play the speech

### Command Line Quick Test

For a quick test, run:
```powershell
python azure_tts_app.py
```
Then choose option "2" for the quick test.

## How It Works

1. **Text Processing**: Long texts are intelligently split into chunks of ~4000 characters
2. **Parallel Generation**: Multiple API calls are made simultaneously to generate audio for each chunk
3. **Sequential Playback**: Audio chunks are played in the correct order to maintain speech flow
4. **Performance**: About 5x faster than real-time audio generation

## Configuration Options

- **Voices**: alloy, echo, fable, onyx, nova, shimmer
- **Max Characters**: 4000 per chunk (API limit is 4096)
- **Max Workers**: 3 parallel API calls (configurable)

## File Structure

```
azure-tts-app/
├── azure_tts_app.py    # Main application
├── .env           # Configuration settings
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Troubleshooting

- **Audio not playing**: Ensure pygame is properly installed and your system has audio output
- **API errors**: Check your endpoint URL and API key
- **Import errors**: Make sure all dependencies are installed in the correct conda environment
- **Rate limits**: The application respects API rate limits with parallel processing controls
