# Azure TTS Streamlit App

A web-based Text-to-Speech application using Azure OpenAI's TTS API, built with Streamlit for easy deployment and mobile access.

## 🌟 Features

- **� File Upload Support**: Upload and parse TXT, PDF, or DOCX files automatically
- **�📱 Mobile-Friendly**: Works perfectly on phones and tablets
- **🎵 Multiple Voices**: Choose from 6 different voices (alloy, echo, fable, onyx, nova, shimmer)
- **⚡ Parallel Processing**: Fast audio generation with automatic text chunking
- **🎧 Audio Player**: Built-in player with chunk navigation
- **💾 Download Support**: Download individual chunks or complete audio
- **🆓 Free Deployment**: Deploy for free on Streamlit Cloud

## 🚀 Live Demo

🔗 **[Try it here](https://your-app-name.streamlit.app)** (link will be available after deployment)

## 📱 Mobile Access

This app is optimized for mobile devices! Access it from your phone's browser for on-the-go text-to-speech conversion.

## 🛠️ Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/azure-tts-streamlit.git
   cd azure-tts-streamlit
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run streamlit_tts_app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## ⚙️ Configuration

### Required (TTS)

Choose one provider with `TTS_PROVIDER`:

- `azure_openai`: set `AZURE_TTS_ENDPOINT`, `AZURE_API_KEY`, and
  `AZURE_TTS_MODEL`. The model value is the Azure deployment name. A resource
  base URL such as `https://RESOURCE.openai.azure.com` is recommended.
- `azure_speech`: set `AZURE_SPEECH_ENDPOINT`, `AZURE_SPEECH_KEY`, and
  `AZURE_SPEECH_VOICE`. For the MAI voice in Azure AI Foundry, use a resource
  URL such as `https://RESOURCE.cognitiveservices.azure.com` and a voice such
  as `tr-TR-Elif:MAI-Voice-2-Flash`.

### Optional (Podcast Transformation)
- Set `AZURE_LLM_ENDPOINT` - Your Azure OpenAI v1 endpoint, such as
  `https://RESOURCE.services.ai.azure.com/openai/v1`.
- Set `AZURE_LLM_AUTH` to `api_key` or `default_azure_credential`.
- For `api_key`, set `AZURE_LLM_API_KEY` (it can be the TTS key when both
  services use the same resource).
- Set `AZURE_LLM_DEPLOYMENT` - Your Responses-compatible deployment name,
  such as `gpt-5.6-luna`.

The LLM backend uses the OpenAI Responses API. `default_azure_credential`
authenticates through a local Azure CLI/developer login or a managed identity
with access to the Foundry resource. Streamlit Community Cloud does not provide
an Azure managed identity, so API-key authentication is normally required
there.

Configuration methods:
- **Production**: Use Streamlit secrets (`secrets.toml`)
- **Local Development**: Use `.env` file (copy `.env.example` to `.env`)

The same TTS fields are available in the app sidebar. Configured API keys stay
server-side and are never populated into browser fields. An optional key
override entered in the sidebar remains in that browser session and is not
written to disk.

For Streamlit Community Cloud, open the app settings, choose **Secrets**, and
paste the relevant entries from `.env.example` using TOML syntax:

```toml
TTS_PROVIDER = "azure_speech"
AZURE_SPEECH_ENDPOINT = "https://RESOURCE.cognitiveservices.azure.com"
AZURE_SPEECH_KEY = "your-key"
AZURE_SPEECH_VOICE = "tr-TR-Elif:MAI-Voice-2-Flash"
```

## 📋 Requirements

- Python 3.7+
- Azure OpenAI TTS API access
- Streamlit
- Requests
- MarkItDown (for file parsing)
- python-docx (for DOCX support)
- pypdf2 & pdfplumber (for PDF support)

## 🎯 How It Works

1. **Text Input**: Type/paste text OR upload a document (TXT, PDF, DOCX)
2. **File Parsing**: Documents are automatically parsed using MarkItDown with smart fallback methods
3. **Text Chunking**: Azure OpenAI uses chunks up to 6,000 characters.
   Azure Speech starts at 2,500 characters and automatically subdivides a
   chunk if the MAI backend rejects its generated audio size.
4. **Parallel Processing**: Multiple API calls process chunks simultaneously
5. **Sequential Playback**: Audio chunks play in order for natural speech flow
6. **Mobile Optimization**: Responsive design works on all devices

## 📁 Project Structure

```
azure-tts-streamlit/
├── streamlit_tts_app.py          # Main Streamlit application
├── requirements.txt               # Python dependencies
├── README.md                     # This file
└── web_tts_app.py               # Alternative Flask version
```

## 🆓 Free Deployment Options

### Streamlit Cloud (Recommended)
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Deploy instantly - no cost!

### Other Options
- Heroku (free tier)
- Railway (free tier)
- Render (free tier)

## 🔐 Security Note

For production use, consider using environment variables or Streamlit secrets for API credentials instead of hardcoding them.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/azure-tts-streamlit/issues).

## ⭐ Support

If this project helps you, please give it a ⭐ on GitHub!
