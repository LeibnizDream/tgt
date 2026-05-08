# TGT – Overview

![Watch video](documentation_diagrams/video.png)

## What is TGT?

TGT (Transcription Glossing Translation) is an end-to-end system designed to support linguistic data processing workflows.

It provides a unified pipeline that integrates:

- **Automatic Speech Recognition (ASR)** for transcription
- **Machine Translation (MT)** for translation
- **Glossing models** for linguistic annotation
- **Data handling utilities** for structured datasets (e.g., Excel/CSV)

The system is built to be **usable by non-technical users**, while maintaining a modular and extensible backend architecture that allows developers to integrate new models or processing strategies.

---

## Key Capabilities

- Process raw linguistic data (audio or text) into annotated formats
- Automate repetitive annotation tasks (transcription, translation, glossing)
- Support multiple model providers (local and API-based)
- Handle structured datasets (e.g., spreadsheet-based workflows)
- Provide a web-based interface for interaction

---

## Design Goals

- **Modularity**: Clear separation between processing strategies (ASR, translation, glossing)
- **Extensibility**: Easy integration of new models or providers
- **Reproducibility**: Consistent processing pipelines
- **Usability**: Accessible to linguists without programming experience
- **Deployment flexibility**: Local, server-based, or containerized

---

# Setup Guide

## Prerequisites

- **Python**: `3.11+`
- **uv**: Fast Python package manager
- **Node.js**: `18+` (for the frontend build)

## Configuration

### Environment Variables

Create a `.env` file in the `backend/materials/` directory to store your API keys and configuration secrets.

> ⚠️ **Important**: Without this file, OneDrive integration will not be available, and some models for transcription and translation will not be accessible.

### `.env` File Contents

```env
# Hugging Face API Key (required)
# Get your key from: https://huggingface.co/settings/tokens
HUGGING_KEY=your_huggingface_api_key_here

# OneDrive Integration (required for OneDrive support)
# Register an app in Azure Portal and obtain these credentials
TENANT_ID=your_azure_tenant_id_here
CLIENT_ID=your_azure_client_id_here
CLIENT_SECRET=your_azure_client_secret_here

# Optional: DeepL Translation API
# Sign up at: https://www.deepl.com/pro-api
DEEPL_API_KEY=your_deepl_api_key_here

# Optional: Google Gemini API
# Get your key from: https://aistudio.google.com
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

## Installation & Deployment Options

### Option 1: ZAS Members

If you're part of ZAS:

1. Contact the code owner for server connection details
2. Follow the specific connection instructions provided by the owner

> 🔒 **Security Note**: Server paths are not included in this documentation for security reasons.

### Option 2: Local Development Setup

For local development or custom server deployment (always pull the latest version of the code):

#### 1. Clone repository

```bash
# Clone repository
git clone https://github.com/camelo-cruz/TGT.git

# Navigate into the project directory
cd TGT
```

#### 2. Set up the Python environment

```bash
# Create virtual environment and install all dependencies from the lock file
uv sync

# Activate the environment
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
```

All exact package versions are pinned in `backend/uv.lock` — no version conflicts, reproducible across machines.

#### 3. Build the frontend

```bash
cd ../frontend
npm install
npm run build
```

#### 4. Start the Application

From the `backend` directory:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The application will be available at: `http://127.0.0.1:8000`

---

## CLI Usage

The backend can also be used directly from the terminal without the web interface, using `backend/inference/worker.py`.

### Syntax

```bash
python -m inference.worker <action> <language> <base_dir> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `action` | `transcribe` \| `translate` \| `gloss` \| `transliterate` |
| `language` | Language name or code (e.g. `english`, `german`, `zh`) |
| `base_dir` | Path to the folder containing the data to process |

### Options

| Option | Description |
|--------|-------------|
| `--format` | `plain` (default) or `labvanced` |
| `--instruction` | Required for labvanced: `automatic` \| `corrected` \| `sentences` |
| `--translation-model` | Translation model name (e.g. `gemini`, `deepl`, `qwen`) |
| `--glossing-model` | Glossing model name (e.g. `gemini`, `qwen`, `spacy`) |

### Examples

**Transcribe audio files in a folder (plain format):**
```bash
python -m inference.worker transcribe english /data/recordings
```

**Translate an already-transcribed folder:**
```bash
python -m inference.worker translate german /data/recordings --translation-model gemini
```

**Process a Labvanced export:**
```bash
python -m inference.worker transcribe greek /data/experiment --format labvanced --instruction automatic
```

**Gloss a Labvanced dataset using a local model:**
```bash
python -m inference.worker gloss turkish /data/experiment \
  --format labvanced \
  --instruction sentences \
  --glossing-model qwen \
  --translation-model qwen
```

> For labvanced format, `base_dir` should contain one or more folders named `Session_*`. The CLI will only process those.

## Important Notes

### Security Best Practices

- 🚫 **Never commit your `.env` file to version control**
- 🔐 Keep all API keys and secrets private
- 📁 Ensure the `.env` file has proper file permissions (readable only by the application user)

### Optional Features

- **DeepL Translation**: If you don't plan to use DeepL API, you can skip adding the `DEEPL_API_KEY` or modify the translation factory to remove the DeepL strategy.
- **OneDrive Integration**: Requires all Azure-related environment variables to be properly configured. If OneDrive is used, be sure to:
  - Add the correct paths for authentication in Azure.
  - Give full permissions to users.
- **Local Inference with Qwen (via Ollama)**: If you want to run Qwen models locally without an API key, you need to install [Ollama](https://ollama.com) and pull the desired Qwen model before starting the application:

  ```bash
  # Install Ollama (see https://ollama.com for platform-specific instructions)
  # Then pull the Qwen model you intend to use, for example:
  ollama pull qwen2.5:7b
  ```

  Once Ollama is running and the model is available, the application will use it automatically for local inference. No API key is required for this option.

### Troubleshooting

- Verify Python version: `python --version`
- Check installed packages: `uv pip list`
- Re-sync the environment: `uv sync` (run from `backend/`)
- Ensure the `.env` file is in the correct location: `backend/materials/.env`
- Validate API keys are correctly formatted and have necessary permissions

## Getting API Keys

| Service | How to Get API Key |
|---------|-------------------|
| **Hugging Face** | 1. Create account at [huggingface.co](https://huggingface.co)<br>2. Go to Settings > Access Tokens<br>3. Create new token |
| **Azure/OneDrive** | 1. Go to [Azure Portal](https://portal.azure.com)<br>2. Register new application<br>3. Note down Tenant ID, Client ID, and Client Secret |
| **DeepL** | 1. Sign up at [DeepL Pro](https://www.deepl.com/pro-api)<br>2. Get API key from account dashboard |
| **Google Gemini** | 1. Go to https://aistudio.google.com<br>2. Sign in with your Google account<br>3. Click “Get API key”<br>4. Create a new API key<br>5. Copy and store it securely |

## Support

For additional help:
- Contact your code administrator at camelo.cruz@leibniz-zas.de
- Check the application logs for error details
- Verify all environment variables are correctly set
