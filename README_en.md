# Media Gallery Analyzer

[English](README_en.md) | [日本語](README.md)

A powerful media gallery viewer with AI-powered analysis, full-text search, and semantic search capabilities.

## Features

- 📁 **Smart Media Scanning**: Recursive directory scanning with customizable depth
- 🤖 **AI Image Analysis**: Automatic caption and tag generation using LM Studio
- 🎬 **Video Processing**: Frame extraction and analysis for video files
- 🔍 **Advanced Search**:
  - Full-text search with Meilisearch
  - Semantic search with vector embeddings (Qdrant)
  - AI-powered result reranking
- 🌐 **Web Interface**: Clean, responsive gallery interface
- 🔒 **Security**: XSS protection, path validation, secure logging

## Prerequisites

- Node.js (>=14.0.0)
- Python 3.11+
- Docker & Docker Compose (Docker Desktop https://docs.docker.com/desktop/setup/install/mac-install/)
- FFmpeg (for video processing)

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/kz-kk/media-gallery-analyzer.git
cd media-gallery-analyzer
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` file:

```bash
# Required: Set your media directory path
SCAN_PATH=/path/to/your/media/directory

# Optional: Adjust other settings as needed
PORT=3333
LMSTUDIO_URL=http://127.0.0.1:1234
# Scripts (default: audio uses v2/Whisper)
AUDIO_INDEXER_SCRIPT=audio_indexer_v2.py
# ID scheme (default rel8 = sha256(relative path) first 8 hex → int)
ID_SCHEME=rel8
```

### 3. Install Dependencies

```bash
# Node.js dependencies
npm install

# Python dependencies (includes CairoSVG)
python -m venv venv

# Python
python -m venv venv

# mac
source venv/bin/activate  

# Windows Gitbash 
. venv/Scripts/activate

pip install -r requirements.txt
```

### 4. Start Services

```bash
# Start Meilisearch and Qdrant
docker-compose up -d

# Start the application
npm start
```

### 5. Access the Application

| Service | URL | Port/ENV | Description |
|---------|-----|----------|-------------|
| Gallery UI | http://127.0.0.1:3333 | `PORT=3333` | Main application frontend |
| Meilisearch Admin UI | http://127.0.0.1:24900 | Fixed 24900 | Official UI (started via `docker-compose`) |
| Meilisearch API | http://127.0.0.1:17700 | Fixed 17700 → Container 7700 | Search engine API (`MEILI_URL`) |
| Qdrant Dashboard | http://127.0.0.1:26333/dashboard | Fixed 26333 → Container 6333 | Vector DB dashboard |

### Meilisearch UI Initial Connection Setup

When first using the Meilisearch admin interface (http://127.0.0.1:24900), connect with these settings:

- **Name**: `default`
- **Host**: `http://127.0.0.1:17700`
- **API Key**: `masterKey` (default value - from `.env` `MEILI_MASTER_KEY`)

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCAN_PATH` | - | **Required** Media directory path |
| `PORT` | 3333 | Server port |
| `LMSTUDIO_URL` | http://127.0.0.1:1234 | LM Studio API URL |
| `LMSTUDIO_MODEL` | qwen2.5-vl-7b | Vision model name |
| `LMSTUDIO_IMAGE_MODE` | data | Image passing mode (data/http) |
| `MEILI_URL` | http://127.0.0.1:17700 | Meilisearch API URL |
| `QDRANT_URL` | http://127.0.0.1:26333 | Qdrant API URL |
| `EMB_MODEL` | all-MiniLM-L6-v2 | Embedding model (Plamo also supported) |
| `WHISPER_MODEL` | small | Whisper speech model |
| `AUDIO_INDEXER_SCRIPT` | audio_indexer_v2.py | Audio analysis script |
| `ID_SCHEME` | rel8 | Media ID scheme (rel8/rel16/abs8) |

### LM Studio Setup

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load a vision model (e.g., `qwen2.5-vl-7b`)
3. Start the local server on port 1234
4. Update `LMSTUDIO_MODEL` in `.env` if using different model

Notes:
- Image analysis: VLM (e.g., qwen2.5-vl-7b)
- Audio analysis (v2): Whisper for transcription + features; LM Studio is optional for text enrichment
- SVG logos: Two-background rasterization (white/black) via CairoSVG to avoid inverted color or spinner misclassification

## Usage

### Basic Workflow

1. **Scan Media**: Click "Scan Directory" to index your media files
2. **AI Analysis**: Use "Analyze" buttons to generate captions and tags
3. **Search**: 
   - Use the search bar for full-text search
   - Toggle "Semantic Search" for AI-powered similarity search
   - Enable "Rerank Results" for improved accuracy

### Supported Formats

- **Images**: JPG, PNG, GIF, BMP, WebP, SVG (SVG is rasterized before analysis)
- **Videos**: MP4, AVI, MOV, MKV, WebM, M4V, MPG, MPEG
- **Audio**: MP3, WAV, FLAC, AAC

## Development

### File Structure

```
├── analysis/               # Python analysis scripts
│   ├── image_indexer.py   # Image analysis
│   ├── video_indexer.py   # Video analysis
│   ├── get_embedding.py   # Embedding generation
│   └── secure_logging.py  # Security utilities
├── server.js              # Main Node.js server
├── index.html             # Web interface
├── docker-compose.yml     # Service definitions
└── requirements.txt       # Python dependencies
```

### Adding New Features

1. **New Analysis Script**: Add to `analysis/` directory
2. **API Endpoint**: Add route in `server.js`
3. **Environment Variables**: Update `.env.example`
4. **Frontend**: Modify `index.html`

## Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find and kill process using port 3333
lsof -ti:3333 | xargs kill -9
```

**Python Dependencies**
```bash
# Reinstall with specific versions
pip install --force-reinstall -r requirements.txt
```

**Docker Services**
```bash
# Restart all services
docker-compose down && docker-compose up -d
```

## FFmpeg Installation

```bash
# macOS (Homebrew)
brew install ffmpeg

# Windows (winget)
winget install Gyan.FFmpeg

# Verify installation
ffmpeg -version
```

## UI Policy (Audio)

- In the normal gallery view, audio captions/tags are hidden. Click a card to open the lightbox and read the description in the right panel.
- In search results, audio captions/tags are shown to help discovery.

## Acknowledgements

- This project draws inspiration from:
  - media-gallery-viewer (by dai-motoki): https://github.com/dai-motoki/media-gallery-viewer
