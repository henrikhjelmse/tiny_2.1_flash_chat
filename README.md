# TINY 2.1 Flash Chat

A compact yet powerful chatbot web application running [TINY 2.1](https://github.com/WollyMonjac/Tiny-2.1) – an ultraminimal transformer model with just **367,000 parameters**.

## 🚀 Getting Started

### Quick Start (Recommended)

The easiest way to get started is using the startup script, which automatically:
- Checks for Python 3
- Creates a virtual environment
- Installs all dependencies
- Locates TINY.BIN
- Starts the server

**On Linux/Mac:**
```bash
# Make the script executable (first time only)
chmod +x start.sh

# Run it
./start.sh
```

**On Windows:**
```batch
# Double-click start.bat, or run from cmd:
start.bat
```

The server will start at [http://localhost:5014](http://localhost:5014)

**Advanced options (Linux/Mac):**
```bash
./start.sh --port 8080              # Use custom port
./start.sh --host 127.0.0.1         # Listen only locally
./start.sh --bin path/to/TINY.BIN   # Specify model path
./start.sh --port 8080 --host 0.0.0.0
```

**Advanced options (Windows):**
```batch
start.bat --port 8080
start.bat --host 127.0.0.1
start.bat --bin path\to\TINY.BIN
```

### Manual Installation

If you prefer not to use `start.sh`:

#### Requirements
- Python 3.7+

#### Installation

```bash
# Clone or download the project
cd tiny_2.1_flash_chat

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Make sure TINY.BIN exists (model weights)
# Download from: https://github.com/WollyMonjac/Tiny-2.1
```

#### Start the server

```bash
# Simple start (automatically searches for TINY.BIN)
python app.py

# With custom path and port
python app.py --bin path/to/TINY.BIN --port 5000

# Listen on all interfaces
python app.py --host 0.0.0.0 --port 8080
```

Then visit [http://localhost:5014](http://localhost:5014) in your browser.

## 🎮 Usage

1. **Write a question** in English (max 72 characters)
2. **Adjust Creativity** (0–9):
   - **0**: Deterministic – same answer every time
   - **5**: Balanced (default)
   - **9**: Highly variable and imaginative
3. **Click Send** or press Enter
4. Watch the response stream in character by character

### Example Questions

```
what is the capital of peru
what is 7 times 8
is a whale a fish
spell elephant
opposite of hot
```

## 📋 API

### GET `/`
Displays the web interface.

### POST `/ask`
Send a question and receive a streamed response.

**Request:**
```json
{
  "q": "your question here",
  "creativity": 5
}
```

**Response:** Streamed text content (text/plain)

### GET `/info`
Returns model architecture details.

```json
{
  "D": 384,
  "layers": 6,
  "heads": 6,
  "hidden": 1536
}
```

## 🏗️ Architecture

### Model
- **Embedding dimension**: 384
- **Layers**: 6
- **Attention heads**: 6
- **Hidden FFN size**: 1536
- **Vocabulary**: ASCII 32–126 + newline (97 tokens)
- **Context**: 128 tokens (~400 characters)

The model is loaded from TINY.BIN (version 2 format) and runs entirely with NumPy.

### Generation
- Tokenizes input (ASCII offset)
- Feeds through transformer stack
- Uses KV-cache for efficient inference
- Temperature sampling for creative variability
- Generates until `<eof>` or context limit reached

## 🎨 Web Interface

Built with vanilla JavaScript and CSS Grid/Flexbox:
- **Dark theme** with accent color: `#39ff9c`
- **Responsive design** for mobile and desktop
- **Streaming responses** with blinking cursor
- **Conversation history** (session-local only)
- **Quick tips** with example questions

## ⚙️ Configuration

| Flag | Description | Default |
|------|-------------|---------|
| `--bin PATH` | Path to TINY.BIN | Auto-search |
| `--host HOST` | Bind address | `0.0.0.0` |
| `--port PORT` | Listen port | `5014` |

## 📝 Notes

- The model **confidently makes things up** when it doesn't know
- Knows **~3500 facts** (countries, elements, animals, basic math)
- **English only** – other languages produce poor results
- Uses **thread locking** for safe concurrent access (one question at a time)
- **Fully offline** – no internet connection required after startup

## 📂 File Structure

```
tiny_2.1_flash_chat/
├── start.sh         # Linux/Mac launcher (auto-setup & run)
├── start.bat        # Windows launcher (auto-setup & run)
├── app.py           # Main Flask application & model logic
├── requirements.txt # Python dependencies (Flask, NumPy)
├── .gitignore       # Git ignore rules
├── TINY.BIN         # Model weights (not in repo, download separately)
├── venv/            # Virtual environment (created by start.sh/start.bat)
└── README.md        # This file
```

## 🔗 References

- **TINY 2.1 Original**: https://github.com/WollyMonjac/Tiny-2.1
- **Flask**: https://flask.palletsprojects.com/
- **NumPy**: https://numpy.org/

