#!/bin/bash

set -e

echo "================================"
echo "TINY 2.1 Flash Chat - Setup"
echo "================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ Found: $PYTHON_VERSION"
echo ""

# Get the directory where the script is located
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Check for virtual environment
if [ -d "venv" ]; then
    echo "✓ Virtual environment found"
    source venv/bin/activate
else
    echo "• Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created"
fi

echo ""
echo "Installing dependencies..."

# Install required packages from requirements.txt
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1

echo "✓ Dependencies installed from requirements.txt"
echo ""

# Check for TINY.BIN
echo "Checking for TINY.BIN..."
TINY_BIN=""

if [ -f "$DIR/TINY.BIN" ]; then
    TINY_BIN="$DIR/TINY.BIN"
    echo "✓ Found TINY.BIN in project root"
elif [ -f "$DIR/dos/TINY.BIN" ]; then
    TINY_BIN="$DIR/dos/TINY.BIN"
    echo "✓ Found TINY.BIN in dos/ directory"
elif [ -f "$DIR/Tiny-2.1/dos/TINY.BIN" ]; then
    TINY_BIN="$DIR/Tiny-2.1/dos/TINY.BIN"
    echo "✓ Found TINY.BIN in Tiny-2.1/dos/ directory"
else
    echo "❌ TINY.BIN not found!"
    echo ""
    echo "Please download TINY.BIN from:"
    echo "  https://github.com/WollyMonjac/Tiny-2.1"
    echo ""
    echo "And place it in one of these locations:"
    echo "  • $DIR/TINY.BIN"
    echo "  • $DIR/dos/TINY.BIN"
    echo "  • $DIR/Tiny-2.1/dos/TINY.BIN"
    echo ""
    exit 1
fi

echo ""
echo "================================"
echo "Starting TINY 2.1 Flash Chat"
echo "================================"
echo ""

# Parse command line arguments
PORT=5014
HOST="0.0.0.0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --bin)
            TINY_BIN="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./start.sh [--port PORT] [--host HOST] [--bin PATH]"
            exit 1
            ;;
    esac
done

echo "🤖 Model: $TINY_BIN"
echo "🌐 Server: http://$HOST:$PORT"
echo "📡 Press Ctrl+C to stop"
echo ""

# Start the Flask app
python3 app.py --bin "$TINY_BIN" --host "$HOST" --port "$PORT"
