#!/bin/bash
# setup.sh - Setup script for the EEG Project on Linux/macOS

# Function to check if a command exists.
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

echo "Checking for Python 3.10..."
if ! command_exists python3.10; then
  echo "Python 3.10 not found."
  
  OS=$(uname)
  if [ "$OS" = "Linux" ]; then
    echo "Detected Linux. Attempting to install Python 3.10 via apt-get..."
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
  elif [ "$OS" = "Darwin" ]; then
    echo "Detected macOS. Attempting to install Python 3.10 via Homebrew..."
    if ! command_exists brew; then
      echo "Homebrew not found. Please install Homebrew first: https://brew.sh/"
      exit 1
    fi
    brew install python@3.10
  else
    echo "Unsupported OS. Please install Python 3.10 manually."
    exit 1
  fi

  if ! command_exists python3.10; then
    echo "Python 3.10 installation failed. Please install it manually."
    exit 1
  fi
else
  echo "Python 3.10 is installed."
fi

echo "Upgrading pip..."
python3.10 -m pip install --upgrade pip

echo "Installing required packages..."
python3.10 -m pip install mne numpy matplotlib rich jinja2 antropy nolds pandas scipy

echo "Setup complete! To run the project, navigate to your project folder and execute:"
echo "    python3.10 main.py"
