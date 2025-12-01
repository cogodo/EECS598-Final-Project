#!/bin/bash
# Quick setup and test script for GRPO training

echo "=================================================="
echo "GRPO Training Setup for TinyLlama"
echo "=================================================="

# Check Python version
echo -e "\n1. Checking Python version..."
python --version

# Create virtual environment (optional but recommended)
echo -e "\n2. Do you want to create a virtual environment? (y/n)"
read -r create_venv
if [ "$create_venv" = "y" ]; then
    echo "Creating virtual environment..."
    python -m venv grpo_env
    source grpo_env/bin/activate
    echo "Virtual environment activated!"
fi

# Install dependencies
echo -e "\n3. Installing dependencies..."
pip install --upgrade pip
pip install torch transformers accelerate
pip install wandb
pip install verl

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Run tests
echo -e "\n4. Running setup tests..."
python test_setup.py

echo -e "\n=================================================="
echo "Setup complete!"
echo "=================================================="
echo -e "\nTo start training:"
echo "  cd tiny-grpo"
echo "  python train.py"
echo -e "\nTo monitor training:"
echo "  - Set wandb_project in train.py"
echo "  - Check checkpoints in ./output/"
echo "=================================================="
