#!/bin/bash

echo "=============================="
echo "Starting Email ML Pipeline"
echo "=============================="

cd "$(dirname "$0")/.."

echo "Step 1: Running Label Generator..."
python Code/Train/Label_Email.py

if [ $? -ne 0 ]; then
    echo "Labeling failed. Stopping pipeline."
    exit 1
fi

echo "Step 2: Training Model..."
python Code/Train/Model.py

if [ $? -ne 0 ]; then
    echo "Training failed. Stopping pipeline."
    exit 1
fi

echo "=============================="
echo "Pipeline Completed Successfully!"
echo "=============================="