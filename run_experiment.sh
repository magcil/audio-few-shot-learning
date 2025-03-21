#!/bin/bash

# Hardcoded configuration file
CONFIG_FILE="config/experiment_config.json"

# Get experiment folder as input
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_folder>"
    exit 1
fi
NEW_FOLDER="$1"

# Hardcoded command
COMMAND="taskset -c 1-7 python3 src/train_test.py -e $CONFIG_FILE -m config/model_params.json"

# Ensure the experiment folder exists
mkdir -p "$NEW_FOLDER"

# Run the command with nohup, redirecting both stdout and stderr to a log file
nohup $COMMAND > "$NEW_FOLDER/output.log" 2>&1 &

echo "Command output saved to $NEW_FOLDER/output.log"

# Copy the updated config file into the experiment folder
cp "$CONFIG_FILE" "$NEW_FOLDER/"
echo "Copied updated config to $NEW_FOLDER"
