#!/bin/bash

set -eo pipefail

# Define the expected file name
EXPECTED_FILE="GOT_weights.zip"
EXTRACTED_DIR="weights" 

# Check if the file exists
if [ ! -f "$EXTRACTED_DIR" ]; then
    echo "Not found $EXTRACTED_DIR. Try to download from the internet or unzip locally"

    if [ ! -f "$EXPECTED_FILE" ]; then
        echo "File $EXPECTED_FILE not found. Downloading..."
        gdown https://drive.google.com/uc?id=1OQrXq_NB_QbJD9yPab6MSj0mcUD4DcrX --output "$EXPECTED_FILE"
    fi

    # Check if the download was successful
    if [ -f "$EXPECTED_FILE" ]; then
        echo "Unziping..."
        
        # Unzip the downloaded file
        unzip "$EXPECTED_FILE" -d "$EXTRACTED_DIR"
        echo "Unzipping completed."

        # Delete the zip file after unzipping
        rm "$EXPECTED_FILE"
        echo "Deleted $EXPECTED_FILE."
        
    else
        echo "Download failed. Exiting..."
        exit 1
    fi
else
    echo "Dir $EXTRACTED_DIR already exists. Skipping download."
fi
