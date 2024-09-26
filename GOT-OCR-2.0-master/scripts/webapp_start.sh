#!/bin/bash

# Run your uvicorn server
uvicorn webapp.serve:app --port 8881 --workers 1