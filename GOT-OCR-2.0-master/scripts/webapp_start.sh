#!/bin/bash

# Run your uvicorn server
uvicorn webapp.serve:app --host 0.0.0.0 --port 8881 --workers 1