#!/bin/bash
# Serve the tutorial over HTTP so videos and images load correctly.
# Browsers block media when opening HTML via file://
cd "$(dirname "$0")"
echo "Serving at http://localhost:8000"
echo "Open http://localhost:8000/index.html in your browser"
echo "Press Ctrl+C to stop"
python3 -m http.server 8000
