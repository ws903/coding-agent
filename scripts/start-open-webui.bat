@echo off
REM start-open-webui.bat -- launch Open WebUI and open browser
REM Starts existing container; creates it on first run.
REM Requires: Docker Desktop running, OLLAMA_HOST=0.0.0.0 set for Ollama.

docker start open-webui 2>nul || docker run -d -p 8080:8080 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main

start http://localhost:8080
