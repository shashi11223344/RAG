@echo off
set GROQ_API_KEY=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
cd /d "%~dp0"
python -m uvicorn backend.main:app --reload --port 8000
pause
