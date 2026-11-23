# Airllmmetrix

Airllmmetrix profiles local language-model inference and monitors system resource
use, with an optional security audit for local model runners.

## Requirements

- Python
- `psutil` (`python -m pip install psutil`)
- A locally running Ollama or OpenAI-compatible inference server for profiling
  and behavioral security checks

## Usage

Run these commands from the repository root:

```powershell
python src\analytics.py --model qwen2.5-coder:3b
python src\core_monitor.py
python src\security_guard.py --model qwen2.5-coder:3b
```

The profiler and security audit write reports and telemetry under
`benchmarks\profiles`. These machine-specific generated captures are excluded
from Git; run the tools locally to create fresh results.
