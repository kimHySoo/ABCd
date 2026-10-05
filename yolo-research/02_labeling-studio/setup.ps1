$ErrorActionPreference = "Stop"

# Shared with rummikub-webcam: this venv (Python 3.10, folder "abcd") is also
# registered as the "boardgame-ai" Jupyter kernel used by its notebooks.
# Uses uv (https://docs.astral.sh/uv/) to fetch Python 3.10 and install
# packages without needing a system-wide Python install.
if (-not (Test-Path -LiteralPath "abcd")) {
    uv venv --python 3.10 abcd
}

uv pip install torch==2.13.0 torchvision==0.28.0 `
    --index-url https://download.pytorch.org/whl/cu130 `
    --python abcd\Scripts\python.exe
uv pip install -r requirements.txt --python abcd\Scripts\python.exe

.\abcd\Scripts\python.exe -m ipykernel install --user --name boardgame-ai --display-name "Python 3.10 (boardgame-ai)"

@'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
'@ | .\abcd\Scripts\python.exe -
