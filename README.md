# voice-to-birdcall

Python package + notebook workspace for transformation of voice to birdcall experiments.

## Project Layout

```text
voice-to-birdcall/
|-- data/
|   |-- models/
|   |-- raw/
|   `-- processed/
|-- external/
|-- inputs/
|-- notebooks/
|   |-- 01_preprocess_voice.ipynb
|   |-- 02_extract_chirp.ipynb
|   |-- 03_mix.ipynb
|   |-- 04_apply_post_effects.ipynb
|   `-- setup.ipynb
|-- outputs/
|-- src/
|   `-- voice_to_birdcall/
|       |-- __init__.py
|       |-- config.py
|       `-- paths.py
|-- tests/
|   `-- test_paths.py
|-- pyproject.toml
`-- requirements.txt
```

## Setup

This project is written and tested with Python 3.12.10.

Create a virtual environment, install the package in editable mode, and register
the project notebook kernel.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name python3 --display-name ".venv"
```

### FFmpeg

Audio conversion requires FFmpeg. The Python dependencies include
`imageio-ffmpeg`, which provides a bundled FFmpeg executable. The preprocessing
notebook uses a system `ffmpeg` command when it is available on `PATH` and falls
back to the bundled executable otherwise.

Verify the executable selected by the fallback package:

```powershell
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```

VS Code can open and run `.ipynb` notebooks directly after this setup. To use
the browser-based JupyterLab UI instead, start it separately:

```powershell
jupyter lab
```

## Notebook Usage

After `python -m pip install -e .`, notebooks can import project code without modifying
`sys.path`:

```python
from voice_to_birdcall import paths

paths.DATA_DIR
```

Use `inputs/` for source voice recordings, `data/raw/` for birdcall samples,
`data/processed/` for intermediate audio and feature data, and `outputs/` for
final generated audio. These folders contain `.gitkeep` files so the layout is
versioned, while generated contents can remain ignored by git.

Do not commit pretrained model files to this repository. `data/models/.gitkeep` is
tracked only to preserve the directory; model weight files such as `.pth`, `.pt`,
`.ckpt`, `.onnx`, and `.safetensors` are ignored by git.

Use `external/` for local third-party libraries, SDKs, or source checkouts that need
to be referenced from a local path. The directory itself is versioned, while its
contents are ignored by git by default.
