# voice-to-birdcall

Python package + notebook workspace for transformation of voice to birdcall experiments.

## Project Layout

```text
voice-to-birdcall/
|-- models/
|-- data/
|   |-- raw/
|   `-- processed/
|-- external/
|-- inputs/
|-- notebooks/
|   |-- 01_preprocess_voice.ipynb
|   |-- 02_extract_chirp.ipynb
|   |-- 03_mix_chirp_and_pattern.ipynb
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

Run notebooks `01` through `04` in order.
Notebook `01` reads `inputs/voice.wav` directly with SoundFile, averages channels
to mono, and resamples to 44.1 kHz only when necessary. It writes the prepared
PCM16 audio to `data/processed/voice.wav` without modifying the input. Voice
preprocessing does not require FFmpeg.
Notebook `01` filters background-like events while retaining clear tonal sounds,
and records a confidence-gated recording pitch and `shift_semitones` in
`voice-events.json`. Notebook `03` lowers each completed harmony chirp by that
shift before mixing the original consonant texture. Unknown pitches and pitches
at or above 220 Hz use -1 semitone; lower pitches move down continuously to a
maximum additional shift of -12 semitones. Bird sample extraction (`02`) and
audio post-processing (`04`) retain their existing algorithms. If all events are
rejected, the result is silence with the input duration plus 2.5 seconds.

Run `01` before `03` to generate the required event and pitch shift metadata.
Event analysis functions are defined in notebook `01`; pitch shift functions are
defined in notebook `03`.
The standalone `voice_to_birdcall_pipeline.py` also reads `inputs/voice.wav`,
but its synthesis does not include the event filtering and recording-wide pitch
shift used by the notebooks.

Do not commit pretrained model files to this repository. `models/.gitkeep` is
tracked only to preserve the directory; model weight files such as `.pth`, `.pt`,
`.ckpt`, `.onnx`, and `.safetensors` are ignored by git.

Use `external/` for local third-party libraries, SDKs, or source checkouts that need
to be referenced from a local path. The directory itself is versioned, while its
contents are ignored by git by default.
