"""Standalone integration of the four tested voice-to-birdcall notebooks.

This file contains the complete processing code. It does not execute, import,
or convert any notebook.

Inputs:
    inputs/voice.webm
    data/raw/sample.mp3

Intermediate artifacts:
    data/processed/voice.wav
    data/processed/consonant.wav
    data/processed/voice-events.json
    data/processed/birdcall-chirp.wav
    data/processed/mix.wav

Output:
    outputs/audio.wav
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

SR = 44_100
CONSONANT_CUTOFF = 4_000
CONSONANT_ZCR_THRESHOLD = 0.1
CHIRP_DURATION = 0.25

BIRD_GAIN = 0.6
CONSONANT_GAIN = 0.25
HARMONY_INTERVALS = [0, 7, 12]
HARMONY_GAINS = [1.0, 0.35, 0.18]
HARMONY_DELAYS = [0.0, 0.14, 0.28]

PROJECT_DIR = Path(__file__).resolve().parent
INPUTS_DIR = PROJECT_DIR / "inputs"
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"

VOICE_SOURCE = INPUTS_DIR / "voice.webm"
BIRD_SOURCE = RAW_DATA_DIR / "sample.mp3"

VOICE_PATH = PROCESSED_DATA_DIR / "voice.wav"
CONSONANT_PATH = PROCESSED_DATA_DIR / "consonant.wav"
EVENTS_PATH = PROCESSED_DATA_DIR / "voice-events.json"
CHIRP_PATH = PROCESSED_DATA_DIR / "birdcall-chirp.wav"
MIX_PATH = PROCESSED_DATA_DIR / "mix.wav"
OUTPUT_PATH = OUTPUT_DIR / "audio.wav"


def ensure_pipeline_dirs() -> None:
    """Create the directories written by the integrated pipeline."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def finite_audio(audio: np.ndarray) -> np.ndarray:
    """Return float32 audio with NaN and infinity values replaced by silence."""

    audio = np.asarray(audio, dtype=np.float32)
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio while preserving silent signals."""

    audio = finite_audio(audio)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio if peak < 1e-8 else audio / peak


def filter_audio(
    audio: np.ndarray,
    cutoff: float,
    filter_type: str,
) -> np.ndarray:
    """Apply a fourth-order Butterworth high-pass or low-pass filter."""

    if len(audio) < 32:
        return audio
    b, a = butter(4, cutoff / (SR / 2), btype=filter_type)
    padlen = min(3 * max(len(a), len(b)), len(audio) - 1)
    return filtfilt(b, a, audio, padlen=padlen)


def load_normalized_audio(path: Path) -> np.ndarray:
    """Load mono audio at the pipeline sample rate and peak-normalize it."""

    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    if len(audio) == 0:
        raise ValueError(f"Audio file is empty: {path.name}")
    return normalize_audio(audio)


def load_preserved_wav(path: Path) -> np.ndarray:
    """Load a WAV without changing its stored amplitude."""

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if sample_rate != SR:
        raise ValueError(f"Unexpected sample rate for {path.name}: {sample_rate}")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return finite_audio(audio)


def convert_voice_to_wav(source: Path, destination: Path) -> None:
    """Decode the WebM voice input as mono 44.1 kHz WAV using FFmpeg."""

    if not source.exists():
        raise FileNotFoundError(f"Voice source not found: {source}")

    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        str(SR),
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("WAV conversion failed: " + result.stderr[-500:])


def extract_consonants(audio: np.ndarray) -> np.ndarray:
    """Isolate high-frequency, high-zero-crossing portions of the voice."""

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=2048,
        hop_length=512,
    )[0]
    rms = librosa.feature.rms(
        y=audio,
        frame_length=2048,
        hop_length=512,
    )[0]
    high_frequency = filter_audio(audio, CONSONANT_CUTOFF, "highpass")
    sample_axis = np.linspace(0, 1, len(audio))
    zcr_up = np.interp(sample_axis, np.linspace(0, 1, len(zcr)), zcr)
    rms_up = np.interp(sample_axis, np.linspace(0, 1, len(rms)), rms)
    threshold = np.percentile(rms_up, 65)
    mask = (zcr_up > CONSONANT_ZCR_THRESHOLD) & (rms_up > threshold)
    return high_frequency * mask * 0.6


def apply_consonant_effects(audio: np.ndarray) -> np.ndarray:
    """Add short delay and reverb taps to the extracted consonant texture."""

    if len(audio) == 0 or np.max(np.abs(audio)) < 1e-6:
        return audio

    delayed = np.zeros_like(audio)
    delay_30ms = int(0.03 * SR)
    if delay_30ms < len(audio):
        delayed[delay_30ms:] = audio[:-delay_30ms] * 0.4

    combined = audio + delayed
    reverb = np.zeros_like(combined)
    feedback = 0.3
    for delay in [0.015, 0.025]:
        samples = int(delay * SR)
        if samples < len(combined):
            reverb[samples:] += combined[:-samples] * feedback
        feedback *= 0.7
    return (combined + reverb) * 0.5


def get_voice_features(audio: np.ndarray) -> tuple[list[dict], float]:
    """Detect onset events and estimate their amplitude and pitch."""

    onset_frames = librosa.onset.onset_detect(
        y=audio,
        sr=SR,
        units="frames",
        backtrack=True,
        delta=0.08,
        wait=2,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=SR)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=SR,
        hop_length=512,
    )
    f0, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C3"),
        fmax=librosa.note_to_hz("C7"),
        sr=SR,
    )
    pitch_times = librosa.frames_to_time(np.arange(len(f0)), sr=SR)
    valid = ~np.isnan(f0)
    valid_pitch = f0[valid]
    median_pitch = float(np.median(valid_pitch)) if len(valid_pitch) else 440.0

    events = []
    for time in onset_times:
        amplitude = float(np.interp(time, rms_times, rms))
        pitch = float(
            np.interp(
                time,
                pitch_times[valid] if np.any(valid) else [0],
                f0[valid] if np.any(valid) else [median_pitch],
            )
        )
        events.append({"time": float(time), "amp": amplitude, "pitch": pitch})
    return events, median_pitch


def stage_01_preprocess_voice_and_analyze() -> None:
    """Run notebook 01: voice preprocessing and feature analysis."""

    convert_voice_to_wav(VOICE_SOURCE, VOICE_PATH)
    voice = load_normalized_audio(VOICE_PATH)

    consonant = apply_consonant_effects(extract_consonants(voice))
    sf.write(str(CONSONANT_PATH), finite_audio(consonant), SR, subtype="PCM_16")

    events, median_pitch = get_voice_features(voice)
    if not events:
        duration = len(voice) / SR
        events = [
            {"time": float(time), "amp": 0.7, "pitch": median_pitch}
            for time in np.arange(0.15, max(0.2, duration), 0.35)
        ]

    analysis = {
        "sample_rate": SR,
        "median_pitch_hz": median_pitch,
        "events": events,
    }
    EVENTS_PATH.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print(f"  Voice duration: {len(voice) / SR:.2f} seconds")
    print(f"  Median pitch: {median_pitch:.1f} Hz")
    print(f"  Detected voice events: {len(events)}")
    print(f"  Saved: {VOICE_PATH}")
    print(f"  Saved: {CONSONANT_PATH}")
    print(f"  Saved: {EVENTS_PATH}")


def make_chirp_from_bird(
    bird_audio: np.ndarray,
    duration: float = CHIRP_DURATION,
) -> np.ndarray:
    """Extract and fade the highest-energy chirp-sized birdcall segment."""

    length = int(duration * SR)
    if len(bird_audio) < length:
        bird_audio = np.pad(bird_audio, (0, length - len(bird_audio)))

    rms = librosa.feature.rms(
        y=bird_audio,
        frame_length=2048,
        hop_length=512,
    )[0]
    max_frame = int(np.argmax(rms))
    start = max(0, max_frame * 512 - length // 4)
    end = start + length
    if end > len(bird_audio):
        start, end = max(0, len(bird_audio) - length), len(bird_audio)

    chirp = bird_audio[start:end].copy()
    if len(chirp) < length:
        chirp = np.pad(chirp, (0, length - len(chirp)))

    fade_length = min(int(0.02 * SR), len(chirp) // 2)
    if fade_length > 0:
        chirp[:fade_length] *= np.linspace(0, 1, fade_length)
        chirp[-fade_length:] *= np.linspace(1, 0, fade_length)
    return normalize_audio(chirp)


def stage_02_extract_chirp() -> None:
    """Run notebook 02: extract a base chirp from the birdcall sample."""

    if not BIRD_SOURCE.exists():
        raise FileNotFoundError(f"Bird source not found: {BIRD_SOURCE}")

    bird = load_normalized_audio(BIRD_SOURCE)
    bird = normalize_audio(filter_audio(bird, 1_000, "highpass"))
    chirp = make_chirp_from_bird(bird)
    sf.write(str(CHIRP_PATH), chirp, SR, subtype="PCM_16")
    print(f"  Chirp duration: {len(chirp) / SR:.2f} seconds")
    print(f"  Saved: {CHIRP_PATH}")


def pitch_shift_chirp(
    chirp: np.ndarray,
    voice_pitch: float,
    base_pitch: float,
) -> np.ndarray:
    """Shift a chirp toward a detected voice pitch within a safe range."""

    if voice_pitch <= 0 or base_pitch <= 0:
        return chirp
    semitones = float(np.clip(12 * np.log2(voice_pitch / base_pitch), -6, 8))
    return librosa.effects.pitch_shift(y=chirp, sr=SR, n_steps=semitones)


def make_harmony_chirp(
    chirp: np.ndarray,
    voice_pitch: float,
    base_pitch: float,
) -> np.ndarray:
    """Layer delayed unison, fifth, and octave versions of one chirp."""

    length = len(chirp) + int(max(HARMONY_DELAYS) * SR)
    harmony = np.zeros(length, dtype=np.float32)
    for interval, gain, delay in zip(
        HARMONY_INTERVALS,
        HARMONY_GAINS,
        HARMONY_DELAYS,
    ):
        shifted_pitch = voice_pitch * (2 ** (interval / 12))
        shifted = pitch_shift_chirp(chirp, shifted_pitch, base_pitch)
        start = int(delay * SR)
        harmony[start : start + len(shifted)] += shifted * gain
    return normalize_audio(harmony)


def stage_03_mix_chirp_and_pattern() -> None:
    """Run notebook 03: arrange chirps using the analyzed voice pattern."""

    required_paths = [VOICE_PATH, CONSONANT_PATH, EVENTS_PATH, CHIRP_PATH]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    voice = load_normalized_audio(VOICE_PATH)
    consonant = load_preserved_wav(CONSONANT_PATH)
    base_chirp = load_normalized_audio(CHIRP_PATH)
    analysis = json.loads(EVENTS_PATH.read_text(encoding="utf-8"))
    if analysis["sample_rate"] != SR:
        raise ValueError(f"Unexpected event sample rate: {analysis['sample_rate']}")

    events = analysis["events"]
    if not events:
        raise ValueError("No voice events found in voice-events.json.")
    median_pitch = float(analysis["median_pitch_hz"])

    mix = np.zeros(int((len(voice) / SR + 2.5) * SR), dtype=np.float32)
    max_amplitude = max(max(event["amp"] for event in events), 1e-8)

    for event in events:
        chirp = make_harmony_chirp(base_chirp, event["pitch"], median_pitch)
        chirp *= 0.35 + 0.65 * (event["amp"] / max_amplitude)
        start = int(event["time"] * SR)
        end = start + len(chirp)
        if end > len(mix):
            continue

        if end <= len(consonant):
            voice_consonant = consonant[start:end]
        else:
            voice_consonant = np.zeros_like(chirp)

        event_sound = np.clip(
            chirp * BIRD_GAIN + voice_consonant * CONSONANT_GAIN,
            -1.0,
            1.0,
        )
        mix[start:end] += event_sound

    sf.write(str(MIX_PATH), finite_audio(mix), SR, subtype="FLOAT")
    print(f"  Arranged voice events: {len(events)}")
    print(f"  Saved: {MIX_PATH}")


def dreamy_echo(audio: np.ndarray) -> np.ndarray:
    """Apply three echo taps and soften the result with a low-pass filter."""

    result = audio.copy()
    for delay, gain in zip([0.18, 0.34, 0.52], [0.28, 0.18, 0.10]):
        samples = int(delay * SR)
        if samples < len(audio):
            result[samples:] += audio[:-samples] * gain
    return normalize_audio(filter_audio(result, 5_200, "lowpass"))


def stage_04_apply_post_effects() -> None:
    """Run notebook 04: apply post effects and write the final WAV."""

    if not MIX_PATH.exists():
        raise FileNotFoundError(f"Event mix not found: {MIX_PATH}")

    mix = load_preserved_wav(MIX_PATH)
    output = mix.copy()
    delay_90ms = int(0.09 * SR)
    if delay_90ms < len(output):
        output[delay_90ms:] += output[:-delay_90ms] * 0.25

    output = normalize_audio(dreamy_echo(output)) * 0.85
    peak = float(np.max(np.abs(output))) if len(output) else 0.0
    if peak > 1e-8:
        silence_threshold = peak * (10 ** (-50 / 20))
        audible_samples = np.flatnonzero(np.abs(output) > silence_threshold)
        if len(audible_samples):
            tail_padding = int(0.08 * SR)
            final_sample = min(
                len(output),
                int(audible_samples[-1]) + 1 + tail_padding,
            )
            output = output[:final_sample]

    fade_samples = min(int(0.06 * SR), len(output) // 2)
    if fade_samples > 0:
        output[-fade_samples:] *= np.linspace(
            1.0,
            0.0,
            fade_samples,
            dtype=np.float32,
        )

    sf.write(str(OUTPUT_PATH), finite_audio(output), SR, subtype="PCM_16")
    print(f"  Output duration: {len(output) / SR:.2f} seconds")
    print(f"  Saved: {OUTPUT_PATH}")


def run_pipeline() -> Path:
    """Execute the complete integrated 01-04 pipeline."""

    ensure_pipeline_dirs()

    print("[1/4] Preprocessing voice and analyzing features")
    stage_01_preprocess_voice_and_analyze()

    print("[2/4] Extracting birdcall chirp")
    stage_02_extract_chirp()

    print("[3/4] Mixing chirps and consonant texture")
    stage_03_mix_chirp_and_pattern()

    print("[4/4] Applying post effects")
    stage_04_apply_post_effects()

    print(f"Completed: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    run_pipeline()
