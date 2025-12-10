from pathlib import Path
import webrtcvad
import soundfile as sf
import numpy as np
import tqdm
import argparse
import pdb

def read_wav(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1:  # if multi-channel, take the first channel
        wav = wav[:, 0]
    # webrtc is only compatible with 16/8/32/48 kHz mono 16-bit PCM
    if sr != 16000:
        raise ValueError("Sampling rate must be 16 kHz.")
    return wav, sr

def frame_generator(wav, sr, frame_ms=30):
    frame_len = int(sr * frame_ms / 1000)
    if len(wav) == 0:
        return
    num_frames = (len(wav) + frame_len - 1) // frame_len  # ceil division
    for i in range(num_frames):
        start = i * frame_len
        end = start + frame_len
        frame = wav[start:end]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        yield frame, start, min(end, len(wav))

def vad_trim(wav, sr, frame_ms=20, aggressiveness=3, hangover_frames=1, min_voiced_run=10):
    if frame_ms not in (10, 20, 30):
        raise ValueError("frame_ms must be one of 10, 20, or 30 ms for webrtcvad.")
    frame_len = int(sr * frame_ms / 1000)
    vad = webrtcvad.Vad(aggressiveness)  # 0-3 
    frames = list(frame_generator(wav, sr, frame_ms))
    if len(frames) == 0:
        return wav
    voiced = []

    for f, s, e in frames:
        f16 = (f * 32768).astype(np.int16).tobytes()
        is_speech = vad.is_speech(f16, sr)
        voiced.append(is_speech)

    # search first and last voiced frame to trim leading and trailing silence
    if not any(voiced):  # if all frames are unvoiced, original was is returned
        return wav

    # find first voiced run long enough to treat as real speech
    run = 0
    start_voice = None
    for i, v in enumerate(voiced):
        run = run + 1 if v else 0
        if run >= min_voiced_run:
            start_voice = i - (min_voiced_run - 1)
            break
    if start_voice is None:  # voicedが散発的で連続しない場合は全無音扱い
        return wav

    # add hangover frames to give margin
    idx_speech = [i for i, v in enumerate(voiced) if v]
    start_i = max(start_voice - hangover_frames, 0)
    end_i = min(idx_speech[-1] + hangover_frames, len(frames) - 1)

    start_sample = int(start_i * frame_len)
    end_sample = min(len(wav), int((end_i + 1) * frame_len))

    return wav[start_sample:end_sample]

def main():
    parser = argparse.ArgumentParser(description="Trim silence from UASpeech audio using VAD")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input UASpeech wav directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output trimmed wav directory")
    args = parser.parse_args()
    
    root = Path(args.input_dir)  # original wav directory
    out_root = Path(args.output_dir)  # trimmed wav directory
    out_root.mkdir(parents=True, exist_ok=True)

    for wav_path in tqdm.tqdm(sorted(root.rglob("*.wav"))):
        if wav_path.name.startswith("._"):
            continue  # skip system files
        
        rel = wav_path.relative_to(root)
        out_path = out_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wav, sr = read_wav(str(wav_path))
        wav_trim = vad_trim(wav, sr)
        sf.write(str(out_path), wav_trim, sr)

if __name__ == "__main__":
    main()
