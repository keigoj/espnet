import numpy as np
import webrtcvad
import trim_silence
import soundfile as sf

# # 積極性モードは以下のようにVADオブジェクト作成時にも指定できる。
# vad = webrtcvad.Vad(1)

# sample_rate = 16000
# frame_duration = 10
# frame_length = int(sample_rate * frame_duration / 1000)

# # VADに有音データを渡す。結果はTrue（発話を含む）になるはず
# # - サイン波を生成して確認する
# frequency = 440  # Hz（A4の音）
# t = np.linspace(
#     0,
#     frame_duration / 1000,
#     frame_length,
#     endpoint=False
# )
# import pdb; pdb.set_trace()
# amplitude = 0.5  # 振幅（-1.0〜1.0の範囲）
# waveform = amplitude * np.sin(2 * np.pi * frequency * t)
# pcm_waveform = np.int16(waveform * 32767) # 16ビットPCMに変換
# frame_speech = pcm_waveform.tobytes()

# # 与えられたフレームが有音かを判定する
# is_speech = vad.is_speech(frame_speech, sample_rate)
# print('発話を含む？: {}'.format(is_speech))

in_path = ["/home/hojo/dataset/UASpeech/UASpeech_noisereduce/audio/CF02/CF02_B1_C1_M2.wav", "/home/hojo/dataset/UASpeech/UASpeech_noisereduce/audio/F02/F02_B1_C3_M2.wav"]
out_path = ["./CF02_B1_C1_M2_trimmed.wav", "./F02_B1_C3_M2_trimmed.wav"]

for in_path, out_path in zip(in_path, out_path):
    wav, sr = trim_silence.read_wav(in_path)
    wav_trim = trim_silence.vad_trim(wav, sr)
    sf.write(out_path, wav_trim, sr)
