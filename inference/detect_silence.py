import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def detect_silence(audio, sr, frame_length=2048, hop_length=512, silence_threshold=0.01):
    """
    Detects silence in an audio signal.

    :param audio: The audio signal.
    :param sr: The sampling rate of the audio.
    :param frame_length: The length of each frame for analysis.
    :param hop_length: The number of samples to shift between frames.
    :param silence_threshold: The threshold for considering a frame as silent.
    :return: A list of tuples containing the start and end times of silent segments.
    """
    # Compute the short-time energy of the audio signal
    energy = librosa.feature.rms(audio, frame_length=frame_length, hop_length=hop_length)[0]

    # Normalize the energy
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))

    # Find frames that are below the silence threshold
    silent_frames = np.where(energy < silence_threshold)[0]
    silent_times = librosa.frames_to_time(silent_frames, sr=sr, hop_length=hop_length)
    print(silent_times)
    
    # Group consecutive frames and convert to time
    silent_segments = []
    for group in np.split(silent_frames, np.where(np.diff(silent_frames) != 1)[0]+1):
        start_time = librosa.frames_to_time(group[0], sr=sr, hop_length=hop_length)
        end_time = librosa.frames_to_time(group[-1], sr=sr, hop_length=hop_length)
        silent_segments.append((start_time, end_time))


    return silent_segments

# Load an audio file
audio, sr = librosa.load('/root/autodl-tmp/shijieqi/GeneFacePlusPlus/data/raw/val_wavs/mouth_close.wav', sr=None)

# Detect silence
silent_segments = detect_silence(audio, sr)
print(silent_segments)
# Plot the audio waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio, sr=sr, alpha=0.5, color="blue")
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform with Detected Silence')

# Mark the silent segments on the waveform
for start_time, end_time in silent_segments:
    plt.axvspan(start_time, end_time, color='red', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()