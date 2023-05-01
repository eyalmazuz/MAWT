import numpy as np
import ffmpeg
import streamlink
from scipy.io import wavfile
from faster_whisper import WhisperModel
import threading

model_size = "medium"

# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

url = "https://www.youtube.com/watch?v=EheqWg4LOLA"  # Replace with your desired URL
streams = streamlink.streams(url)
stream_url = streams["best"].url

sample_rate = 16000 # sample rate of the audio
duration = 5  # duration in seconds
total_time = 0

audio_buffer = np.array([], dtype=np.float32).reshape(-1, 1)  # create an empty buffer

# Create a lock to synchronize access to the audio buffer
audio_lock = threading.Lock()

# Keep track of the previous segment to avoid printing duplicates
prev_segment = None


# Define a function to read the audio data from ffmpeg and store it in the buffer
def read_audio():
    global audio_buffer

    while True:
        # Start the ffmpeg process to read audio data
        out = (
            ffmpeg
            .input(stream_url, **{})
            .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar='16k', t=duration)
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        # Read the raw audio data from the stdout pipe
        raw_audio = out.communicate()[0]

        # Convert the raw audio data to a NumPy array
        audio_array = np.frombuffer(raw_audio, dtype=np.int16)

        # Reshape the array to match the number of audio channels and samples
        num_channels = 1
        audio_array = audio_array.reshape((-1, num_channels))

        # Acquire the lock to write to the buffer
        with audio_lock:
            audio_buffer = np.concatenate((audio_buffer, audio_array.astype(np.float32) / 32768.0))

# Define a function to transcribe the audio data using the whisper model
def transcribe_audio():
    global total_time
    global audio_buffer
    global prev_segment

    while True:
        # Wait for the audio buffer to be filled with data
        with audio_lock:
            audio_data = audio_buffer.copy()

            # Check if there is enough data for the current duration
            if len(audio_data) < ((duration) * sample_rate):
                continue

            # Extract the audio data for the current duration
            audio_array = audio_data[-(duration+1) * sample_rate:].ravel()

            # Truncate the buffer to the desired size
            max_buffer_size = sample_rate * duration * 2
            if len(audio_data) > max_buffer_size:
                audio_buffer = audio_data[-max_buffer_size:]
            else:
                audio_buffer = audio_data

        segments, info = model.transcribe(audio_array, beam_size=5, vad_filter=True, language='ja')

        for segment in segments:
            if not prev_segment or segment.text != prev_segment.text:
                print("[%.2fs -> %.2fs] %s" % (total_time+segment.start, total_time+segment.end, segment.text))
                total_time += segment.end - segment.start

            prev_segment = segment

# Create and start the threads
audio_thread = threading.Thread(target=read_audio)
transcribe_thread = threading.Thread(target=transcribe_audio)
audio_thread.start()
transcribe_thread.start()

# Wait for the threads to finish
audio_thread.join()
transcribe_thread.join()
