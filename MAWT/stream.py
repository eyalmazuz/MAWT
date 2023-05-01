import multiprocessing as mp
import os
import threading
import time

import ffmpeg
import numpy as np
import streamlink
from scipy.io import wavfile

from consts import *

# Define a function to read the audio data from ffmpeg and store it in the buffer
def read_audio(audio_lock: threading.Lock,
               url: str,
               duration: int,
               mp_buffer: mp.Array):

    global FILE_NUM

    np_buffer = np.frombuffer(mp_buffer.get_obj(), dtype=np.int16).reshape(duration * SAMPLE_RATE, 1)

    last_buffer_size = 0

    # url = "https://www.youtube.com/watch?v=EheqWg4LOLA"  # Replace with your desired URL
    streams = streamlink.streams(url)
    stream_url = streams["best"].url

    while True:

        start_time = time.time()
        # Start the ffmpeg process to read audio data
        out = (
        ffmpeg
        # capture from stream
        .input(stream_url, **{})
        # pipe the input to stdout in audio only format using $NUM_CHANNELS
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=NUM_CHANNELS, ar=str(SAMPLE_RATE), t=str(duration))
        # make reading from the stream async by itself
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

        # Read the raw audio data from the stdout pipe
        raw_audio = out.communicate()[0]
        # raw_audio = out.stdout.read(NUM_CHANNELS * SAMPLE_RATE * duration * 2)
        # if not raw_audio:
        #     break

        # Convert the raw audio data to a NumPy array
        audio_array = np.frombuffer(raw_audio, dtype=np.int16).reshape(-1, NUM_CHANNELS)

        # locking the buffer to make sure no read/writes are done while writing new data
        with audio_lock:
            # If the audio buffer is not empty, find the longest matching sublist between the buffer and the new data
            # if audio_buffer.size > 0:
            #     for i in range(audio_array.size, 0, -1):
            #         if np.array_equal(audio_buffer[-i:], audio_array[:i]):
            #             # print('something equals', i)
            #             # If there is a matching sublist, discard the matching portion of the new data
            #             audio_array = audio_array[i:]
            #             break
            #
            # # Append the new data to the buffer
            # audio_buffer = np.concatenate((audio_buffer, audio_array))

            np_buffer[:] = audio_array[:]

        end_time = time.time()

        # # for debugging purposes
        # # if np_buffer.size != last_buffer_size:
        # # Save the audio array to a WAV file
        # out_filename = f'output_{FILE_NUM}.wav'
        #
        # wavfile.write(out_filename, SAMPLE_RATE, np_buffer)
        # # print(f'saving {out_filename} buffer size {np_buffer.shape=}', "Execution time:", end_time - start_time, "seconds")
        # last_buffer_size = np_buffer.size
        # FILE_NUM += 1
