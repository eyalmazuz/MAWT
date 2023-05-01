import multiprocessing as mp
import os
import threading
import time

import ffmpeg
import numpy as np
import streamlink
from scipy.io import wavfile


from consts import *
from stream import read_audio
from transcripts import transcribe_audio
from utils import parse_args

audio_buffer = np.array([], dtype=np.int16).reshape(-1, NUM_CHANNELS)  # create an empty buffer



# Create a lock to synchronize access to the audio buffer
audio_lock = threading.Lock()


def main():
    args = parse_args()

    mp_buffer = mp.Array('h', SAMPLE_RATE * args.duration)

    # Create stream capture thread
    audio_thread = threading.Thread(target=read_audio, args=(audio_lock, args.url,
                                                             args.duration, mp_buffer))

    # Create transribe thread
    transcription_thread = threading.Thread(target=transcribe_audio, args=(audio_lock, args.duration,
                                                                           args.langauge, args.model,
                                                                           mp_buffer))
    # start both threads
    audio_thread.start()
    transcription_thread.start()

    # Wait for the threads to finish
    audio_thread.join()
    transcription_thread.join()

if __name__ == "__main__":
    main()
