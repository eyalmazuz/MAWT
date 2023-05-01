import multiprocessing as mp
import os
import threading
import time

import ctranslate2
from faster_whisper import WhisperModel
import numpy as np
import ffmpeg
import transformers


from consts import SAMPLE_RATE, TOTAL_TIME, NUM_CHANNELS

# Define a function to transcribe the audio data using the whisper model
def transcribe_audio(audio_lock: threading.Lock,
                     duration: int,
                     language: str,
                     model_size: str,
                     mp_buffer: mp.Array):

    global TOTAL_TIME

    # load whisper model onto CPU in int8 format
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    translator = ctranslate2.Translator("../models/nllb-200-distilled-600M", device="cpu", compute_type="int8")
    tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang='jpn_Jpan')

    prev_segment = None

    while True:
        np_buffer = np.frombuffer(mp_buffer.get_obj(), dtype=np.int16).reshape(duration * SAMPLE_RATE, 1)

        # Wait for the audio buffer to be filled with data
        with audio_lock:

            audio_array = np_buffer.copy()

            # print(f'{audio_data.shape=}')
            # # Check if there is enough data for the current duration
            # if len(audio_data) < ((duration) * SAMPLE_RATE):
            #     continue

            # Extract the audio data for the current duration
            # audio_array = audio_data[-(duration+1) * SAMPLE_RATE:].ravel()

            # Truncate the buffer to the desired size
            # max_buffer_size = SAMPLE_RATE * duration * 2
            # if len(audio_data) > max_buffer_size:
            #     audio_buffer = audio_data[-max_buffer_size:]
            # else:
                # audio_buffer = audio_data

        # print(f'{(audio_array.astype(np.float32) / 32768.0)=}')
        segments, info = model.transcribe(audio_array.astype(np.float32).ravel() / 32768.0, beam_size=5, vad_filter=True, language=language)

        for segment in segments:

            sentence_tokenized = [tokenizer.convert_ids_to_tokens(tokenizer.encode(segment.text))]
            translations = translator.translate_batch(sentence_tokenized, target_prefix=[['eng_Latn']])

            translated_sentence = tokenizer.decode(tokenizer.convert_tokens_to_ids(translations[0].hypotheses[0][1:]))

            if not prev_segment or segment.text != prev_segment.text:
                print("[%.2fs -> %.2fs] Original: %s" % (TOTAL_TIME+segment.start, TOTAL_TIME+segment.end, segment.text))
                print("[%.2fs -> %.2fs] Translated: %s" % (TOTAL_TIME+segment.start, TOTAL_TIME+segment.end, translated_sentence))


                TOTAL_TIME += segment.end - segment.start

            prev_segment = segment

# model_size = "medium"
#
# # or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4, num_workers=2)
#
# url = "https://www.youtube.com/watch?v=EheqWg4LOLA"  # Replace with your desired URL
# streams = streamlink.streams(url)
# stream_url = streams["best"].url
#
# duration = 5  # duration in seconds
# total_time = 0
#
# while True:  # total_duration is the duration of the input file in seconds
#
#     out = (
#         ffmpeg
#         .input(stream_url, **{})
#         .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar='16k', t=duration)
#         .run_async(pipe_stdout=True, pipe_stderr=True)
#     )
#
#     # Read the raw audio data from the stdout pipe
#     raw_audio = out.communicate()[0]
#
#     # Convert the raw audio data to a NumPy array
#     audio_array = np.frombuffer(raw_audio, dtype=np.int16)
#
#     # Reshape the array to match the number of audio channels and samples
#     num_channels = 1
#     sample_rate = 16000
#     audio_array = audio_array.reshape((-1, num_channels))
#
#     # Save the audio array to a WAV file
#     out_filename = f'output.wav'
#     wavfile.write(out_filename, sample_rate, audio_array)
#
#     segments, info = model.transcribe(audio_array.ravel().astype(np.float32) / 32768.0, beam_size=5, vad_filter=True, language='ja')
#
#     for segment in segments:
#         print("[%.2fs -> %.2fs] %s" % (total_time+segment.start, total_time+segment.end, segment.text))
#         total_time += segment.end - segment.start
