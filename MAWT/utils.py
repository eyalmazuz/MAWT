import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-u', '--url', type=str, required=True, help='URL to capture stream from')
    parser.add_argument('-d', '--duration', type=int, default=5, required=False, help='duration of audio to parse by the model')
    parser.add_argument('-l', '--langauge', type=str, default='ja', required=False, help='which langauge to transcribe from')
    parser.add_argument('-m', '--model', type=str, default='small', choices=['tiny', 'base', 'small', 'medium', 'large'],
                        required=False, help='which langauge to transcribe from')


    return parser.parse_args()
