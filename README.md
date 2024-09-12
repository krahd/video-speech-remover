# video-speech-remover
Takes a video and tries to delete every frame where someone is speaking.


**Tomas Laurenzo**
tomas@laurenzo.net

## usage
shh.py [-h] [--buffer BUFFER] [--min-speech MIN_SPEECH] input_file output_file

* input_file: path to the input video file
* output_file: path to the output video file
* --buffer", default=0.5, buffer duration around speech segments (in seconds)
* --min-speech", default=0.5, minimum speech segment duration (in seconds)

## acknowledgments

It uses librosa (http://librosa.org) for audio processing.

## Requirements:
sys, numpy, librosa, moviepy, tqdm, argparse, subprocess, tempfile, and  webrtcvad