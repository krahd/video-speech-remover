#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import librosa
import moviepy.editor as mp
from tqdm import tqdm
import argparse
import subprocess
import tempfile
import webrtcvad
import struct

# Ensure we're using the Python interpreter from the conda environment
if 'CONDA_PREFIX' in os.environ:
    python_path = os.path.join(os.environ['CONDA_PREFIX'], 'bin', 'python')
    if sys.executable != python_path:
        os.execl(python_path, python_path, *sys.argv)

def detect_speech(audio_file, aggressiveness=3, frame_duration=30, buffer_duration=0.5, min_speech_duration=0.5):
    vad = webrtcvad.Vad(aggressiveness)
    
    audio, sample_rate = librosa.load(audio_file, sr=16000, mono=True)
    frame_length = int(sample_rate * (frame_duration / 1000.0))
    num_frames = len(audio) // frame_length
    
    speech_frames = []
    for i in range(num_frames):
        frame = audio[i*frame_length:(i+1)*frame_length]
        frame_bytes = struct.pack("%dh" % len(frame), *np.round(frame * 32767).astype(np.int16))
        is_speech = vad.is_speech(frame_bytes, sample_rate)
        speech_frames.append(is_speech)
    
    # Convert frame-level decisions to time segments with buffer
    speech_segments = []
    in_speech = False
    start_time = 0
    buffer_frames = int(buffer_duration * 1000 / frame_duration)
    
    for i, is_speech in enumerate(speech_frames):
        if is_speech and not in_speech:
            start_time = max(0, (i - buffer_frames) * frame_duration / 1000.0)
            in_speech = True
        elif not is_speech and in_speech:
            end_time = min(len(speech_frames), (i + buffer_frames)) * frame_duration / 1000.0
            if end_time - start_time >= min_speech_duration:
                speech_segments.append((start_time, end_time))
            in_speech = False
    
    if in_speech:
        end_time = len(speech_frames) * frame_duration / 1000.0
        if end_time - start_time >= min_speech_duration:
            speech_segments.append((start_time, end_time))
    
    # Merge overlapping segments
    merged_segments = []
    for segment in speech_segments:
        if not merged_segments or segment[0] > merged_segments[-1][1]:
            merged_segments.append(segment)
        else:
            merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], segment[1]))
    
    print(f"Detected {len(merged_segments)} speech segments")
    for i, (start, end) in enumerate(merged_segments):
        print(f"Speech segment {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    
    return merged_segments

def apply_red_tint(frame):
    return np.clip(frame + [50, 0, 0], 0, 255).astype('uint8')

def get_video_duration(input_file):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout)

def split_video(input_file, chunk_duration, temp_dir):
    total_duration = get_video_duration(input_file)
    chunks = []
    for start_time in np.arange(0, total_duration, chunk_duration):
        end_time = min(start_time + chunk_duration, total_duration)
        chunk_file = os.path.join(temp_dir, f'chunk_{start_time:.0f}_{end_time:.0f}.mp4')
        cmd = [
            'ffmpeg', '-i', input_file, '-ss', f'{start_time:.3f}', '-to', f'{end_time:.3f}',
            '-c', 'copy', '-avoid_negative_ts', 'make_zero', '-y', chunk_file
        ]
        subprocess.run(cmd, check=True)
        chunks.append(chunk_file)
    return chunks

def process_chunk(chunk_file, speech_segments, chunk_start, chunk_duration, temp_dir):
    chunk_end = chunk_start + chunk_duration
    chunk_non_speech = []
    last_end = 0
    for start, end in speech_segments:
        if start > chunk_start and start < chunk_end:
            if start - chunk_start > last_end:
                chunk_non_speech.append((last_end, start - chunk_start))
        if end > chunk_start and end < chunk_end:
            last_end = end - chunk_start
    if chunk_duration > last_end:
        chunk_non_speech.append((last_end, chunk_duration))

    processed_chunk = os.path.join(temp_dir, f'processed_{os.path.basename(chunk_file)}')
    
    if chunk_non_speech:
        filter_complex = []
        for i, (start, end) in enumerate(chunk_non_speech):
            filter_complex.append(f"[0:v]trim={start:.3f}:{end:.3f},setpts=PTS-STARTPTS[v{i}];")
            filter_complex.append(f"[0:a]atrim={start:.3f}:{end:.3f},asetpts=PTS-STARTPTS[a{i}];")
        
        all_segments = [f'[v{i}][a{i}]' for i in range(len(chunk_non_speech))]
        filter_complex.append(f"{''.join(all_segments)}concat=n={len(chunk_non_speech)}:v=1:a=1[outv][outa]")
        filter_complex = ''.join(filter_complex)
        
        cmd = ['ffmpeg', '-i', chunk_file, '-filter_complex', filter_complex,
               '-map', '[outv]', '-map', '[outa]', '-y', processed_chunk]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return processed_chunk
        except subprocess.CalledProcessError as e:
            print(f"Error processing chunk: {e}")
            print(f"FFmpeg error output: {e.stderr}")
            return chunk_file
    else:
        return chunk_file

def concatenate_videos(video_files, output_file):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        for video in video_files:
            temp_file.write(f"file '{video}'\n")
    
    cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', temp_file.name, '-c', 'copy', '-y', output_file]
    subprocess.run(cmd, check=True)
    os.unlink(temp_file.name)

def process_video_in_chunks(input_file, output_file, chunk_duration=300, buffer_duration=0.5, min_speech_duration=0.5):
    temp_dir = tempfile.mkdtemp()
    try:
        # Extract audio and detect speech
        audio_file = os.path.join(temp_dir, 'audio.wav')
        subprocess.run(['ffmpeg', '-i', input_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_file], check=True)
        speech_segments = detect_speech(audio_file, aggressiveness=3, buffer_duration=buffer_duration, min_speech_duration=min_speech_duration)
        
        # Split video into chunks
        chunks = split_video(input_file, chunk_duration, temp_dir)
        
        # Process each chunk
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_start = i * chunk_duration
            processed_chunk = process_chunk(chunk, speech_segments, chunk_start, chunk_duration, temp_dir)
            processed_chunks.append(processed_chunk)
        
        # Concatenate processed chunks
        concatenate_videos(processed_chunks, output_file)
        
        print(f"Processed video saved as: {output_file}")
    
    finally:
        # Clean up temporary files
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)

def main():
    parser = argparse.ArgumentParser(description="Process a video file to detect speech and remove speech segments.")
    parser.add_argument("input_file", help="Path to the input video file")
    parser.add_argument("output_file", help="Path to the output video file")
    parser.add_argument("--buffer", type=float, default=0.5, help="Buffer duration around speech segments (in seconds)")
    parser.add_argument("--min-speech", type=float, default=0.5, help="Minimum speech segment duration (in seconds)")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)

    print(f"Processing file: {args.input_file}")
    process_video_in_chunks(args.input_file, args.output_file, buffer_duration=args.buffer, min_speech_duration=args.min_speech)

if __name__ == "__main__":
    main()