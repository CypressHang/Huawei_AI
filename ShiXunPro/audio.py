import threading
import time

import pyaudio
import wave

input_filename = "input.wav"  # 麦克风采集的语音输入
input_filepath = "wav_file/"  # 输入文件的path
in_path = input_filepath + input_filename
CHUNK = 256
FORMAT = pyaudio.paInt16
CHANNELS = 1  # 声道数
RATE = 16000  # 采样率
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = in_path

allowRecording = False

def get_audio():

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    print("*" * 10, "开始录音")

    while allowRecording:
        data = stream.read(CHUNK)
        wf.writeframes(data)
    wf.close()
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("*" * 10, "结束录音")



def start():
    global allowRecording
    allowRecording = True
    threading.Thread(target=get_audio).start()

def stop():
    global allowRecording
    allowRecording = False
    return in_path

