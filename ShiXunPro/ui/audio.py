import pyaudio
import wave

class Audio():
    input_filename = "test.wav"  # 麦克风采集的语音输入
    input_filepath = "../wav_file/"  # 输入文件的path
    in_path = input_filepath + input_filename

    def get_audio(self):
        CHUNK = 256
        FORMAT = pyaudio.paInt16
        CHANNELS = 1  # 声道数
        RATE = 16000  # 采样率
        RECORD_SECONDS = 10
        WAVE_OUTPUT_FILENAME = self.in_path
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("*" * 10, "开始录音：请在10秒内输入语音")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("*" * 10, "录音结束\n")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()