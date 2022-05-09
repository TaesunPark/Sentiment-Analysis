import os
import sys
import wave
import numpy as np
from datetime import datetime
from pyaudio import PyAudio, paInt16


class GenAudio(object):
    def __init__(self):
        self.num_samples = 2000  # pyaudio
        self.sampling_rate = 8000  #
        self.level = 1500  #
        self.count_num = 20  # count_num       COUNT_NUM   LEVEL
        self.save_length = 8  # ：save_length * num_samples
        self.time_count = 8  # ，  s
        self.voice_string = []

    #
    def save_wav(self, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(self.sampling_rate)
        wf.writeframes(np.array(self.voice_string).tostring())
        wf.close()

    def read_audio(self):
        pa = PyAudio()
        stream = pa.open(format=paInt16, channels=1, rate=self.sampling_rate, input=True,
                         frames_per_buffer=self.num_samples)

        save_count = 0
        save_buffer = []
        time_count = self.time_count

        while True:
            time_count -= 1

            #   num_samples
            string_audio_data = stream.read(self.num_samples)
            #
            audio_data = np.fromstring(string_audio_data, dtype=np.short)
            #     level
            large_sample_count = np.sum(audio_data > self.level)

            print(np.max(audio_data)), "large_sample_count=>", large_sample_count

            #       COUNT_NUM，     SAVE_LENGTH
            if large_sample_count > self.count_num:
                save_count = self.save_length
            else:
                save_count -= 1
            if save_count < 0:
                save_count = 0

            if save_count > 0:
                save_buffer.append(string_audio_data)
            else:
                if len(save_buffer) > 0:
                    self.voice_string = save_buffer
                    save_buffer = []
                    print("Recode a piece of  voice successfully!")
                    return True

            if time_count == 0:
                if len(save_buffer) > 0:
                    self.voice_string = save_buffer
                    save_buffer = []
                    print("Recode a piece of  voice successfully!")
                    return True
                else:
                    return False
        return True


if __name__ == "__main__":
    r = GenAudio()
    r.read_audio()
    r.save_wav("./test2.wav")