from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np
from torch.utils.data import Dataset


class MelSpectroGramDataset(Dataset):

    def __init__(self, audio_dir, sample_rate=22050, context_duration=5.0, target_duration=3.0, stride_duration=3.0):

        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.hop_length = 256
        self.n_fft = 1024
        self.n_mels = 128

        self.time_step_duration = self.hop_length / self.sample_rate
        self.context_steps = int(np.ceil(context_duration / self.time_step_duration))
        self.target_steps = int(np.ceil(target_duration / self.time_step_duration))
        self.stride_steps = int(np.ceil(stride_duration / self.time_step_duration))
        self.window_steps = self.context_steps + self.target_steps

        print(f"Time step duration: {self.time_step_duration:.4f} seconds")
        print(f"Context: {self.context_steps} steps ({self.context_steps * self.time_step_duration:.2f}s)")
        print(f"Target: {self.target_steps} steps ({self.target_steps * self.time_step_duration:.2f}s)")
        print(f"Stride: {self.stride_steps} steps ({self.stride_steps * self.time_step_duration:.2f}s)")
        print(f"Total window: {self.window_steps} steps ({self.window_steps * self.time_step_duration:.2f}s)\n")

        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        self.examples = []
        self._create_examples()

    def _create_examples(self):

        audio_files = sorted(list(self.audio_dir.glob("*wav")))
        print(len(audio_files), "audio files found in", self.audio_dir)
        for audio_file in audio_files:
            try:
                waveform, native_sample_rate = torchaudio.load(str(audio_file))
                if native_sample_rate != self.sample_rate:
                    resampler = T.Resample(native_sample_rate, self.sample_rate)
                    waveform = resampler(waveform)
                
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                mel_spectrogram = self.mel_transform(waveform)
                mel_db = librosa.power_to_db(mel_spectrogram.squeeze().numpy(), ref=np.max)

                num_time_steps = mel_db.shape[1]

                num_windows = (num_time_steps - self.window_steps) // self.stride_steps + 1

                print(f"Processing {audio_file.name}: {num_time_steps} time steps, {num_windows} windows")
                for window_idx in range(num_windows):
                    start_step = window_idx * self.stride_steps
                    self.examples.append((audio_file, start_step, mel_db))

            except Exception as err:
                print(f"Error in processing {audio_file} -> {err}")
            
        print(f"total training examples created: {len(self.examples)}")
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        _, start_step, mel_db = self.examples[idx]
        context_end = start_step + self.context_steps
        target_end = context_end + self.target_steps

        context = mel_db[:, start_step:context_end]
        target = mel_db[:, context_end:target_end]

        context = torch.from_numpy(context).float()
        target = torch.from_numpy(target).float()

        return context, target
