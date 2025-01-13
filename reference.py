'''        
data = {
    "text":          text,                        # list
    "fname":         self.text_to_filename(text)  # list
    "label_vector":  label_vector.float(),        # tensor, [B, class_num]
    "waveform":      waveform.float(),            # tensor, [B, 1, samples_num] = [1, 1, 163840] (=10.24*16000)
    "stft":          stft.float(),                # tensor, [B, t-steps, f-bins]
    "log_mel_spec":  log_mel_spec.float(),        # tensor, [B, t-steps, mel-bins] = [1, 1024, 64]
    "duration":      self.duration,
    "sampling_rate": self.sampling_rate,
    "random_start_sample_in_original_audio_file": random_start,}
'''