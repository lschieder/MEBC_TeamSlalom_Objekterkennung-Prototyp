import torch
# Validierung der Installation
# Prüft, ob PyTorch korrekt installiert ist und ob die NVIDIA Grafikkarte (CUDA) erkannt wird.
print(torch.__version__)
print(torch.cuda.is_available())        # True = GPU verfügbar
print(torch.cuda.get_device_name(0))    # z. B. "NVIDIA RTX 3080"
print(torch.cuda.device_count())        # Anzahl verfügbarer GPUs