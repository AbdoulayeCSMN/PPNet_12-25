import torch

ckpt = torch.load("20_9push0.7337.pth", map_location="cpu")
print(type(ckpt))

print(f'{60*"-"}\nfeatures:\n{ckpt.features}')
print(f'{60*"-"}\nadd_on_layers:\n{ckpt.add_on_layers}')
print(f'{60*"-"}\nlast_layer:\n{ckpt.last_layer}')