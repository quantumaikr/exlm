

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params / 1e9} Billion")

