"""Shared constants for the benchmarking scripts: sample-size schedule and seed list."""

def get_num_samples(dim):
    if dim <= 2:
        num_samples = 32
    elif dim <= 4:
        num_samples = 64
    elif dim <= 8:
        num_samples = 128
    else:
        num_samples = 256
    return num_samples

seeds = [1, 10, 50, 100, 150]
# seeds = [1]