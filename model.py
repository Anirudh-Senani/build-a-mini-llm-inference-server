"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
def stable_softmax(logits):
    # TODO: compute a numerically stable softmax over the last axis of logits.
    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return shifted/(shifted.sum(axis=-1, keepdims=True))

# Step 2 - apply_temperature
def apply_temperature(logits, temperature):
    # TODO: scale logits by 1 / temperature; if temperature <= 0, return logits unchanged (greedy).
    if temperature <= 0:
        temperature = 1.0
    
    return logits/temperature

