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

# Step 3 - top_k_filter
import numpy as np

def top_k_filter(logits, k):
    """Mask logits outside the top-k per row to -inf."""
    # TODO: keep only the k largest logits along the last axis, set the rest to -inf
    flag = False
    if logits.ndim == 1:
        logits = logits[None, :]
        flag = True

    if k == 0:
        k = logits.shape[1]

    k = min(k, logits.shape[1])

    top_k_ind = np.argsort(-logits, axis=-1)[:, k-1]
    top_k_val = logits[np.arange(logits.shape[0]), top_k_ind]

    logits[logits < top_k_val[:, None]] = -np.inf

    if flag:
        logits = logits[0]
    return logits

# Step 4 - top_p_filter
def top_p_filter(logits, p):
    # TODO: keep smallest set of tokens whose cumulative prob >= p, mask the rest to -inf.
    flag = False
    if logits.ndim == 1:
        logits = logits[None, :]
        flag = True

    probs = stable_softmax(logits)
    inds = np.argsort(-probs, axis=-1)
    cum_prob = np.cumsum(probs[np.arange(logits.shape[0])[:,None],inds], axis=-1)
    top_p_ind = np.minimum((cum_prob <= p).sum(axis=-1, keepdims=True), logits.shape[1]-1)

    mask = np.arange(logits.shape[1]) <= top_p_ind
    mask = mask[np.arange(logits.shape[0])[:, None], np.argsort(inds, axis=-1)]
    logits[~mask] = -np.inf

    if flag:
        logits = logits[0]
    return logits

# Step 5 - sample_from_probs
def sample_from_probs(probs, rng):
    # TODO: draw a single token id from the categorical distribution probs using rng
    return int(rng.choice(probs.shape[0], p=probs))

# Step 6 - greedy_select
def greedy_select(logits):
    # TODO: return the index of the maximum logit (ties -> lowest index).
    return int(np.argmax(logits))

