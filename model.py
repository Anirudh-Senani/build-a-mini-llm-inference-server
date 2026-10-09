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

# Step 7 - build_vocab
def build_vocab(corpus, special_tokens):
    # TODO: build a character-level vocab; specials get the lowest ids, then sorted unique chars.
    token_to_id = {}
    id_to_token = []
    cur_id = 0
    for tok in special_tokens:
        token_to_id[tok] = cur_id
        cur_id += 1
        id_to_token.append(tok)

    vocab = set()
    for word in corpus:
        for ch in word:
            vocab.add(ch)

    for tok in sorted(vocab):
        token_to_id[tok] = cur_id
        cur_id += 1
        id_to_token.append(tok)

    return dict(
        token_to_id=token_to_id,
        id_to_token=id_to_token
    )

# Step 8 - encode_prompt
def encode_prompt(text, vocab, add_bos=True):
    # TODO: encode text into token ids using vocab, optionally prepending <bos>.
    encoded = []
    if add_bos:
        encoded.append(vocab['token_to_id']['<bos>'])

    unk_tok = vocab['token_to_id'].get('<unk>', None)
    return encoded + [vocab['token_to_id'].get(ch, unk_tok) for ch in text if ch in vocab['token_to_id'] or unk_tok is not None]

# Step 9 - decode_tokens
def decode_tokens(token_ids, vocab, skip_special=True):
    # TODO: convert token ids back into a string using vocab['id_to_token'], optionally skipping specials.
    return ''.join([vocab['id_to_token'][tok] for tok in token_ids if not(skip_special and '<' in vocab['id_to_token'][tok])])

# Step 10 - embed_tokens
import numpy as np

def embed_tokens(token_ids, embedding_matrix):
    # TODO: return the (T, D) embedding rows for each token id in token_ids
    return embedding_matrix[token_ids]

# Step 11 - linear_projection
def linear_projection(x, weight, bias=None):
    # TODO: Apply y = x @ weight + bias, with bias optional and broadcasting over leading axes.
    proj = x @ weight
    if bias is not None:
        proj += bias

    return proj

# Step 12 - init_kv_cache
import numpy as np

def init_kv_cache(max_seq_len, d_model):
    # TODO: allocate empty K and V buffers and a length counter for a single sequence
    K = np.zeros((max_seq_len, d_model), dtype=np.float32)
    V = np.zeros((max_seq_len, d_model), dtype=np.float32)

    return dict(
        K=K,
        V=V,
        length=0
    )

# Step 13 - append_kv
import numpy as np

def append_kv(cache, k_new, v_new):
    # TODO: write k_new and v_new into the cache starting at cache['length'] and bump length.
    seq_len = cache['length']
    cur_len = k_new.shape[0]

    cache['K'][seq_len:seq_len+cur_len] = k_new
    cache['V'][seq_len:seq_len+cur_len] = v_new
    cache['length'] += cur_len

    return cache

# Step 14 - causal_attention
import numpy as np

def causal_attention(q, k, v, is_causal=True):
    # TODO: scaled dot-product attention with optional causal mask, returns (Tq, D)
    seq_len, d_model = q.shape
    scale = 1.0/(d_model**0.5)
    scores = (q @ k.T) * scale

    if is_causal:
        mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
        scores += mask

    attn = stable_softmax(scores)

    return attn @ v

# Step 15 - model_prefill
def model_prefill(token_ids, params):
    # TODO: embed tokens, project Q/K/V, fill the KV cache, run causal attention, return last-position logits.
    cache = init_kv_cache(params['max_seq_len'], params['Wq'].shape[0])

    x = embed_tokens(token_ids, params['embedding'])
    q = linear_projection(x, params['Wq'], params.get('bq', None))
    k = linear_projection(x, params['Wk'], params.get('bk', None))
    v = linear_projection(x, params['Wv'], params.get('bv', None))

    attn = causal_attention(q, k, v, is_causal=True)
    cache = append_kv(cache, k, v)

    out = linear_projection(attn, params['Wo'], params.get('bo', None))
    logits = linear_projection(out[-1], params['W_out'], params.get('b_out', None))

    return np.array(logits.tolist()), cache

# Step 16 - model_decode_step
def model_decode_step(token_id, cache, params):
    """Advance generation by one token using the existing KV cache."""
    # TODO: advance generation by one token using the existing KV cache and return next-token logits
    x = embed_tokens([token_id], params['embedding'])
    q = linear_projection(x, params['Wq'], params.get('bq', None))
    k = linear_projection(x, params['Wk'], params.get('bk', None))
    v = linear_projection(x, params['Wv'], params.get('bv', None))

    cache = append_kv(cache, k, v)
    attn = causal_attention(q, cache['K'][:cache['length']], cache['V'][:cache['length']], is_causal=False)

    out = linear_projection(attn, params['Wo'], params.get('bo', None))
    logits = linear_projection(out, params['W_out'], params.get('b_out', None))

    return np.array(logits[0].tolist()), cache

# Step 17 - blocks_needed
def blocks_needed(num_tokens, block_size):
    # TODO: return the number of fixed-size blocks needed to store num_tokens tokens.
    return (num_tokens + block_size - 1)//block_size

