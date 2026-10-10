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

# Step 18 - init_block_allocator
def init_block_allocator(num_blocks, block_size, d_model):
    # TODO: build the paged KV allocator dict with K_blocks, V_blocks, free_list, seq_tables, and config.
    return dict(
        K_blocks=np.zeros((num_blocks, block_size, d_model)),
        V_blocks=np.zeros((num_blocks, block_size, d_model)),
        free_list=list(range(num_blocks)),
        block_size=block_size,
        num_blocks=num_blocks,
        d_model=d_model,
        seq_tables={}
    )

# Step 19 - allocate_block
def allocate_block(allocator, seq_id):
    # TODO: pop one free block id and append it to allocator['seq_tables'][seq_id]; raise RuntimeError if OOM.
    if not allocator['free_list']:
        raise RuntimeError("Out of Memory blocks to allocate")

    block_id = allocator['free_list'].pop()
    allocator['seq_tables'][seq_id] = allocator['seq_tables'].get(seq_id, []) + [block_id]

    return block_id

# Step 20 - free_block
def free_block(allocator, block_id):
    # TODO: return block_id to allocator['free_list']
    allocator['free_list'] += [block_id]

# Step 21 - append_to_paged_cache
def append_to_paged_cache(allocator, seq_id, k_new, v_new):
    """Write t new K/V rows into the sequence's paged blocks, allocating as needed."""
    # TODO: append k_new/v_new (t, d_model) rows into seq_id's paged blocks, growing the block table when needed.
    if 'seq_lengths' not in allocator:
        allocator['seq_lengths'] = {sid: len(allocator['seq_tables'][sid]) for sid in allocator['seq_tables']}

    L = allocator['seq_lengths'].get(seq_id, 0)
    t = k_new.shape[0]
    num_blocks = blocks_needed(L + t, allocator['block_size'])
    for _ in range(num_blocks-L):
        bid = allocate_block(allocator, seq_id)
        # allocator['seq_lengths'][seq_id] += 1

    for i in range(t):
        p = L + i
        b = p//allocator['block_size']
        s = p%allocator['block_size']
        block_id = allocator['seq_tables'][seq_id][b]
        allocator['K_blocks'][block_id, s] = k_new[i]
        allocator['V_blocks'][block_id, s] = v_new[i]

    allocator['seq_lengths'][seq_id] = L + t

# Step 22 - gather_kv_from_blocks
def gather_kv_from_blocks(allocator, seq_id):
    # TODO: reconstruct contiguous (length, d_model) K and V from the sequence's paged blocks.
    L = allocator['seq_lengths'].get(seq_id, 0)
    K = []
    V = []

    for bid in allocator['seq_tables'][seq_id]:
        K.append(allocator['K_blocks'][bid])
        V.append(allocator['V_blocks'][bid])

    K = np.concatenate(K, axis=0)[:L].astype(np.float32)
    V = np.concatenate(V, axis=0)[:L].astype(np.float32)

    return K, V

# Step 23 - paged_attention_step
def paged_attention_step(q, allocator, seq_id):
    # TODO: gather K, V for seq_id from the paged allocator and run causal attention with q
    k, v = gather_kv_from_blocks(allocator, seq_id)
    logits = causal_attention(q, k, v, is_causal=True)

    return np.array(logits.tolist())

# Step 24 - free_sequence_blocks
def free_sequence_blocks(allocator, seq_id):
    # TODO: release all blocks owned by seq_id and remove its entry from seq_tables.
    for bid in allocator['seq_tables'].get(seq_id, []):
        free_block(allocator, bid)

    if seq_id in allocator['seq_tables']:
        del allocator['seq_tables'][seq_id]

# Step 25 - kv_blocks_in_use
def kv_blocks_in_use(allocator):
    # TODO: report allocator usage as {'used': int, 'free': int, 'total': int}.
    return dict(
        used=allocator['num_blocks'] - len(allocator['free_list']),
        free=len(allocator['free_list']),
        total=allocator['num_blocks']
    )

# Step 26 - make_request
def make_request(request_id, prompt_token_ids, max_new_tokens, sampling_params):
    # TODO: package the request id, prompt tokens, generation budget, and sampling params into a dict.
    return dict(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids.copy(),
        max_new_tokens=max_new_tokens,
        sampling_params=sampling_params
    )

# Step 27 - init_sequence_state
def init_sequence_state(request, params):
    # TODO: Initialize per-sequence state by running prefill and storing cache/logits.
    logits, cache = model_prefill(request['prompt_token_ids'], params)
    return dict(
        request_id=request['request_id'],
        prompt_token_ids=request['prompt_token_ids'].copy(),
        generated=[],
        cache=cache,
        last_logits=logits,
        done=False,
        sampling_params=request['sampling_params'],
        max_new_tokens=request['max_new_tokens']
    )

# Step 28 - sequence_decode_step
def sequence_decode_step(state, params, rng):
    # TODO: sample next token from state['last_logits'], advance cache via model_decode_step, append token.
    if not 'temperature' in state['sampling_params'] or state['sampling_params']['temperature']<=0:
        token_id = greedy_select(state['last_logits'])
    else:
        logits = state['last_logits']
        if 'temperature' in state['sampling_params']:
            logits = apply_temperature(logits, state['sampling_params']['temperature'])
        if 'top_k' in state['sampling_params']:
            logits = top_k_filter(logits, state['sampling_params']['top_k'])
        if 'top_p' in state['sampling_params']:
            logits = top_p_filter(logits, state['sampling_params']['top_p'])

        probs = stable_softmax(logits)
        token_id = sample_from_probs(probs, rng)

    logits, cache = model_decode_step(token_id, state['cache'], params)
    state['cache'] = cache
    state['last_logits'] = logits
    state['generated'].append(token_id)

    return token_id, state

# Step 29 - is_sequence_done
def is_sequence_done(state, eos_token_id):
    # TODO: return True if state has hit max_new_tokens budget or last generated token is EOS
    ret = False
    if len(state['generated']) >= state['max_new_tokens']:
        state['done'] = True
        ret = True
    if len(state['generated'])>0 and state['generated'][-1] == eos_token_id:
        state['done'] = True
        ret = True

    return ret

# Step 30 - generate_single_sequence
def generate_single_sequence(request, params, eos_token_id, rng):
    # TODO: drive end-to-end generation for one request and return only the generated token ids.
    state = init_sequence_state(request, params)
    tokens = []
    while not is_sequence_done(state, eos_token_id):
        token_id, state = sequence_decode_step(state, params, rng)
        tokens.append(token_id)

    return tokens

# Step 31 - build_batch_step_input
import numpy as np

def build_batch_step_input(sequences):
    # TODO: collect the last token id from each non-done sequence into a (B,) int64 array.
    input_ids = []
    active_indices = []

    for i, seq in enumerate(sequences):
        if not seq['done']:
            if len(seq['token_ids']) > 0:
                input_ids.append(seq['token_ids'][-1])
            active_indices.append(i)

    return dict(
        active_indices=active_indices,
        input_ids=np.array(input_ids, dtype=np.int64)
    )

# Step 32 - batched_decode_step
def batched_decode_step(params, sequences, sampling_config):
    """Run one synchronized decode step across active sequences."""
    # TODO: For each active sequence, run a decode step and append the sampled token.
    batch = build_batch_step_input(sequences)
    for idx, tok in zip(batch['active_indices'], batch['input_ids']):
        seq = sequences[idx]
        logits, _ = model_decode_step(int(tok), seq['kv_cache'], params)

        if not 'temperature' in sampling_config or sampling_config['temperature']<=0:
            token_id = greedy_select(logits)
        else:
            if 'temperature' in sampling_config:
                logits = apply_temperature(logits, sampling_config['temperature'])
            if 'top_k' in sampling_config:
                logits = top_k_filter(logits, sampling_config['top_k'])
            if 'top_p' in sampling_config:
                logits = top_p_filter(logits, sampling_config['top_p'])

            probs = stable_softmax(logits)
            token_id = sample_from_probs(probs, sampling_config['rng'])

        sequences[idx]['token_ids'].append(token_id)

    return sequences

