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
        if 'max_new_tokens' not in seq or ('max_new_tokens' in seq and seq['max_new_tokens'] > len(seq['generated'])):
            logits, cache = model_decode_step(int(tok), seq['kv_cache'], params)

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
            sequences[idx]['generated'].append(token_id)
            sequences[idx]['kv_cache'] = cache

    return sequences

# Step 33 - static_batch_generate
def static_batch_generate(params, requests, sampling_config, max_new_tokens):
    """Run prefill for all requests, then iterate batched decode steps until each
    sequence hits its per-request budget or the global max_new_tokens cap."""
    # TODO: prefill each request, then loop sampling next tokens until done.
    sequences = []
    for req in requests:
        state = init_sequence_state(req, params)
        state['token_ids'] = state['prompt_token_ids'] + state['generated']
        state['kv_cache'] = state['cache']
        sequences.append(state)

    for _ in range(max_new_tokens):
        sequences = batched_decode_step(params, sequences, sampling_config)
    return [dict(request_id=seq['request_id'], output_ids=seq['generated']) for seq in sequences]

# Step 34 - has_free_capacity
def has_free_capacity(allocator, required_blocks):
    # TODO: return True iff allocator has at least required_blocks free blocks.
    return len(allocator['free_list']) >= required_blocks

# Step 35 - continuous_batch_step
def continuous_batch_step(params, running, allocator, sampling_config):
    """Advance every active sequence in `running` by one decoded token using the paged allocator."""
    # TODO: for each non-done sequence, project Q/K/V from its last token, append to the paged cache, run paged attention, sample, and append.
    for ind, seq in enumerate(running):
        seq_id = seq.get('seq_id', ind)
        if seq['done']:
            continue
        # done = is_sequence_done(seq, sampling_config['eos_token_id'])
        # if done:
        #     running[seq_id]['done'] = True
        #     free_sequence_blocks(allocator, seq_id)
        else:
            token_id = seq['token_ids'][-1]
            x = embed_tokens([token_id], params['embedding'])
            q = linear_projection(x, params['Wq'], params.get('bq', None))
            k_new = linear_projection(x, params['Wk'], params.get('bk', None))
            v_new = linear_projection(x, params['Wv'], params.get('bv', None))

            append_to_paged_cache(allocator, seq_id, k_new, v_new)
            attn = paged_attention_step(q, allocator, seq_id)

            out = linear_projection(attn, params['Wo'], params.get('bo', None))
            logits = linear_projection(out, params['W_out'], params.get('b_out', None))[0]
            # logits = linear_projection(out, params['lm_head'])[0]

            if ('greedy' in sampling_config and sampling_config['greedy']) or not 'temperature' in sampling_config or sampling_config['temperature']<=0:
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

            running[ind]['token_ids'].append(token_id)
            running[ind]['generated'].append(token_id)
            running[ind]['length'] += 1
            running[ind]['done'] = is_sequence_done(seq, sampling_config['eos_token_id'])
            if running[ind]['done']:
                free_sequence_blocks(allocator, seq_id)

    return running

# Step 36 - run_continuous_batching
def run_continuous_batching(params, requests, allocator, sampling_config, max_steps):
    # TODO: Drive the continuous-batching loop: admit, decode, retire finished sequences.
    waiting = [req for req in requests]
    max_seq_len = 0
    for seq_id in range(len(waiting)):
        waiting[seq_id]['seq_id'] = seq_id
        waiting[seq_id]['sampling_params'] = waiting[seq_id].get('sampling_params', {})
        max_seq_len = max(max_seq_len, len(waiting[seq_id]['prompt_token_ids']) + waiting[seq_id]['max_new_tokens'])
    running = []
    completed = []

    params['max_seq_len'] = max_seq_len

    for _ in range(max_steps):
        stats = kv_blocks_in_use(allocator)
        i = 0
        while i < len(waiting):
            if blocks_needed(len(waiting[i]['prompt_token_ids']), allocator['block_size']) <= stats['free']:
                state = init_sequence_state(waiting[i], params)
                append_to_paged_cache(allocator, waiting[i]['seq_id'], state['cache']['K'], state['cache']['V'])
                del state['cache']
                state['seq_id'] = waiting[i]['seq_id']
                state['token_ids'] = state['prompt_token_ids'] + state['generated']
                state['length'] = len(state['token_ids'])
                running.append(state)
                stats = kv_blocks_in_use(allocator)
                waiting.pop(i)
            else:
                i += 1

        continuous_batch_step(params, running, allocator, sampling_config)
        r = 0
        while r < len(running):
            if running[r]['done']:
                completed.append(running.pop(i))
            else:
                r += 1

        if len(waiting) + len(running) == 0:
            break
    else:
        completed.extend(running)

    return [dict(request_id=req['request_id'], output_ids=req['generated']) for req in completed]

# Step 37 - priority_queue_push
import heapq

def priority_queue_push(heap, priority, request):
    # TODO: push (priority, counter, request) onto the min-heap with stable tie-breaking
    counter = 0
    if heap:
        counter = max(heap, key=lambda x: x[1])[1] + 1

    heapq.heappush(heap, (priority, counter, request))
    return heap

# Step 38 - priority_queue_pop
import heapq

def priority_queue_pop(heap):
    # TODO: Pop and return the request with the smallest priority from the min-heap, or None if empty.
    return heapq.heappop(heap)[-1] if heap else None

# Step 39 - select_admissions
def select_admissions(waiting_heap, allocator, block_size, max_admit):
    # TODO: Pop requests from the waiting priority queue and admit as many as the allocator can host.
    admitted = []
    available = len(allocator['free_list'])
    while len(admitted) < max_admit:
        req = priority_queue_pop(waiting_heap)
        if req is None:
            break
        n_blocks = blocks_needed(len(req['prompt_token_ids']), block_size)
        if n_blocks <= available:
            admitted.append(req)
            available -= n_blocks
        else:
            priority_queue_push(waiting_heap, req['priority'], req)
            break

    return admitted

# Step 40 - preempt_sequence
def preempt_sequence(sequence, allocator, waiting_heap):
    # TODO: free the sequence's KV blocks and re-queue its request on the waiting heap.
    free_sequence_blocks(allocator, sequence['request_id'])
    priority_queue_push(waiting_heap, sequence['priority'], sequence)

    return sequence

# Step 41 - schedule_step
def schedule_step(waiting_heap, running, allocator, block_size, max_running):
    # TODO: preempt over-capacity sequences, then admit from the waiting heap up to max_running.
    while len(running) > max_running:
        ind = max(range(len(running)), key=lambda x: (running[x]['priority'], x))
        sequence = running.pop(ind)
        preempt_sequence(sequence, allocator, waiting_heap)

    slots = max(max_running - len(running), 0)
    admitted = select_admissions(waiting_heap, allocator, block_size, slots)

    return dict(
        running=running,
        newly_admitted=admitted
    )

# Step 42 - format_stream_chunk
def format_stream_chunk(request_id, token_id, token_text, finished):
    # TODO: package a streaming token event into a chunk dict with keys request_id, token_id, text, finished.
    return dict(
        request_id=request_id,
        token_id=token_id,
        text=token_text,
        finished=finished
    )

# Step 43 - submit_request
def submit_request(server_state, prompt, max_new_tokens, priority, vocab):
    # TODO: encode the prompt, build a request record, push it on the waiting heap, return new id.
    token_ids = encode_prompt(prompt, vocab)
    request_id = f"req-{server_state['next_request_id']}"
    server_state['next_request_id'] += 1

    req = make_request(request_id, token_ids, max_new_tokens, {})
    req['priority'] = priority

    priority_queue_push(server_state['waiting_heap'], priority, req)
    return request_id

# Step 44 - drive_until_complete
def drive_until_complete(server_state, params, allocator, sampling_config, vocab, max_steps):
    # TODO: run admit/decode until queues are empty or max_steps is hit; return the new stream chunks.
    max_running = server_state.get('max_running', 4)
    server_state['completed'] = {}
    server_state['streams'] = {}
    if len(server_state['waiting_heap']) + len(server_state['running']) == 0:
        return []

    eos_token_id = server_state.get('eos_token_id', sampling_config.get('eos_token_id', -1))
    sampling_config['eos_token_id'] = eos_token_id
    block_size = allocator['block_size']

    # chunks = {}
    chunks = []
    # step_chunks = []

    for _ in range(max_steps):
        # chunks = []
        schedule = schedule_step(server_state['waiting_heap'], server_state['running'], allocator, block_size, max_running)
        for seq in schedule['newly_admitted']:
            seq['seq_id'] = seq['request_id']
            token_ids = seq.get(
                'token_ids', seq['prompt_token_ids'] + seq.get('generated', []))
            prefix_ids = token_ids[:-1]
            if prefix_ids:
                x = embed_tokens(prefix_ids, params['embedding'])
                k = linear_projection(x, params['Wk'], params.get('bk'))
                v = linear_projection(x, params['Wv'], params.get('bv'))
                append_to_paged_cache(allocator, seq['seq_id'], k, v)

        server_state['running'] = schedule['running'] + schedule['newly_admitted']
        for i in range(len(server_state['running'])):
            server_state['running'][i]['done'] = server_state['running'][i].get('done', False)
            if 'generated' not in server_state['running'][i]:
                server_state['running'][i]['generated'] = []
            if 'token_ids' not in server_state['running'][i]:
                server_state['running'][i]['token_ids'] = server_state['running'][i]['prompt_token_ids'] + server_state['running'][i]['generated']
            server_state['running'][i]['length'] = len(server_state['running'][i]['token_ids'])
        continuous_batch_step(params, server_state['running'], allocator, sampling_config)

        i = 0
        while i < len(server_state['running']):
            seq = server_state['running'][i]
            token_id = seq['token_ids'][-1]
            token_text = decode_tokens([token_id], vocab)
            finished = seq['done']
            chunk = format_stream_chunk(seq['request_id'], token_id, token_text, finished)
            chunks.append(chunk)
            # chunks[seq['request_id']] = chunks.get(seq['request_id'], []) + [chunk]
            server_state['streams'][seq['request_id']] = server_state['streams'].get(seq['request_id'], []) + [chunk]

            if finished:
                finish_reason = 'EOS token' if seq['generated'][-1]==eos_token_id else "Budget reached"
                server_state['completed'][seq['request_id']] = dict(request_id=seq['request_id'], output_ids=seq['generated'], chunks=server_state['streams'][seq['request_id']], finish_reason=finish_reason)
                free_sequence_blocks(allocator, seq['request_id'])
                server_state['running'].pop(i)
            else:
                 i += 1

        # step_chunks.append(chunks)

    return chunks

# Step 45 - collect_request_output
def collect_request_output(server_state, request_id):
    # TODO: look up the completed record for request_id and return its output_ids and chunks
    if 'completed' in server_state and request_id in server_state['completed']:
        return dict(
            request_id=request_id,
            output_ids=server_state['completed'][request_id]['output_ids'],
            chunks=server_state['completed'][request_id].get('chunks', server_state['completed'][request_id].get('streams', None))
        )

# Step 46 - build_completion_response
def build_completion_response(server_state, request_id, vocab):
    # TODO: build the final OpenAI-style completion dict from the completed record.
    out = collect_request_output(server_state, request_id)
    if out is None:
        return out

    ends_with_eos = False
    if len(out['output_ids']):
        ends_with_eos = decode_tokens([out['output_ids'][-1]], vocab, False)

    finish_reason = server_state['completed'][request_id].get('finish_reason', None) or "eos" if ends_with_eos else "stop"
    return dict(
        request_id=request_id,
        text=decode_tokens(out['output_ids'], vocab),
        output_ids=out['output_ids'].copy(),
        finish_reason=finish_reason
    )

# Step 47 - time_to_first_token
def time_to_first_token(events):
    # TODO: compute per-request TTFT from a list of timestamped serving events.
    submit = {}
    ttft = {}

    for event in events:
        if event['event'] == 'submit':
            submit[event['request_id']] = event['time']
        elif event['event'] == 'token' and event['request_id'] not in ttft:
        # elif (event['event'] == 'token' or event['event'] == 'finished') and event['request_id'] not in ttft:
            if event['request_id'] in submit:
                ttft[event['request_id']] = event['time'] - submit[event['request_id']]
        elif event['event'] == 'token' and event['request_id'] in ttft:
            if event['request_id'] in submit:
                ttft[event['request_id']] = min(ttft[event['request_id']], event['time'] - submit[event['request_id']])

    return ttft

