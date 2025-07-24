import torch




def simplified_attention(inputs): 
    # simplest form of attention mechanism
    full_attn = True
    if full_attn:
        attn_scores = inputs @ inputs.T
        print("Attn scores:\n", attn_scores)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        print("Attn weights:\n", attn_weights)
        print("Attn weights row sums:\n", attn_weights.sum(dim=-1))
        context_vec = attn_weights @ inputs
        print("Context vec:\n", context_vec)
    else:
        # only 2nd query for visualisation purposes
        query = inputs[1]
        print("Query shape: ", query.shape)
        attn_scores_2 = torch.empty(inputs.shape[0])
        for i, x_in in enumerate(inputs):
            attn_scores_2[i] = torch.dot(query, x_in)
        print("Attn scores", attn_scores_2)
        attn_weights_2_tmp = torch.softmax(attn_scores_2, dim=0)
        print("Attn weights: ", attn_weights_2_tmp)
        print("Attn weights sum: ", attn_weights_2_tmp.sum())

        context_vec_2 = torch.zeros(query.shape)
        for i, x_in in enumerate(inputs):
            context_vec_2 += attn_weights_2_tmp[i] * x_in
        print("Context vec: ", context_vec_2)
#simplified_attention(inputs)

def scaled_dot_product_attention(inputs):
    # scaled dot product attention (SDPA) used in popular LLMs
    x_2 = inputs[1]
    d_in = inputs.shape[1]
    d_out = 2 # in GPT models this is usally the same as input dim ("d_in")
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print("query_2: ", query_2)
    print("key_2: ", key_2)
    print("value_2: ", value_2)
    keys = inputs @ W_key
    values = inputs @ W_value
    print("keys shape:", keys.shape)
    print("values shape:", values.shape)

    attn_scores_2 = query_2 @ keys.T
    print("Attn scores: ", attn_scores_2)
    d_k = keys.shape[-1]
    scale_factor = 1.0 / (d_k**0.5)  
    # Divide by square root of key dimensions to stabilize gradients
    # and prevents softmax from becoming too sharp when dot products are too large (>1000)
    attn_weights_2 = torch.softmax(attn_scores_2 * scale_factor, dim=-1)
    print("Attn weights: ", attn_weights_2)
    context_vector = attn_weights_2 @ values
    print("Context vector: ", context_vector)
#scaled_dot_product_attention(inputs)

class SelfAttention_v1(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / (d_k**0.5), dim=-1)
        context_vec = attn_weights @ values
        return context_vec

#ss_v1 = SelfAttention_v1(3, 2)
#print(ss_v1(inputs))

class SelfAttention_v2(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

        #self.W_query.weight.data = ss_v1.W_query.T.data
        #self.W_key.weight.data = ss_v1.W_key.T.data
        #self.W_value.weight.data = ss_v1.W_value.T.data

    def forward(self, x):
        keys = self.W_key(x)   
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / (d_k**0.5), dim=-1)
        context_vec = attn_weights @ values
        return context_vec

#ss_v2 = SelfAttention_v2(3, 2)
#print(ss_v2(inputs))


class CausalAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout)
        # buffers are automatically moved to appropriate  device (CPU or GPU), so no need to manually ensure tensors are on the same device
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)   
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool(), -torch.inf)
        #print(attn_scores)
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / (d_k**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
    

#batch = torch.stack([inputs, inputs], dim=0)
#print(batch.shape)
#context_length = batch.shape[1]
#ca = CausalAttention(d_in=3, d_out=2, context_length=context_length, dropout=0.0)
#context_vecs = ca(batch)
#print("context_vecs.shape: ", context_vecs.shape)
#print(context_vecs)


class MultiHeadAttentionWrapper(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [CausalAttention(d_in=d_in, d_out=d_out, context_length=context_length, dropout=dropout) for i in range(num_heads)]
        )
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

#mha = MultiHeadAttentionWrapper(d_in=3, d_out=2, context_length=context_length, dropout=0.0, num_heads=2)
#context_vecs = mha(batch)
#print("context_vecs.shape: ", context_vecs.shape)
#print(context_vecs)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query  = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_out, d_out, bias=True)
        self.dropout  = torch.nn.Dropout(dropout)
        # buffers are automatically moved to appropriate  device (CPU or GPU), so no need to manually ensure tensors are on the same device
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x : torch.Tensor):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x).reshape(b, num_tokens, self.num_heads, self.head_dim)
        keys    = self.W_key(x).reshape(b, num_tokens, self.num_heads, self.head_dim)
        values  = self.W_value(x).reshape(b, num_tokens, self.num_heads, self.head_dim)

        # flip num_tokens and num_heads
        keys.transpose_(1, 2)
        queries.transpose_(1, 2)
        values.transpose_(1, 2)
        
        # compute attention for each head
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool() # [:num_tokens, :num_tokens]

        attn_scores.masked_fill(mask_bool, -torch.inf)

        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
         # flip back num_tokens and num_heads and reshape back
        context_vec.transpose_(1, 2)
        context_vec = self.out_proj(context_vec.reshape(b, num_tokens, self.d_out))
        return context_vec

#d_in = 3
#d_out = 2
#num_heads = 2
#batch_size, context_length, d_in = batch.shape
#mha = MultiHeadAttention(d_in=d_in, d_out=d_out, context_length=context_length,
#                          dropout=0.0, num_heads=num_heads)
#context_vecs = mha(batch)
#print("context_vecs.shape: ", context_vecs.shape)
#print(context_vecs)