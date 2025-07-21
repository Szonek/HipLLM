import torch
torch.manual_seed(123)

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]])
print("input shape:", inputs.shape)



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

ss_v1 = SelfAttention_v1(3, 2)
print(ss_v1(inputs))
