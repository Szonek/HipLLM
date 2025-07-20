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

simplified_attention(inputs)

