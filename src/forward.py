'''
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(x,w,b,n_layers):
    for i in range(n_layers):
        x = np.dot(x,w[i]) + b[i]
    y = np.sum(x)
    output = sigmoid(y)
    return output

if __name__ == "__main__":
    x = np.random.rand(2, 1000)

    n_layers = 2  # Number of layers; adjust as needed
    layer_sizes = [1]  # Size of each layer; for a single layer, it's just 1

    # Initialize weights and biases
    w = [np.random.rand(1000, layer_sizes[0]) for _ in range(n_layers)]  # Adjust dimensions as per layer_sizes
    b = [np.random.rand(layer_sizes[0]) for _ in range(n_layers)]

    # Get the output
    output = forward(x, w, b, n_layers)
    print(output)
'''
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def forward(x, w, b, n_layers):
    buffer = 1
    x = x.astype(np.float64)
    w = [iw.astype(np.float64) for iw in w]
    b = [ib.astype(np.float64) for ib in b]
    #print(x)
    #print(w)
    #print(b)
    for i in range(n_layers):
        x = np.dot(x, w[i]*buffer) + b[i]*buffer
        #print(x)
        #if i < n_layers - 1:  # Apply sigmoid activation for all layers except the last
        #x = sigmoid(x)
        #print(x)
        #a = 1 + '1'
    y = np.sum(x)/1e10
    #print(y)
    y = sigmoid(y)
    return y

def attention(weights, query, keys, values):
    # Simple dot-product attention
    attn_scores = np.dot(query, keys.T)
    attn_distribution = softmax(attn_scores)
    output = np.dot(attn_distribution, values)
    return output * weights

def forward_with_attention(x, w, b, n_layers, attn_weights, attn_query, attn_keys, attn_values):
    x = x.astype(np.float64)
    w = [iw.astype(np.float64) for iw in w]
    b = [ib.astype(np.float64) for ib in b]

    for i in range(n_layers):
        x = np.dot(x, w[i]) + b[i]

        # Add attention mechanism at a specific layer
        if i == 0:
            x = attention(attn_weights, attn_query, attn_keys, x)

    y = np.sum(x) / 1e10
    y = sigmoid(y)
    return y

if __name__ == "__main__":
    from datetime import datetime
    np.random.seed(0)
    x = np.random.rand(1000)

    n_layers = 5
    input_size = 1000
    layer_sizes = [500, 200, 100, 50, 1]

    np.random.seed(None)
    # Initialize weights and biases
    w = [np.random.rand(input_size if i == 0 else layer_sizes[i - 1], size) for i, size in enumerate(layer_sizes)]
    b = [np.random.rand(size) for size in layer_sizes]

    attention_layer_index = 0  # Index of the layer after which you want to apply attention

    # Dimension of the layer output where attention is applied
    layer_output_dim = layer_sizes[attention_layer_index]

    # Initialize attention weights
    # Assuming the query, keys, and values have the same dimension for simplicity
    attn_dim = layer_output_dim  # You can choose a different dimension if needed

    attn_query = np.random.rand(attn_dim).astype(np.float64)
    attn_keys = np.random.rand(attn_dim, attn_dim).astype(np.float64)
    attn_values = np.random.rand(attn_dim, attn_dim).astype(np.float64)
    attn_weights = np.random.rand(attn_dim).astype(np.float64)

    # Get the output
    stopwatch = datetime.now()
    #output = forward(x, w, b, n_layers)
    output = forward_with_attention(x, w, b, n_layers, attn_weights, attn_query, attn_keys, attn_values)
    runtime = datetime.now() - stopwatch
    print(output)
    print(f"Runtime: {runtime}")
