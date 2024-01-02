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

def forward(x, w, b, n_layers):
    x = x.astype(np.float32)
    w = [iw.astype(np.float32) for iw in w]
    b = [ib.astype(np.float32) for ib in b]
    for i in range(n_layers):
        x = np.dot(x, w[i]) + b[i]
        #if i < n_layers - 1:  # Apply sigmoid activation for all layers except the last
        x = sigmoid(x)
    y = np.sum(x)
    output = sigmoid(y)
    return output

if __name__ == "__main__":
    from datetime import datetime
    x = np.random.rand(2, 1000)

    n_layers = 5
    input_size = 1000
    layer_sizes = [500, 200, 100, 50, 1]

    # Initialize weights and biases
    w = [np.random.rand(input_size if i == 0 else layer_sizes[i - 1], size) for i, size in enumerate(layer_sizes)]
    b = [np.random.rand(size) for size in layer_sizes]

    # Get the output
    stopwatch = datetime.now()
    output = forward(x, w, b, n_layers)
    runtime = datetime.now() - stopwatch
    print(output)
    print(f"Runtime: {runtime}")
