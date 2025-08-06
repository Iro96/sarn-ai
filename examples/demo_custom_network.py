from SarnAi import SARNLayer, SARNNetwork

# Define a custom SARN-based network
input_layer = SARNLayer(input_size=4, output_size=8, k=3)
hidden_layer = SARNLayer(input_size=8, output_size=6, k=2)
output_layer = SARNLayer(input_size=6, output_size=2, k=1)

model = SARNNetwork([input_layer, hidden_layer, output_layer])

# Forward pass
output = model.forward([0.3, 0.7, 0.1, 0.9])
print("Output:", output)

# Self-adapt via rewiring and neuron growth
model.adapt()