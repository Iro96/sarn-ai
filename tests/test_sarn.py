from sarn.network import SARNNetwork

def test_output_shape():
    model = SARNNetwork(input_size=4, hidden_size=8, output_size=2)
    output = model.forward([0.1, 0.2, 0.3, 0.4])
    assert len(output) == 2
