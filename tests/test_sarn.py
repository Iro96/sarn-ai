from SarnAi.network import SARNNetwork

def test_output_shape():
    model = SARNNetwork(4, 8, 2)
    output = model.forward([0.1, 0.2, 0.3, 0.4])
    assert len(output) == 2
