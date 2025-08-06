import numpy as np

def plasticity_update(layer):
    if layer.last_input is None or layer.last_output is None:
        return
    used = np.abs(layer.last_output) > 0
    for i, active in enumerate(used):
        if active:
            layer.weights[i] += 0.01 * layer.last_output[i] * layer.last_input
        else:
            layer.weights[i] *= 0.99  # decay

def neurogenesis_check(memory):
    if len(memory.memory) < 2:
        return False
    recent = memory.get_recent(2)
    delta = np.linalg.norm(recent[-1] - recent[-2])
    return delta > 50
