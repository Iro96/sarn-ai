<div align="center">
<a name="readme-top"></a>

# SARN: Self-Adaptive Rewiring Neural Network

**SARN** is a modular, brain-inspired neural network engine that supports **sparse top-k activations**, **neurogenesis**, **self-rewiring**, and **biological activation functions**. Designed to simulate real neuron behavior, it's ideal for building your own evolving neural systems with full control over nodes, weights, connections, and more.

<img src="https://github.com/Iro96/SarnAI/blob/main/img/logo.png" width="200" height="200"/>

[![License](https://img.shields.io/badge/License-Apache2.0-blue.svg)](https://opensource.org/license/apache2.0/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sarn)
![PyPI](https://img.shields.io/pypi/v/sarn)
[![Stars](https://img.shields.io/github/stars/Iro96/sarnAI.svg?style=social&label=Star)](https://github.com/yourusername/sarnAI)

</div>

---

## 🌟 Features

- **SARN Layers** — Top-k activated sparse neurons with rewiring
- **Neuron Growth** — Dynamically adds neurons when novelty is detected
- **Hebbian Plasticity** — Rewires weights based on usage strength
- **Fully Modular** — Build networks with any size, connection config, and activation

---

## Getting Started

### 🔧 Install via PyPI

```bash
pip install sarn
```
or

### $ Clone this repository

```bash
git clone https://github.com/yourusername/sarn-ai.git
cd sarn-ai
pip install -e .
```

### ☁ Usage

**Example**

```bash
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
```

**Terminal**

```bash
Input: [0.2 0.8 0.5 0.1]
Output: [0.00000000e+00 5.00453877e-10]
```

---

***Customize***

| Feature          | How to Modify                                |
| ---------------- | -------------------------------------------- |
| # of neurons     | Change `output_size` in `SARNLayer`          |
| Activation       | Set `activation=your_func` in `SARNLayer`    |
| Weights / biases | Use `layer.set_weights()` or `.set_biases()` |
| Neuron growth    | `model.adapt()` + auto-neurogenesis          |


***Activation Functions***

```bash
from SarnAi.functions import neuro_spike, adaptive_pulse, neuro_softmax
```

| Function         | Uses                                 |
| ---------------- | ------------------------------------ |
| `neuro_spike`    | Peak around x=1 like a firing neuron |
| `adaptive_pulse` | Smooth pulse that adapts with input  |
| `neuro_softmax`  | Confidence-weighted class activation |

---

## ⚙️ API Overeview

**Class**

| Class         | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `SARNLayer`   | Customizable sparse neural layer with weights and top-k fire |
| `SARNNetwork` | Connect multiple layers and manage memory + neurogenesis     |

**Functions**

| Function               | Purpose                                     |
| ---------------------- | ------------------------------------------- |
| `model.forward(input)` | Runs input through all SARN layers          |
| `model.adapt()`        | Rewires weights and grows neurons if needed |
| `layer.set_weights()`  | Manually set layer weights                  |
| `layer.add_neuron()`   | Dynamically add a neuron to a layer         |

---

## Contributing
*Pull requests are welcome! If you want to propose improvements or report issues, feel free to fork the repo and open a PR or GitHub Issue.*
> ***!Note:*** This project is in active development and experimental.

---

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Iro96/SarnAI&type=Date&theme=dark)](https://star-history.com/#Iro96/SarnAI&Date)

---

<p align="right">
  <a href="#readme-top">↑ Back to Top ↑</a>
</p>
