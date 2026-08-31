# Transformer from Scratch (NumPy + Word2Vec)

A custom, educational implementation of a multi-layer **Encoder-Decoder Transformer** built entirely from scratch using **NumPy** and pretrained **Word2Vec (Gensim)** embeddings. 

Instead of relying on high-level black-box libraries like PyTorch or Hugging Face, this project manually wires up the raw mathematical operations required to process text sequences and run transformer blocks.

---

## 🚀 Features & Architecture

* **Custom Token Embeddings:** Integrates `gensim-downloader` (`glove-wiki-gigaword-50`) to convert raw text strings into dense vector representations with out-of-vocabulary fallback handling.
* **Encoder & Decoder Blocks:** Built with explicit object-oriented classes (`Encoder`, `Decoder`) tracking weights, biases, and dimensions.
* **Scaled Dot-Product Attention:** Manually implements Query ($Q$), Key ($K$), and Value ($V$) matrix dot products scaled by $\sqrt{d_k}$ and routed through a numerical-stable softmax function.
* **Cross-Attention:** Connects the decoder layers to encoder states.
* **Feed-Forward Networks (FFN):** Implements linear projections and activation functions (`tanh`).
* **Layer Normalization:** Manually calculates mean, standard deviation, gamma, and beta parameters with stabilization epsilon ($\epsilon$).

---

## 🛠️ Tech Stack

* **Python**
* **NumPy** (for raw tensor manipulation and matrix math)
* **Gensim** (for word vector embeddings)

---

## 💻 Code Structure

The project is structured as a standalone Google Colab / Python script containing:
1. **Embedding Pipeline:** Tokenizing user input and mapping words to vector spaces.
2. **Encoder Class:** Self-attention loops, feed-forward layers, and layer normalization.
3. **Decoder Class:** Self-attention, cross-attention, projection layers, and final output mappings.
4. **Training/Inference Loop:** Iterative forward-pass simulation projecting model outputs back into embedding space.

---

## 🏃‍♂️ Quick Start

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Transformer_from_scratch.git](https://github.com/YOUR_USERNAME/Transformer_from_scratch.git)
