| file  | comments                                                                                                                          |
|-------|-----------------------------------------------------------------------------------------------------------------------------------|
| gen01 | cpu training of NN that generates names (single hidden layer)                                                                     |
| gen02 | modules for embedding and flattening as shown in the video from 11:36 to 18:00                                                    |
| gen03 | switch architecture to wavenet-like (video discusses this from 19:00 and 31:40)                                                   |
| gen04 | convert 03 to pytorch, i want to plot activations with hooks     (i can't figure pytorch out, first i will try to learn flax/jax) |
| gen05 | convert 03 to jax                                                                                                                                  |

# install dependencies with micrconda

```
microconda activate
microconda install optax flax -c conda-forge
```

# Youtube video

- i go through the following video:


Link: https://www.youtube.com/watch?v=t3YJ5hKiMQ0
Title: Building makemore Part 5: Building a WaveNet
Author: Andrej Karpathy


## Abstract

This video continues the "makemore" series, focusing on improving the character-level language model by transitioning from a simple multi-layer perceptron (MLP) to a deeper, tree-like architecture inspired by WaveNet. The video delves into the implementation details, discussing PyTorch modules, containers, and debugging challenges encountered along the way. A key focus is understanding how to progressively fuse information from input characters to predict the next character in a sequence. While the video doesn't implement the exact WaveNet architecture with dilated causal convolutions, it lays the groundwork for future explorations in that direction. Additionally, the video provides insights into the typical development process of building deep neural networks, including reading documentation, managing tensor shapes, and using tools like Jupyter notebooks and VS Code.

## Summary

### Starter Code Walkthrough (1:43)

  - The starting point is similar to Part 3, with minor modifications.
  - Data generation code remains unchanged, providing examples of three characters to predict the fourth.
  - Layer modules like Linear, BatchNorm1D, and Tanh are reviewed.
  - The video emphasizes the importance of setting BatchNorm layers to training=False during evaluation.
  - Loss function visualization is improved by averaging values.

### PyTorchifying Our Code: Layers, Containers, Torch.nn, Fun Bugs (9:19)
  - Embedding table and view operations are encapsulated into custom Embedding and Flatten modules.
  - A Sequential container is created to organize layers, similar to torch.nn.Sequential.
  - The forward pass is simplified using these new modules and container.
  - A bug related to BatchNorm in training mode with single-example batches is identified and fixed.

### Overview: WaveNet (17:12)
  - The limitations of the current MLP architecture are discussed, particularly the issue of squashing information too quickly.
  - The video introduces the WaveNet architecture, which progressively fuses information in a tree-like structure.
  - The concept of dilated causal convolutions is briefly mentioned as an implementation detail for efficiency.

### Implementing WaveNet (19:35)
  - The dataset block size is increased to 8 to provide more context for predictions.
  - The limitations of directly scaling up the context length in the MLP are highlighted.
  - A hierarchical model is implemented using FlattenConsecutive layers to group and process characters in pairs.
  - The shapes of tensors at each layer are inspected to ensure the network functions as intended.
  - A bug in the BatchNorm1D implementation is identified and fixed to correctly handle multi-dimensional inputs.

### Re-training the WaveNet with Bug Fix (45:25)
  - The network is retrained with the BatchNorm1D bug fix, resulting in a slight performance improvement.
  - The video notes that PyTorch's BatchNorm1D has a different API and behavior compared to the custom implementation.

### Scaling up Our WaveNet (46:07)
  - The number of embedding and hidden units are increased, leading to a model with 76,000 parameters.
  - Despite longer training times, the validation performance improves to 1.993.
  - The need for an experimental harness to efficiently conduct hyperparameter searches is emphasized.

### Experimental Harness (46:59)
  - The lack of a proper experimental setup is acknowledged as a limitation of the current approach.
  - Potential future topics are discussed, including:
      - Implementing dilated causal convolutions
      - Exploring residual and skip connections
      - Setting up an evaluation harness
      - Covering recurrent neural networks and transformers

### Improve on My Loss! How Far Can We Improve a WaveNet on This Data? (55:27)
  - The video concludes with a challenge to the viewers to further improve the WaveNet model's performance.
  - Suggestions for exploration include:
      - Trying different channel allocations
      - Experimenting with embedding dimensions
      - Comparing the hierarchical network to a large MLP
      - Implementing layers from the WaveNet paper
      - Tuning initialization and optimization parameters

i summarized the transcript with gemini 1.5 pro


# Additional References

## Port of makemore using Jax/Flax

https://www.kaggle.com/code/shaochuanwang/makemore-jax-transformer
