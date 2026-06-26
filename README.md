# Image Segmentation with CNN and Stochastic Optimization

Comparative study of stochastic optimization algorithms for image segmentation using Convolutional Neural Networks in PyTorch.

## Overview

This project implements and compares different optimization algorithms (SGD, Adam, RMSProp) for the task of segmenting synthetic textured images. The goal is to train a CNN to identify two distinct texture regions in an image by learning their statistical properties.

## Problem Statement

Given a grayscale image composed of two regions with different textures, train a neural network to segment and identify the boundary between them. Textures are modeled as stationary Gaussian processes.

## Key Concepts Explored

- **PyTorch Fundamentals**: Tensors, autograd, and neural network architecture design
- **Stochastic Optimization**: Comparison of SGD (with/without momentum), RMSProp, and Adam
- **Hyperparameter Tuning**: Learning rate impact, batch size, and convergence behavior
- **Network Architectures**: Baseline CNN vs U-Net comparison
- **Evaluation Metrics**: Loss curves, segmentation accuracy, error visualization

## Technical Stack

- **Framework**: PyTorch 2.0+
- **Data Generation**: NumPy (Gaussian processes via convolution)
- **Visualization**: Matplotlib
- **Hardware**: GPU recommended (CUDA support included)

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.3+ (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/Ykuzaa/Image-Segmentation-by-Neural-Networks.git
cd Image-Segmentation-by-Neural-Networks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Notebook

The main analysis is in `ImageSegmentation.ipynb`:


### Key Experiments

1. **Data Generation**: Experiment with `sigma1`, `sigma2`, `sigma` parameters to control texture difficulty
2. **Optimizer Comparison**: Compare convergence speed of SGD, Adam, and RMSProp
3. **Learning Rate Analysis**: Test impact of different learning rates on optimization trajectory
4. **Architecture Comparison**: Evaluate CNN baseline vs U-Net for segmentation quality

## Results Summary

### Optimizer Performance

The notebook demonstrates:
- **Adam**: Fast convergence, stable gradients, adaptive learning rates
- **RMSProp**: Intermediate convergence, good for non-stationary problems
- **SGD with Momentum**: Slower but competitive with proper tuning

### Architecture Insights

- **CNN Baseline**: Simpler, faster training, sufficient for synthetic textures
- **U-Net**: Better spatial preservation, useful for preserving boundary details

## Key Findings

- Learning rate selection is critical; too large values cause divergence
- Adam generally converges faster than SGD across different difficulty levels
- Network depth and kernel size impact segmentation quality and training time
- Texture similarity (measured by sigma parameters) directly affects task difficulty

## Reproducibility

To reproduce results:
1. Keep hardware (GPU type) constant for fair comparison
2. Use fixed random seeds for data generation
3. Maintain identical batch sizes and learning rate schedules
4. Run multiple trials to account for stochastic variation

## Dependencies

See `requirements.txt` for complete list:
- torch >= 2.0.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0

## Authors

- **El-Mehdi BOULAALAM** 
- **Matthias Herla**

**Course**: Stochastic Optimization (Prof. Pierre Weiss)  
**Institution**: INSA Toulouse, ModIA Program

## References

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Deep Learning Optimization: https://www.deeplearningbook.org/
- Stochastic Optimization Theory: Classical optimization literature

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, feel free to open an issue on GitHub or contact the authors.

