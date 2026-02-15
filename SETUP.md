# Setup Instructions

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/image-segmentation-cnn-optim.git
cd image-segmentation-cnn-optim

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter ImageSegmentation.ipynb
```


## Using the Package

### Import from Source

```python
from src.data_generation import generate_data
from src.models import SimpleCNN, UNet
from src.training import create_optimizer, train_step

# Generate synthetic data
images, masks = generate_data(
    batch_size=32,
    size_image=64,
    sigma1=0.5,
    sigma2=0.5,
    sigma=0.1
)

# Create model
model = SimpleCNN(num_channels=16)

# Create optimizer
optimizer = create_optimizer(
    model, 
    optimizer_name='Adam', 
    learning_rate=0.001
)

# Training step
loss = train_step(model, images, masks, optimizer, criterion, device)
```

## Key Hyperparameters

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `sigma1`, `sigma2` | Texture difficulty | [0.2, 2.0] |
| `sigma` | Noise level | [0.05, 0.2] |
| `learning_rate` | Optimizer step size | [1e-4, 1e-2] |
| `batch_size` | Gradient estimation | [8, 128] |
| `num_channels` | Network capacity | [8, 64] |

## Hardware Requirements

- **Minimum**: CPU with 4GB RAM
- **Recommended**: NVIDIA GPU with CUDA 11.3+
- **Tested on**: NVIDIA A6000 (48GB VRAM)

## Troubleshooting

### GPU Not Detected

```python
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))
```

### Out of Memory

- Reduce `batch_size`
- Reduce `image_size`
- Use smaller `num_channels` in model

### Convergence Issues

- Adjust `learning_rate`
- Check `sigma1`, `sigma2` (task difficulty)
- Try different optimizers

## Next Steps

1. **Experiment**: Modify texture parameters to change problem difficulty
2. **Compare**: Run multiple optimizers and analyze convergence
3. **Extend**: Add custom losses, regularization, or architectures
4. **Deploy**: Convert trained models to production format

## References

- PyTorch Docs: https://pytorch.org/docs/
- Course Materials: Prof. Pierre Weiss, Stochastic Optimization
- Datasets: Synthetic textures (Gaussian processes)

---

**Last Updated**: February 2025  
**Python Version**: 3.8+  
**PyTorch Version**: 2.0+
