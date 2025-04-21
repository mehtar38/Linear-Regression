# Linear Regression Implementation from Scratch

Implementing linear regression with gradient descent, first from scratch then using PyTorch.

## Project Overview
Two key implementations:
1. **Linear Regression from Scratch**
   - Manual gradient descent implementation
   - Regularization techniques
   - Custom loss computation

2. **PyTorch Implementation**  
   - Built using `torch.nn.Linear`  
   - Automatic differentiation  
   - SGD optimizer with learning rate tuning  
   - Training visualization  

## Key Components
### Problem 1: From Scratch
- NumPy-based implementation
- Polynomial feature expansion
- Custom gradient descent algorithm
- L2 regularization support

### Problem 2: PyTorch Version
- Tensor operations and autograd
- MSE loss with `torch.nn.MSELoss()`
- Training visualization:
  ```python
  plt.plot(range(epochs), loss_record)  # Loss curve
  plt.plot(xs, ys.detach().numpy())    # Prediction function
