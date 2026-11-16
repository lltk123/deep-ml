import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    # Your code here
    d = x.shape[-1]//2
    scores = x[:,:d] * (x[:,d:] / (1+np.exp(-x[:,d:])))
    return scores