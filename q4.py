import numpy as np

class Module:
    def forward(self, input):
        raise NotImplementedError("Forward pass not implemented.")

    def backward(self, grad_output):
        raise NotImplementedError("Backward pass not implemented.")


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((out_features, 1))
        self.x = None  # Store input for backward

    def forward(self, x):
        """
        x: (in_features, batch_size)
        Returns: (out_features, batch_size)
        """
        # TODO: Implement forward pass for Linear
        self.x = x
        temp = self.W @ x + self.b 
        return temp   


    def backward(self, grad_output):
        """
        grad_output: (out_features, batch_size)
        Returns: (in_features, batch_size)
        Also computes gradients w.r.t W and b
        """
        # TODO: Implement backward pass for Linear
        batch_size = self.x.shape[1]
        # Gradient w.r.t weights
        self.dW = grad_output @ self.x.T / batch_size
        # Gradient w.r.t bias
        self.db = np.sum(grad_output, axis=1, keepdims=True) / batch_size
        # Gradient w.r.t input
        grad_input = self.W.T @ grad_output
        return grad_input


class ReLU(Module):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # TODO: Implement ReLU activation
        self.mask = x > 0  # store mask for backward
        return x * self.mask

    def backward(self, grad_output):
        # TODO: Implement gradient of ReLU
        return grad_output * self.mask


class CrossEntropy(Module):
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, logits, labels):
        """
        logits: (num_classes, batch_size)
        labels: (num_classes, batch_size) one-hot encoded
        Returns: scalar loss
        """
        # TODO: Implement forward pass for cross-entropy
        self.y_true = labels
        logits_stable = logits - np.max(logits, axis=0, keepdims=True)
        # Softmax: convert logits to probabilities
        exp_scores = np.exp(logits_stable)
        probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
        self.y_pred = probs
        batch_size = logits.shape[1]
        loss = -np.sum(labels * np.log(probs + 1e-12)) / batch_size  
        return loss

        pass

    def backward(self):
        """
        Returns: gradient of loss w.r.t. logits
        """
        # TODO: Implement backward pass for cross-entropy
        batch_size = self.y_pred.shape[1]
        grad_logits = (self.y_pred - self.y_true) / batch_size
        return grad_logits
        pass

