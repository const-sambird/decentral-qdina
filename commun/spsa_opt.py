import torch
from commun.qiskit_spsa import SPSA
import numpy as np

class SPSAOptimiser:
    """
    A PyTorch-compatible wrapper for Qiskit's SPSA optimizer.
    Supports both CPU and GPU tensors and uses SPSA.minimize().
    """
    def __init__(self, model, lr=1e-1, spsa=None, perturbation=0.01, maxiter=100, device=None):
        self.model = model
        self.lr = lr
        self.spsa = spsa or SPSA(
            maxiter=maxiter,
            learning_rate=1e-1,
            perturbation=1e-1
        )

        # Keep shapes to re-slice flat vectors into parameter tensors.
        self.param_shapes = [p.shape for p in model.parameters()]
        self.num_params = sum(p.numel() for p in model.parameters())
        self.device = device

    def _flatten_params(self):
        """Return a detached copy of all parameters flattened into a 1D tensor (on the model device)."""
        return torch.cat([p.view(-1) for p in self.model.parameters()]).detach().clone()

    def _set_params(self, flat_params):
        """
        Overwrite model parameters with values from a 1D tensor `flat_params`.
        flat_params must be a torch tensor on the same device as the model parameters.
        """
        idx = 0
        for p, shape in zip(self.model.parameters(), self.param_shapes):
            numel = p.numel()
            new_vals = flat_params[idx: idx + numel].view(shape)
            # Copy into p.data to avoid messing with gradients/history.
            p.data.copy_(new_vals)
            idx += numel

    def step(self, loss_fn, *loss_args):
        """
        Perform one SPSA minimization step.

        Args:
            loss_fn: callable that returns a scalar torch.Tensor loss when invoked as loss_fn(*loss_args).
                     The callable should compute loss **on the device** where the model parameters live.
            *loss_args: positional arguments forwarded to loss_fn (e.g. a batch tuple).
        """

        # Get current parameters as a flattened tensor on the model/device.
        theta_torch = self._flatten_params().to(self.device)

        # Convert to NumPy on CPU for SPSA.minimize()
        theta_np = theta_torch.detach().cpu().numpy()

        # Define a CPU-callable objective that SPSA.minimize will call.
        # The objective will receive NumPy arrays and must return a scalar float.
        def eval_loss(params_np):
            # Convert NumPy -> torch tensor on the model device
            params_t = torch.tensor(params_np, dtype=torch.float32, device=self.device)
            # Overwrite model parameters (in-place)
            self._set_params(params_t)
            # Evaluate loss on device
            loss = loss_fn(*loss_args)
            # Ensure we return a Python float
            return float(loss.detach().cpu().numpy())

        # Call the current Qiskit API: minimize(objective, x0=initial_point)
        # The SPSA instance's maxiter controls how many internal iterations occur.
        result = self.spsa.minimize(eval_loss, x0=theta_np)

        # result is an OptimizerResult-like object; final parameters live in result.x
        new_theta = np.asarray(result.x, dtype=np.float32)

        # Copy updated parameters back to the model device and set them
        new_theta_t = torch.tensor(new_theta, dtype=torch.float32, device=self.device)
        max_update_norm = 1.0  # tune this
        update_vec = new_theta_t - theta_torch
        update_norm = torch.norm(update_vec)
        if update_norm > max_update_norm:
            update_vec = update_vec * (max_update_norm / update_norm)
        new_theta_t = theta_torch + update_vec
        self._set_params(new_theta_t)

        # Optionally return the result object for logging/inspection
        return result
