"""
Neural network activation hooks for extracting intermediate layer outputs.
Useful for getting point clouds (activations) from deep learning models.
"""

from functools import partial
from typing import Any, Dict, List, Optional

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class ActivationHook:
    """
    Simple activation hook for extracting layer outputs from PyTorch models.

    Usage:
        model = YourModel()
        hook = ActivationHook(model)
        hooks = hook.register_hooks(['layer1', 'layer2'])  # specify layers

        # Run inference
        output = model(input_data)

        # Get activations
        activations = hook.get_activations()

        # Clean up
        hook.remove_hooks(hooks)
    """

    def __init__(self, model):
        """
        Initialize hook manager.

        Args:
            model: PyTorch model to extract activations from
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ActivationHook. Install with: pip install torch"
            )

        self.model = model
        self.activations: Dict[str, Any] = {}

    def register_hooks(self, layer_names: Optional[List[str]] = None) -> List:
        """
        Register forward hooks on specified layers.

        Args:
            layer_names: List of layer names to hook. If None, hooks all layers.

        Returns:
            List of hook handles for cleanup
        """
        hooks = []

        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                hook = module.register_forward_hook(
                    partial(self._hook_function, name=name)
                )
                hooks.append(hook)

        return hooks

    def _hook_function(self, module, input: Any, output: Any, name: str):
        """
        Hook function to capture layer outputs.

        Args:
            module: The module being hooked
            input: Input to the module
            output: Output from the module
            name: Name of the layer
        """
        if isinstance(output, (list, tuple)):
            # Handle multiple outputs (e.g., transformer layers)
            self.activations[name] = [out.detach().cpu() for out in output]
        elif TORCH_AVAILABLE and isinstance(output, torch.Tensor):
            # Single tensor output
            self.activations[name] = output.detach().cpu()
        else:
            print(f"Warning: Unsupported output type at layer {name}: {type(output)}")

    def get_activations(self) -> Dict[str, Any]:
        """
        Get captured activations.

        Returns:
            Dictionary mapping layer names to activation tensors
        """
        return self.activations

    def clear_activations(self):
        """Clear stored activations to free memory."""
        self.activations.clear()

    def remove_hooks(self, hooks: List):
        """
        Remove all registered hooks.

        Args:
            hooks: List of hook handles returned by register_hooks()
        """
        for hook in hooks:
            hook.remove()

    def get_layer_names(self) -> List[str]:
        """
        Get all available layer names in the model.

        Returns:
            List of layer names
        """
        return [name for name, _ in self.model.named_modules()]


# For backwards compatibility
Probe = ActivationHook
