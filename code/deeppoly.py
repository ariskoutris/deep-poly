import torch
import torch.nn as nn

def get_C(y_batch, n_class=10):
    def _get_C(n_class, y):
        I = [i for i in range(n_class) if i != y]
        return torch.eye(n_class, dtype=torch.float32, device=y_batch.device)[y].unsqueeze(dim=0) - torch.eye(n_class, dtype=torch.float32, device=y_batch.device)[I]
    return torch.stack([_get_C(n_class,y) for y in y_batch], dim=0)

deeppoly_layers = []

class DpConstraints:
    def __init__(self, lr: torch.Tensor, ur: torch.Tensor, lo: torch.Tensor, uo: torch.Tensor):
        self.lr = lr
        self.ur = ur
        self.lo = lo
        self.uo = uo
        assert self.lr.shape == self.ur.shape
        assert self.lo.shape == self.uo.shape


class DpBounds:
    def __init(self, lb: torch.Tensor, ub: torch.Tensor):
        self.lb = lb
        self.ub = ub
        assert self.lb.shape == self.ub.shape
        assert (self.lb > self.ub).sum() == 0


class DpLinear():
    def __init__(self, layer_pos : int, fc : nn.Linear):
        self.layer = layer_pos
        lr = fc.weight.detach()
        ur = fc.weight.detach()
        lo = fc.bias.detach()
        uo = fc.bias.detach()
        self.constraints = DpConstraints(lr, ur, lo, uo)

    # Call on forward pass of bounds
    def compute_bound(self, bounds: DpBounds):
        self.dpl = DpBounds(...)
        raise NotImplementedError()


class DpFlatten():
    def __init__():
        return NotImplementedError()
    def propagate():
        return NotImplementedError()


class DpRelu():
    def __init__():
        return NotImplementedError()
    def propagate():
        return NotImplementedError()


class DpConv():
    def __init__():
        return NotImplementedError()
    def propagate():
        return NotImplementedError()


def check_postcondition(y, bounds: DpBounds) -> bool:
    try:
        target = y.item()
    except AttributeError:
        target = y
    target_lb = bounds.lb[0][target].item()
    for i in range(bounds.ub.shape[-1]):
        if i != target and bounds.ub[0][i] >= target_lb:
            return False
    return True

# Function to get the 0th deepoly object with the initial bounds
# and the upper + lower identity constra
def get_input_bounds(x: torch.Tensor, eps: float) -> 'DeepPoly':
    lb = x - eps
    lb.clamp_(min=0, max=1)

    ub = x + eps
    ub.clamp_(min=0, max=1)

    return DpBounds(lb, ub)


def deeppoly_backsub():
    raise NotImplementedError()

def propagate_sample(model, x, eps) -> DeepPoly:

    for layer in model:
        if isinstance(layer, nn.Flatten):
            pass
        elif isinstance(layer, nn.Linear):
            pass
        elif isinstance(layer, nn.ReLU):
            pass

def certify_sample(model, x, y, eps) -> bool:
    box = propagate_sample(model, x, eps)
    return box.check_postcondition(y)
