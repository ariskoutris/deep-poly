import torch
import torch.nn as nn
from modules import Normalize, View

def get_C(y_batch, n_class=10):
    def _get_C(n_class, y):
        I = [i for i in range(n_class) if i != y]
        return torch.eye(n_class, dtype=torch.float32, device=y_batch.device)[y].unsqueeze(dim=0) - torch.eye(n_class, dtype=torch.float32, device=y_batch.device)[I]
    return torch.stack([_get_C(n_class,y) for y in y_batch], dim=0)

class DeepPoly:

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert (lb > ub).sum() == 0
        self.lb = lb
        self.ub = ub

    @staticmethod
    def construct_initial_box(x: torch.Tensor, eps: float) -> 'DeepPoly':
        lb = x - eps
        lb.clamp_(min=0, max=1)

        ub = x + eps
        ub.clamp_(min=0, max=1)

        return DeepPoly(lb, ub)

    def propagate_normalize(self, normalize: Normalize) -> 'DeepPoly':
        # Follows from the rules in the lecture.
        lb = normalize(self.lb)
        ub = normalize(self.ub)
        return DeepPoly(lb, ub)

    def propagate_view(self, view: View) -> 'DeepPoly':
        lb = view(self.lb)
        ub = view(self.ub)
        return DeepPoly(lb, ub)
    
    def propagate_flatten(self, flatten: nn.Flatten) -> 'DeepPoly':
        lb = flatten(self.lb)
        ub = flatten(self.ub)
        return DeepPoly(lb, ub)

    def propagate_linear(self, fc: nn.Linear) -> 'DeepPoly':
        assert self.lb.shape == self.ub.shape
        center = (self.lb + self.ub) / 2
        eps = (self.ub - self.lb) / 2  # width

        center_out = center@fc.weight.t()
        if fc.bias is not None:
            center_out = center_out + fc.bias
        eps_out = eps@fc.weight.abs().t()
        lb = center_out - eps_out
        ub = center_out + eps_out
        return DeepPoly(lb, ub)

    def propagate_linear_LE(self, fc: nn.Linear, C) -> 'DeepPoly':
        assert self.lb.shape == self.ub.shape
        center = (self.lb + self.ub) / 2
        eps = (self.ub - self.lb) / 2


        CW = torch.bmm(C, fc.weight.unsqueeze(0).repeat(C.shape[0], 1, 1))
        center_out = torch.bmm(CW, center.unsqueeze(2)).squeeze(dim=2)
        # if fc.bias is not None:
        #     center_out = center_out +  torch.bmm(C, fc.bias.unsqueeze(0).unsqueeze(2).repeat(C.shape[0], 1, 1)).squeeze(dim=2)
        eps_out = torch.bmm(CW.abs(), eps.unsqueeze(2)).squeeze(dim=2)
        lb = center_out - eps_out
        ub = center_out + eps_out
        return DeepPoly(lb, ub)

    def propagate_relu(self, relu: nn.ReLU) -> 'DeepPoly':
        # Follows from the rules in the lecture.
        lb = relu(self.lb)
        ub = relu(self.ub)
        return DeepPoly(lb, ub)

    def check_postcondition(self, y) -> bool:
        try:
            target = y.item()
        except AttributeError:
            target = y 
        target_lb = self.lb[0][target].item()
        for i in range(self.ub.shape[-1]):
            if i != target and self.ub[0][i] >= target_lb:
                return False
        return True

def propagate_sample(model, x, eps) -> DeepPoly:
    box = DeepPoly.construct_initial_box(x, eps)
    for layer in model:
        if isinstance(layer, Normalize):
            box = box.propagate_normalize(layer)
        elif isinstance(layer, View):
            box = box.propagate_view(layer)
        elif isinstance(layer, nn.Flatten):
            box = box.propagate_flatten(layer)
        elif isinstance(layer, nn.Linear):
            box = box.propagate_linear(layer)
        elif isinstance(layer, nn.ReLU):
            box = box.propagate_relu(layer)
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
    return box

def propagate_sample_LE(model, x, eps, C=None) -> DeepPoly:
    box = DeepPoly.construct_initial_box(x, eps)
    for i, layer in enumerate(model):
        last = i == len(model) - 1
        if isinstance(layer, nn.Linear):
            if last and C is not None:
                box = box.propagate_linear_LE(layer, C=C)
            else:
                box = box.propagate_linear(layer)
        elif last and C is not None:
            assert False, "If last-layer trick is used last layer must be linear"
        elif isinstance(layer, Normalize):
            box = box.propagate_normalize(layer)
        elif isinstance(layer, View):
            box = box.propagate_view(layer)
        elif isinstance(layer, nn.ReLU):
            box = box.propagate_relu(layer)
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
    return box

def certify_sample(model, x, y, eps) -> bool:
    box = propagate_sample(model, x, eps)
    return box.check_postcondition(y)

def certify_sample_LE(model, x, y, eps) -> bool:
    C = get_C(y)
    box = propagate_sample_LE(model, x, eps, C=C)
    return (box.lb > 0).all()
