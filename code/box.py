import torch
import torch.nn as nn
from modules import Normalize, View

import logging
logger = logging.getLogger(__name__)

def get_C(y_batch, n_class=10):
    def _get_C(n_class, y):
        I = [i for i in range(n_class) if i != y]
        return torch.eye(n_class, dtype=torch.float32, device=y_batch.device)[y].unsqueeze(dim=0) - torch.eye(n_class, dtype=torch.float32, device=y_batch.device)[I]
    return torch.stack([_get_C(n_class,y) for y in y_batch], dim=0)

class AbstractBox:

    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        assert lb.shape == ub.shape
        assert (lb > ub).sum() == 0
        self.lb = lb
        self.ub = ub

    @staticmethod
    def construct_initial_box(x: torch.Tensor, eps: float, min_val: float = 0, max_val: float = 1) -> 'AbstractBox':
        lb = x - eps
        lb.clamp_(min=min_val, max=max_val)

        ub = x + eps
        ub.clamp_(min=min_val, max=max_val)

        return AbstractBox(lb, ub)

    def propagate_normalize(self, normalize: Normalize) -> 'AbstractBox':
        # Follows from the rules in the lecture.
        lb = normalize(self.lb)
        ub = normalize(self.ub)
        return AbstractBox(lb, ub)

    def propagate_view(self, view: View) -> 'AbstractBox':
        lb = view(self.lb)
        ub = view(self.ub)
        return AbstractBox(lb, ub)
    
    def propagate_flatten(self, flatten: nn.Flatten) -> 'AbstractBox':
        lb = flatten(self.lb)
        ub = flatten(self.ub)
        return AbstractBox(lb, ub)

    def propagate_linear(self, fc: nn.Linear) -> 'AbstractBox':
        assert self.lb.shape == self.ub.shape
        center = (self.lb + self.ub) / 2
        eps = (self.ub - self.lb) / 2  # width

        center_out = center@fc.weight.t()
        if fc.bias is not None:
            center_out = center_out + fc.bias
        eps_out = eps@fc.weight.abs().t()
        lb = center_out - eps_out
        ub = center_out + eps_out
        return AbstractBox(lb, ub)

    def propagate_linear_LE(self, fc: nn.Linear, C) -> 'AbstractBox':
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
        return AbstractBox(lb, ub)

    def propagate_relu(self, relu: nn.ReLU) -> 'AbstractBox':
        # Follows from the rules in the lecture.
        lb = relu(self.lb)
        ub = relu(self.ub)
        return AbstractBox(lb, ub)

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

def propagate_sample(model, x, eps, min_val=0, max_val=1) -> AbstractBox:
    box = AbstractBox.construct_initial_box(x, eps, min_val, max_val)
    logger.debug(f'Input Layer: {box.lb} {box.ub}')
    for i, layer in enumerate(model):
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
        logger.debug(f'Layer {i}: {box.lb.detach()} {box.ub.detach()} || [{type(layer)}]')
    return box

def propagate_sample_LE(model, x, eps, C=None, min_val=0, max_val=1) -> AbstractBox:
    box = AbstractBox.construct_initial_box(x, eps, min_val=min_val, max_val=max_val)
    logger.debug(f'Input Layer: {box.lb} {box.ub}')
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
        logger.debug(f'Layer {i}: {box.lb.detach()} {box.ub.detach()} || [{type(layer)}]')
    return box

def certify_sample(model, x, y, eps) -> bool:
    box = propagate_sample(model, x, eps)
    return box.check_postcondition(y)

def certify_sample_LE(model, x, y, eps) -> bool:
    C = get_C(y)
    box = propagate_sample_LE(model, x, eps, C=C)
    return (box.lb > 0).all()

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(
        description="Neural Network Verification Example"
    )
    parser.add_argument(
        "--weight",
        type=float,
        required=False,
        help="Neural network weight parameter value",
    )
    
    args = parser.parse_args()
    weight = args.weight if args.weight is not None else 2.0

    def simulate(w):

        flatten = nn.Flatten()
        linear1 = nn.Linear(1, 2)
        linear1.weight.data = torch.tensor([[1], [1]], dtype=torch.float)
        linear1.bias.data = torch.tensor([w, w], dtype=torch.float)
        relu1 = nn.ReLU()
        linear2 = nn.Linear(2, 2)
        linear2.weight.data = torch.tensor([[1, 0], [-1, 2]], dtype=torch.float)
        linear2.bias.data = torch.tensor([1, 0], dtype=torch.float)
        model = nn.Sequential(flatten, linear1, relu1, linear2)
        model.eval()

        x = torch.tensor([[[0]]])
        eps = 1.0

        print(w)
        box = propagate_sample(model, x, eps, -1, 1)

        lb = box.lb.flatten()
        ub = box.ub.flatten()
        return ub[1].item()

    import numpy as np
    import matplotlib.pyplot as plt

    w_vals = np.linspace(-3, 5, 50)
    ub_vals = []
    for w in w_vals:
        ub_vals.append(simulate(w))
    plt.plot(w_vals, ub_vals)
    plt.show()
    
    #simulate(weight)