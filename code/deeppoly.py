import torch
import torch.nn as nn
from dp_utils import *
import logging
logger = logging.getLogger(__name__)

class DpInput():
    def __init__(self, bounds: DpBounds):
        self.layer = None
        self.bounds = bounds


class DpLinear():

    def __init__(self, layer : nn.Linear):
        self.layer = layer
        lr = layer.weight.detach()
        ur = layer.weight.detach()
        lo = layer.bias.detach()
        uo = layer.bias.detach()
        self.constraints = DpConstraints(lr, ur, lo, uo)

    def compute_bound(self, bounds: DpBounds):
        curr_c = DpConstraints(self.constraints.lr.t(), self.constraints.ur.t(), self.constraints.lo, self.constraints.uo)
        self.bounds = bounds_mul_constraints(curr_c, bounds)

    def backsub(self, accum_c: DpConstraints):
        return constraints_mul(self.constraints, accum_c)


class DpFlatten():
    def __init__(self, layer : nn.Flatten):
        self.layer = layer

    def compute_constraints(self, bounds: DpBounds):
        pass

    def compute_bound(self, bounds: DpBounds):
        self.input_shape = bounds.lb.shape
        lb = self.layer(bounds.lb)
        ub = self.layer(bounds.ub)
        self.bounds = DpBounds(lb, ub)

    def backsub(self, constraints: DpConstraints):
        lr = constraints.lr.reshape((*self.input_shape, *constraints.lr.shape[1:]))
        ur = constraints.ur.reshape((*self.input_shape, *constraints.ur.shape[1:]))
        return DpConstraints(lr, ur, constraints.lo, constraints.uo)


class DpRelu():
    def __init__(self, layer : nn.ReLU):
        self.layer = layer
        self.bounds = None
        self.slope = None
        self.bias_upper = None
        self.constraints = None

    def compute_constraints(self, bounds: DpBounds):
        self.slope = bounds.ub / (bounds.ub - bounds.lb)
        self.bias_upper = - bounds.lb * self.slope

        #  0 >= ub >= lb
        mask_upper = bounds.ub <= 0
        # ub >= lb >= 0
        mask_lower = bounds.lb >= 0
        # ub >= 0 >= lb
        mask_crossing =  ~(mask_lower | mask_upper)
        assert (mask_crossing & mask_upper & mask_lower == False).all()

        ur = torch.zeros_like(bounds.ub)
        ur[mask_crossing] = self.slope[mask_crossing]
        ur[mask_lower] = 1
        ur[mask_upper] = 0

        uo = torch.zeros_like(bounds.lb)
        uo[mask_crossing] = self.bias_upper[mask_crossing]

        # For now use the x >= 0 constraint for lower relu
        lr = torch.zeros_like(bounds.lb)
        lr[mask_crossing] = torch.where(-bounds.lb < bounds.ub, 1.0, 0.0)[mask_crossing]
        lr[mask_lower] = 1
        lo = torch.zeros_like(bounds.lb)

        self.constraints = DpConstraints(torch.diag(lr.flatten()), torch.diag(ur.flatten()), lo, uo)

    def compute_bound(self, bounds: DpBounds):

        self.compute_constraints(bounds)
        self.bounds = bounds_mul_constraints(self.constraints, bounds)

    def backsub(self, accum_c: DpConstraints):
        return constraints_mul(self.constraints, accum_c)


class DpConv():
    def __init__(self, layer : nn.Conv2d):
        self.layer = layer

    def compute_bound(self, bounds: DpBounds):
        return NotImplementedError()


class DiffLayer():
    def __init__(self, target: int, n_classes: int):
        self.target = target
        self.n_classes = n_classes

    def compute_constraints(self, bounds: DpBounds):
        I = [i for i in range(self.n_classes) if i != self.target]
        C = torch.eye(self.n_classes, dtype=torch.float)[self.target].unsqueeze(dim=0) - torch.eye(self.n_classes, dtype=torch.float)[I]
        lr, ur = C, C
        lo = torch.zeros_like(lr[:, 0])
        uo = torch.zeros_like(ur[:, 0])
        self.constraints = DpConstraints(lr, ur, lo, uo)

    def compute_bound(self, bounds: DpBounds):
        self.compute_constraints(bounds)

    def backsub(self, accum_c: DpConstraints):
        return constraints_mul(self.constraints, accum_c)


def deeppoly_backsub(dp_layers):
    constraints_acc = dp_layers[-1].constraints.copy()
    constraints_acc.ur = constraints_acc.ur.t()
    constraints_acc.lr = constraints_acc.lr.t()

    logger.debug(f'Last Layer:\n{str(constraints_acc)}')
    for i, layer in enumerate(reversed(dp_layers[:-1])):
        # If this is the first layer, then just multiply with input bounds
        if isinstance(layer, DpInput):
            ur = constraints_acc.ur.flatten(1, -2)
            lr = constraints_acc.lr.flatten(1, -2)
            lb_in = layer.bounds.lb.flatten(1)
            ub_in = layer.bounds.ub.flatten(1)
            b_curr = bounds_mul_constraints(DpConstraints(lr, ur, constraints_acc.lo, constraints_acc.uo), DpBounds(lb_in, ub_in))
            lb = b_curr.lb.squeeze(0)
            ub = b_curr.ub.squeeze(0)
        else:
            constraints_acc = layer.backsub(constraints_acc)

        logger.debug(f'Layer {len(dp_layers) - 2 - i} [{layer.layer}]:')
        logger.debug(str(constraints_acc))
    return DpBounds(lb, ub)

def propagate_sample(model, x, eps, le_layer=None, min_val=0, max_val=1):
    bounds = get_input_bounds(x, eps, min_val, max_val)
    input_layer = DpInput(bounds)
    dp_layers = [input_layer]

    logger.debug(f'Input Layer')
    logger.debug(f'lb: shape [{input_layer.bounds.lb.shape}], min: {input_layer.bounds.lb.min()}, max: {input_layer.bounds.lb.max()}')
    logger.debug(f'ub: shape [{input_layer.bounds.ub.shape}], min: {input_layer.bounds.ub.min()}, max: {input_layer.bounds.ub.max()}')
    for i, layer in enumerate(model):
        dp_layer = None
        if isinstance(layer, nn.Flatten):
            dp_layer = DpFlatten(layer)
        elif isinstance(layer, nn.Linear):
            dp_layer = DpLinear(layer)
        elif isinstance(layer, nn.ReLU):
            dp_layer = DpRelu(layer)

        dp_layer.compute_bound(dp_layers[-1].bounds)
        dp_layers.append(dp_layer)
        if not isinstance(layer, nn.Flatten):
            dp_layer.bounds = deeppoly_backsub(dp_layers)

        logger.debug(f'Layer {i + 1} {layer}')
        logger.debug(f'lb: shape [{dp_layer.bounds.lb.shape}], min: {dp_layer.bounds.lb.min()}, max: {dp_layer.bounds.lb.max()}')
        logger.debug(f'ub: shape [{dp_layer.bounds.ub.shape}], min: {dp_layer.bounds.ub.min()}, max: {dp_layer.bounds.ub.max()}')

    if le_layer is not None:
        le_layer.compute_bound(dp_layers[-1].bounds)
        dp_layers.append(le_layer)
        le_layer.bounds = deeppoly_backsub(dp_layers)

        logger.debug(f'Layer {len(dp_layers) - 1} [{le_layer}]:')
        logger.debug(f'lb: shape [{le_layer.bounds.lb.shape}], min: {le_layer.bounds.lb.min()}, max: {le_layer.bounds.lb.max()}')
        logger.debug(f'ub: shape [{le_layer.bounds.ub.shape}], min: {le_layer.bounds.ub.min()}, max: {le_layer.bounds.ub.max()}')

    return dp_layers

def certify_sample(model, x, y, eps, use_le=True) -> bool:

    if x.dim() == 3:
        x = x.unsqueeze(0)

    if use_le:
        n_classes = model[-1].out_features
        le_layer = DiffLayer(y, n_classes)
        dp_layers = propagate_sample(model, x, eps, le_layer)
    else:
        dp_layers = propagate_sample(model, x, eps)

    bounds = dp_layers[-1].bounds

    if use_le:
        return check_postcondition_le(bounds)
    else:
        check_postcondition(y, bounds)
