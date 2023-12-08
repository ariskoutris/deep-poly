import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

class DpConstraints:
    def __init__(self, lr: torch.Tensor, ur: torch.Tensor, lo: torch.Tensor, uo: torch.Tensor):
        self.lr = lr
        self.ur = ur
        self.lo = lo
        self.uo = uo
        assert self.lr.shape == self.ur.shape
        assert self.lo.shape == self.uo.shape

    def __repr__(self):
        pass

    def __str__(self):
        out = f"lr: shape [{self.lr.shape}], min: {self.lr.min()}, max: {self.lr.max()}\n"
        out += f"ur: shape [{self.ur.shape}], min: {self.ur.min()}, max: {self.ur.max()}\n"
        out += f"lo: shape [{self.lo.shape}], min: {self.lo.min()}, max: {self.lo.max()}\n"
        out += f"uo: shape [{self.uo.shape}], min: {self.uo.min()}, max: {self.uo.max()}\n"
        return out

    def copy(self):
        return DpConstraints(self.lr.clone(), self.ur.clone(), self.lo.clone(), self.uo.clone())

class DpBounds:
    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        self.lb = lb
        self.ub = ub
        assert self.lb.shape == self.ub.shape
        assert (self.lb > self.ub).sum() == 0

    def __repr__(self):
        return f"lb: {self.lb}\tub: {self.ub}"

def get_input_bounds(x: torch.Tensor, eps: float, min_val=0, max_val=1):
    lb = (x - eps).to(torch.float)
    lb.clamp_(min=min_val, max=max_val)

    ub = (x + eps).to(torch.float)
    ub.clamp_(min=min_val, max=max_val)

    return DpBounds(lb, ub)

# Use only during back propagation
def constraints_mul(curr_c : DpConstraints, accum_c : DpConstraints) -> DpConstraints:

    accum_c_lr_pos = torch.relu(accum_c.lr)
    accum_c_lr_neg = -torch.relu(-accum_c.lr)
    accum_c_ur_pos = torch.relu(accum_c.ur)
    accum_c_ur_neg = -torch.relu(-accum_c.ur)

    lr =  curr_c.lr.t() @ accum_c_lr_pos +  curr_c.ur.t() @ accum_c_lr_neg
    ur =  curr_c.ur.t() @ accum_c_ur_pos +  curr_c.lr.t() @ accum_c_ur_neg
    lo = curr_c.lo @ accum_c_lr_pos + curr_c.uo @ accum_c_lr_neg
    uo = curr_c.uo @ accum_c_ur_pos + curr_c.lo @ accum_c_ur_neg
    uo = uo + accum_c.uo
    lo = lo + accum_c.lo

    return DpConstraints(lr, ur, lo, uo)

def check_postcondition(y, bounds: DpBounds) -> bool:
    try:
        target = y.item()
    except AttributeError:
        target = y

    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    target_lb = lb[target].item()
    min_interval = ub.max() - lb.min()

    out = True
    for i in range(ub.shape[0]):
        if i != target and ub[i] >= target_lb:
            out = False
        if i != target:
            min_interval = min(min_interval, target_lb - ub[i])
    logger.info(f'Certification Distance: {min_interval}\n')
    return out

def check_postcondition_le(bounds: DpBounds) -> bool:
    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()
    logger.info(f'Certification Distance: {lb.min()}\n')
    return lb.min() >= 0

# Multiply bounds with constraints, use during forward pass
def bounds_mul_constraints(constraints : DpConstraints, bounds : DpBounds) -> DpBounds:
    lr_pos = torch.relu(constraints.lr)
    lr_neg = -torch.relu(-constraints.lr)
    ur_pos = torch.relu(constraints.ur)
    ur_neg = -torch.relu(-constraints.ur)

    lb = bounds.lb @ lr_pos + bounds.ub @ lr_neg + constraints.lo
    ub = bounds.ub @ ur_pos + bounds.lb @ ur_neg + constraints.uo

    return DpBounds(lb, ub)

def log_layer_bounds(logger, layer, message):
    logger.debug(message)
    logger.debug(f'lb: shape [{layer.bounds.lb.shape}], min: {layer.bounds.lb.min()}, max: {layer.bounds.lb.max()}')
    logger.debug(f'ub: shape [{layer.bounds.ub.shape}], min: {layer.bounds.ub.min()}, max: {layer.bounds.ub.max()}\n')
