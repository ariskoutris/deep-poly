import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)

DTYPE = torch.float32

class DpConstraints:
    def __init__(self, lr: torch.Tensor, ur: torch.Tensor, lo: torch.Tensor, uo: torch.Tensor):
        self.lr = lr
        self.ur = ur
        self.lo = lo
        self.uo = uo
        assert self.lr.shape == self.ur.shape
        assert self.lo.shape == self.uo.shape

    def __repr__(self):
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
    
    @property
    def shape(self):
        assert self.lb.shape == self.ub.shape
        return self.lb.shape
    
    def get_loss_tensor(self, y):
        target = torch.tensor(y).view(1)
        tensor = self.ub.clone().flatten()
        tensor[target] = self.lb.flatten()[target].clone()
        return tensor[None, :]
    
    def get_certification_distance(self, y=None):
        if y is None:
            return self.lb.flatten().min().item()
        else:
            target = torch.tensor(y).view(1)
            return (self.lb.flatten()[target] - self.ub.flatten().max()).min().item()

def get_input_bounds(x: torch.Tensor, eps: float, min_val=0, max_val=1):
    lb = (x - eps).to(DTYPE)
    lb.clamp_(min=min_val, max=max_val)

    ub = (x + eps).to(DTYPE)
    ub.clamp_(min=min_val, max=max_val)

    return DpBounds(lb, ub)

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

def bounds_mul_constraints(constraints : DpConstraints, bounds : DpBounds) -> DpBounds:
    lr_pos = torch.relu(constraints.lr)
    lr_neg = -torch.relu(-constraints.lr)
    ur_pos = torch.relu(constraints.ur)
    ur_neg = -torch.relu(-constraints.ur)

    lb = bounds.lb.flatten() @ lr_pos + bounds.ub.flatten() @ lr_neg + constraints.lo
    ub = bounds.ub.flatten() @ ur_pos + bounds.lb.flatten() @ ur_neg + constraints.uo

    return DpBounds(lb, ub)

def check_postcondition(y, bounds: DpBounds) -> bool:
    try:
        target = y.item()
    except AttributeError:
        target = y

    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    target_lb = lb[target].item()

    out = True
    for i in range(ub.shape[0]):
        if i != target and ub[i] >= target_lb:
            out = False
    
    logger.info(f'Certification Distance: {bounds.get_certification_distance(y)}\n')
    return out

def check_postcondition_le(bounds: DpBounds) -> bool:
    lb = bounds.lb.flatten()
    logger.info(f'Certification Distance: {bounds.get_certification_distance()}\n')
    return lb.min() >= 0

def log_layer_bounds(logger, layer, message):
    logger.debug(message)
    logger.debug(f'lb: shape [{layer.bounds.lb.shape}], min: {layer.bounds.lb.min()}, max: {layer.bounds.lb.max()}')
    logger.debug(f'ub: shape [{layer.bounds.ub.shape}], min: {layer.bounds.ub.min()}, max: {layer.bounds.ub.max()}\n')
    
class AlphaParams:
    def __init__(self, value: torch.Tensor, lb, ub):
        self.value = value
        self.lb = lb
        self.ub = ub

def init_alphas(model, inp_shape, dp_layers=None) -> list[torch.Tensor]:
    x_dummy = torch.zeros(inp_shape, dtype=DTYPE)
    prev = x_dummy.clone()
    params_dict = {}
    for i, layer in enumerate(model):
        next = layer(prev)
        if isinstance(layer, (nn.ReLU, nn.LeakyReLU)):
            input_shape = prev.shape
            neg_slope = layer.negative_slope if isinstance(layer, nn.LeakyReLU) else 0.0
            lb, ub = min(1.0, neg_slope), max(1.0, neg_slope)
            if dp_layers:
                lb_in, ub_in = dp_layers[i].bounds.lb, dp_layers[i].bounds.ub
                initial_alphas = torch.where(-lb_in < ub_in, 1.0, neg_slope).to(DTYPE)
                initial_alphas = nn.Parameter(initial_alphas)
            else:
                initial_alphas = nn.Parameter(lb + (ub - lb) * (torch.rand(input_shape, dtype=DTYPE)))
            params_dict[i] = AlphaParams(value=initial_alphas, lb=lb, ub=ub)
        prev = next
    return params_dict  

def assign_alphas_to_relus(dp_layers, alphas):
    for i, layer in enumerate(dp_layers):
        if (i-1) in alphas:
            layer.set_alphas(alphas[i - 1].value)
    return dp_layers

