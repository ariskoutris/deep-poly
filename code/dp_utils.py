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
    lb = (x - eps).to(torch.double)
    lb.clamp_(min=min_val, max=max_val)

    ub = (x + eps).to(torch.double)
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

# Multiply bounds with constraints, use during forward pass
def bounds_mul_constraints(constraints : DpConstraints, bounds : DpBounds) -> DpBounds:
    lr_pos = torch.relu(constraints.lr)
    lr_neg = -torch.relu(-constraints.lr)
    ur_pos = torch.relu(constraints.ur)
    ur_neg = -torch.relu(-constraints.ur)

    lb = bounds.lb.flatten() @ lr_pos + bounds.ub.flatten() @ lr_neg + constraints.lo
    ub = bounds.ub.flatten() @ ur_pos + bounds.lb.flatten() @ ur_neg + constraints.uo

    return DpBounds(lb, ub)

def log_layer_bounds(logger, layer, message):
    logger.debug(message)
    logger.debug(f'lb: shape [{layer.bounds.lb.shape}], min: {layer.bounds.lb.min()}, max: {layer.bounds.lb.max()}')
    logger.debug(f'ub: shape [{layer.bounds.ub.shape}], min: {layer.bounds.ub.min()}, max: {layer.bounds.ub.max()}\n')
    
class AlphaParams:
    def __init__(self, value: torch.Tensor, lb, ub):
        self.value = value
        self.lb = lb
        self.ub = ub


def init_alphas(model, inp_shape) -> list[torch.Tensor]:
    """
    Only linear layer can provide some guarantees of out_shape 
    which is why we need to calculate the out_shapes of other layers
    """
    inp_shape = list(inp_shape)
    alphas = {}
    has_relus = False

    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            assert inp_shape[-1] == layer.in_features
            out_shape = inp_shape.copy()
            out_shape[-1] = layer.out_features
        elif isinstance(layer, nn.Flatten):
            #TODO find a better way to handle these
            start_dim = layer.start_dim
            # Here the end dim is inclusive (usually in python end_dim is not included)
            end_dim = len(inp_shape) if layer.end_dim == -1 else layer.end_dim + 1

            out_shape = inp_shape[:start_dim]
            out_shape = out_shape + torch.prod(torch.tensor(inp_shape[start_dim:end_dim])).view(1).tolist()
            out_shape = out_shape + inp_shape[end_dim:]
        elif isinstance(layer, nn.ReLU):
            has_relus = True
            out_shape = inp_shape.copy()
            lb, ub = 0.0, 1.0
            initial_alphas = nn.Parameter(lb + (ub - lb) * (torch.rand(inp_shape, dtype=torch.double)))
            alphas[i] = AlphaParams(value=initial_alphas, lb=lb, ub=ub)
        elif isinstance(layer, nn.LeakyReLU):
            has_relus = True
            out_shape = inp_shape.copy()
            lb = min(1.0, layer.negative_slope)
            ub = max(1.0, layer.negative_slope)
            initial_alphas = nn.Parameter(lb + (ub - lb) * (torch.rand(inp_shape, dtype=torch.double)))
            alphas[i] = AlphaParams(value=initial_alphas, lb=lb, ub=ub)
        elif isinstance(layer, nn.Conv2d):
            C, H, W = inp_shape[-3:]
            assert C == layer.in_channels
            Hout = (H + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
            Wout = (W + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1
            out_shape = torch.tensor([*inp_shape[:-3], layer.out_channels, Hout, Wout]).tolist()
        else:
            raise NotImplementedError(f'Unsupported layer type: {type(layer)}')
        inp_shape = out_shape

    return alphas if has_relus else None


if __name__ == "__main__":
    model0 = nn.Sequential(
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(in_features=784, out_features=50, bias=True),
        nn.Linear(in_features=50, out_features=50, bias=True),
        nn.Linear(in_features=50, out_features=50, bias=True),
        nn.Linear(in_features=50, out_features=10, bias=True)
    )
    input0 = torch.randn((1, 28, 28))

    model1 = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(in_features=392, out_features=50, bias=True),
        nn.Linear(in_features=50, out_features=10, bias=True)
    )
    input1 = torch.randn((1, 1, 28, 28))

    model2 = nn.Sequential(
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(in_features=784, out_features=100, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=100, out_features=100, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=100, out_features=10, bias=True)
    )
    input2 = torch.randn((1, 1, 28, 28))

    model3 = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        nn.ReLU(),
        nn.Conv2d(16, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(in_features=4096, out_features=100, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=100, out_features=100, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=100, out_features=10, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=10, out_features=10, bias=True)
    )
    input3 = torch.randn((1, 3, 32, 32))

    model = model3
    input = input3
    init_alphas(model, input.shape)
    for i, layer in enumerate(model):
        inp_shape = input.shape
        input = layer(input)
        out_shape = input.shape
        print(f"C: inp {i} = {inp_shape} \tout {i} = {out_shape}")
    

