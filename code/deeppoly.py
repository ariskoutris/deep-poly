import torch
import torch.nn as nn

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

class DpBounds:
    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        self.lb = lb
        self.ub = ub
        assert self.lb.shape == self.ub.shape
        assert (self.lb > self.ub).sum() == 0

class DpInput():

    def __init__(self, bounds: DpBounds):
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
        lb, _ = bounded_matmul(bounds.lb, bounds.ub, self.constraints.lr, self.constraints.lo)
        _, ub = bounded_matmul(bounds.lb, bounds.ub, self.constraints.ur, self.constraints.uo)
        self.bounds = DpBounds(lb, ub)

    def backsub(self, constraints: DpConstraints):
        lr, _ = bounded_matmul_alt(self.constraints.lr, self.constraints.ur, constraints.lr)
        _, ur = bounded_matmul_alt(self.constraints.lr, self.constraints.ur, constraints.ur)
        lo, _ = bounded_matmul_alt(self.constraints.lo, self.constraints.uo, constraints.lr)
        _, uo = bounded_matmul_alt(self.constraints.lo, self.constraints.uo, constraints.ur)
        uo = uo + constraints.uo
        lo = lo + constraints.lo
        return DpConstraints(lr, ur, lo, uo)

class DpFlatten():
    def __init__(self, layer : nn.Flatten):
        self.layer = layer

    def compute_bound(self, bounds: DpBounds):
        self.input_shape = bounds.lb.shape
        lb = self.layer(bounds.lb)
        ub = self.layer(bounds.ub)
        self.bounds = DpBounds(lb, ub)

    def backsub(self, constraints: DpConstraints):
        assert self.input_shape != None, "Forward pass not don yet"
        #print(f'flatten shape: {self.input_shape}, {constraints.lo.shape}')
        lr = constraints.lr.reshape((*constraints.lr.shape[:-1], *self.input_shape))
        ur = constraints.ur.reshape((*constraints.ur.shape[:-1], *self.input_shape))
        #print(constraints.uo.shape, self.input_shape)
        uo = constraints.uo.reshape((*constraints.uo.shape[:-1], *self.input_shape[1:]))
        lo = constraints.lo.reshape((*constraints.lo.shape[:-1], *self.input_shape[1:]))
        #print(f'flatten out: {lo.shape}')
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
        mask_upper = torch.where(bounds.ub <= 0, 0, 1)
        # ub >= lb >= 0
        mask_lower = torch.where(bounds.lb >= 0, 1, 0)
        # ub >= 0 >= lb
        mask_crossing = 1 - (mask_lower + mask_upper)

        self.slope = mask_crossing * self.slope
        ur = self.slope

        uo = torch.zeros(bounds.lb.shape)
        uo = mask_lower * bounds.ub + mask_crossing * self.bias_upper

        # For now use the x >= 0 constraint for lower relu
        lr = torch.zeros(bounds.lb.shape)
        lo = mask_lower * bounds.lb

        self.constraints = DpConstraints(lr, ur, lo, uo)

    def compute_bound(self, bounds: DpBounds):

        self.compute_constraints(bounds)

        pos_lr = nn.functional.relu(self.constraints.lr)
        neg_lr = -nn.functional.relu(-self.constraints.lr)
        lb = pos_lr * bounds.lb + neg_lr * bounds.ub + self.constraints.lo

        pos_ur = nn.functional.relu(self.constraints.ur)
        neg_ur = -nn.functional.relu(-self.constraints.ur)
        ub = pos_ur * bounds.ub + neg_ur * bounds.lb + self.constraints.uo

        self.bounds = DpBounds(lb, ub)

    def backsub(self, accum_c : DpConstraints):
        pos_ur = nn.functional.relu(accum_c.ur)
        neg_ur = -nn.functional.relu(-accum_c.ur)

        ur = self.constraints.ur * pos_ur + self.constraints.lr * neg_ur
        uo = self.constraints.uo * pos_ur + self.constraints.lo * neg_ur
        uo += accum_c.uo

        pos_lr = nn.functional.relu(accum_c.lr)
        neg_lr = -nn.functional.relu(-accum_c.lr)

        lr = self.constraints.lr * pos_lr + self.constraints.ur * neg_lr
        lo = self.constraints.lo * pos_lr + self.constraints.uo * neg_lr
        lo += accum_c.lo

        return DpConstraints(lr, ur, lo, uo)

class DpConv():
    def __init__(self, layer : nn.Conv2d):
        self.layer = layer

    def compute_bound(self, bounds: DpBounds):
        return NotImplementedError()

def check_postcondition(y, bounds: DpBounds) -> bool:
    try:
        target = y.item()
    except AttributeError:
        target = y

    lb = bounds.lb.flatten()
    ub = bounds.ub.flatten()

    target_lb = lb[target].item()
    for i in range(ub.shape[0]):
        if i != target and ub[i] >= target_lb:
            return False
    return True

# Function to get the 0th deepoly object with the initial bounds
# and the upper + lower identity constra
def get_input_bounds(x: torch.Tensor, eps: float):
    lb = (x - eps).to(torch.float)
    lb.clamp_(min=0, max=1)

    ub = (x + eps).to(torch.float)
    ub.clamp_(min=0, max=1)

    return DpBounds(lb, ub)

def bounded_matmul(lx: torch.Tensor, ux: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
    center = (ux + lx) / 2
    eps = (ux - lx) / 2
    center_out = center @ weight.t()
    if bias is not None:
        center_out = center_out + bias
    eps_out = eps @ weight.abs().t()
    lb = center_out - eps_out
    ub = center_out + eps_out
    return lb, ub

def bounded_matmul_alt(lx: torch.Tensor, ux: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
    center = (ux + lx) / 2
    eps = (ux - lx) / 2
    center_out = weight @ center
    if bias is not None:
        center_out = center_out + bias
    eps_out = weight.abs() @ eps
    lb = center_out - eps_out
    ub = center_out + eps_out
    return lb, ub

def bounded_mul(lx: torch.Tensor, ux: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
    center = (ux + lx) / 2
    eps = (ux - lx) / 2
    center_out = weight * center
    eps_out = weight.abs() * eps
    lb = center_out - eps_out
    ub = center_out + eps_out
    lb = lb.sum(dim=tuple(range(1, lb.dim())))
    ub = ub.sum(dim=tuple(range(1, ub.dim())))
    if bias is not None:
        #print(lb.shape, " ", bias.squeeze().shape)
        lb += bias.squeeze()
        ub += bias.squeeze()
    return lb, ub

def deeppoly_backsub(dp_layers):
    constraints_acc = dp_layers[-1].constraints
    for layer in reversed(dp_layers[:-1]):
        #print(constraints_acc.ur.shape)
        if isinstance(layer, DpLinear):
            constraints_acc = layer.backsub(constraints_acc)
        elif isinstance(layer, DpFlatten):
            constraints_acc = layer.backsub(constraints_acc)
        elif isinstance(layer, DpRelu):
            constraints_acc = layer.backsub(constraints_acc)
        elif isinstance(layer, DpConv):
            pass
        elif isinstance(layer, DpInput):
            #print(dp_layers[0].bounds.ub.shape, constraints_acc.ur.shape, constraints_acc.uo.shape)
            lb, _ = bounded_mul(dp_layers[0].bounds.lb, dp_layers[0].bounds.ub, constraints_acc.lr, constraints_acc.lo)
            _, ub = bounded_mul(dp_layers[0].bounds.lb, dp_layers[0].bounds.ub, constraints_acc.ur, constraints_acc.uo)
            #print(lb.shape, " ", ub.shape)

    return DpBounds(lb, ub)

def propagate_sample(model, x, eps):

    bounds = get_input_bounds(x, eps)
    input_layer = DpInput(bounds)
    dp_layers = [input_layer]

    for layer in model:
        if isinstance(layer, nn.Flatten):
            dp_layer = DpFlatten(layer)
            dp_layer.compute_bound(dp_layers[-1].bounds)
            dp_layers.append(dp_layer)
        elif isinstance(layer, nn.Linear):
            dp_layer = DpLinear(layer)
            dp_layer.compute_bound(dp_layers[-1].bounds)
            print(dp_layer.bounds.lb.shape)
            dp_layers.append(dp_layer)
        elif isinstance(layer, nn.ReLU):
            dp_layer = DpRelu(layer)
            dp_layer.compute_bound(dp_layers[-1].bounds)
            print(f'uo shape: {dp_layer.constraints.uo.shape}')
            dp_layers.append(dp_layer)
    return dp_layers

def certify_sample(model, x, y, eps) -> bool:
    dp_layers = propagate_sample(model, x, eps)
    if check_postcondition(y, dp_layers[-1].bounds):
        return True
    bounds = deeppoly_backsub(dp_layers)
    #print(bounds.lb, " ", bounds.ub)
    return check_postcondition(y, bounds)

if __name__ == "__main__":

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


        bounds = get_input_bounds(x, eps)
        input_layer = DpInput(bounds)
        dp_layers = [input_layer]

        for layer in model:
            if isinstance(layer, nn.Flatten):
                dp_layer = DpFlatten(layer)
                dp_layer.compute_bound(dp_layers[-1].bounds)
                dp_layers.append(dp_layer)
            elif isinstance(layer, nn.Linear):
                dp_layer = DpLinear(layer)
                dp_layer.compute_bound(dp_layers[-1].bounds)
                dp_layers.append(dp_layer)
            elif isinstance(layer, nn.ReLU):
                dp_layer = DpRelu(layer)
                dp_layer.compute_bound(dp_layers[-1].bounds)
                dp_layers.append(dp_layer)
        
        bounds = deeppoly_backsub(dp_layers)
        lb = bounds.lb.flatten()
        ub = bounds.ub.flatten()

        return ub[1].items()
    
    import numpy as np
    import matplotlib.pyplot as plt

    w_vals = np.linspace(-3, 5, 100)
    ub_vals = []
    for w in w_vals:
        ub_vals.append(simulate(w))
    plt.plot(w_vals, ub_vals)
    plt.show()

