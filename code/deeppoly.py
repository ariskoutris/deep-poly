import torch
import torch.nn as nn

deeppoly_layers = []

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
        print(self.constraints.lr.dtype, bounds.lb.dtype)
        lb = self.constraints.lr @ bounds.lb + self.constraints.lo
        ub = self.constraints.ur @ bounds.ub + self.constraints.uo
        self.bounds = DpBounds(lb, ub)

class DpFlatten():
    def __init__(self, layer : nn.Flatten):
        self.layer = layer

    def compute_bound(self, bounds: DpBounds):
        lb = self.layer(bounds.lb)
        ub = self.layer(bounds.ub)
        self.bounds = DpBounds(lb, ub)

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
    target_lb = bounds.lb[0][target].item()
    for i in range(bounds.ub.shape[-1]):
        if i != target and bounds.ub[0][i] >= target_lb:
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

def deeppoly_backsub():
    raise NotImplementedError()

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
            dp_layers.append(dp_layer)
        elif isinstance(layer, nn.ReLU):
            raise NotImplementedError()
    return dp_layers

def certify_sample(model, x, y, eps) -> bool:
    dp_layers = propagate_sample(model, x, eps)
    return dp_layers.check_postcondition(y)

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

        for i, layer in enumerate(model):
            logger_msg = f'Layer {i} - Type: {type(layer).__name__}'
            if isinstance(layer, nn.Flatten):
                dp_layer = DpFlatten(layer)
                dp_layer.compute_bound(dp_layers[-1].bounds)
                dp_layers.append(dp_layer)
            elif isinstance(layer, nn.Linear):
                dp_layer = DpLinear(layer)
                dp_layer.compute_bound(dp_layers[-1].bounds)
                dp_layers.append(dp_layer)
            elif isinstance(layer, nn.ReLU):
                raise NotImplementedError()
        return dp_layers[-1].ub.data[0,1].item()

    simulate(0.5)
