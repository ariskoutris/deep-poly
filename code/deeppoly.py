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
        out =  "\tlr: " + str(self.lr).replace("\n", "\n   ").replace("tensor(", "").replace(")", "") + "\n"
        out += "\tur: " + str(self.ur).replace("\n", "\n   ").replace("tensor(", "").replace(")", "") + "\n"
        out += "\tlo: " + str(self.lo).replace("\n", "\n   ").replace("tensor(", "").replace(")", "") + "\n"
        out += "\tuo: " + str(self.uo).replace("\n", "\n   ").replace("tensor(", "").replace(")", "") + "\n"

        # out = f"lr: shape [{self.lr.shape}], min: {self.lr.min()}, max: {self.lr.max()}\n"
        # out += f"ur: shape [{self.ur.shape}], min: {self.ur.min()}, max: {self.ur.max()}\n"
        # out += f"lo: shape [{self.lo.shape}], min: {self.lo.min()}, max: {self.lo.max()}\n"
        # out += f"uo: shape [{self.uo.shape}], min: {self.uo.min()}, max: {self.uo.max()}\n"
        return out

class DpBounds:
    def __init__(self, lb: torch.Tensor, ub: torch.Tensor):
        self.lb = lb
        self.ub = ub
        assert self.lb.shape == self.ub.shape
        assert (self.lb > self.ub).sum() == 0

    def __repr__(self):
        return f"lb: {self.lb}\tub: {self.ub}"

class DpInput():
    def __init__(self, bounds: DpBounds):
        self.layer = None
        self.bounds = bounds

class DpLinear():

    def __init__(self, layer : nn.Linear):
        self.layer = layer
        lr = layer.weight.detach().t()
        ur = layer.weight.detach().t()
        lo = layer.bias.detach()
        uo = layer.bias.detach()
        self.constraints = DpConstraints(lr, ur, lo, uo)

    def compute_bound(self, bounds: DpBounds):
        
        lr_pos = torch.relu(self.constraints.lr)
        lr_neg = -torch.relu(-self.constraints.lr)
        ur_pos = torch.relu(self.constraints.ur)
        ur_neg = -torch.relu(-self.constraints.ur)
        
        lb = bounds.lb @ lr_pos + bounds.ub @ lr_neg + self.constraints.lo
        ub = bounds.ub @ ur_pos + bounds.lb @ ur_neg + self.constraints.uo
        
        self.bounds = DpBounds(lb, ub)

    def backsub(self, accum_c: DpConstraints):
        
        accum_c_lr_pos = torch.relu(accum_c.lr)
        accum_c_lr_neg = -torch.relu(-accum_c.lr)
        accum_c_ur_pos = torch.relu(accum_c.ur)
        accum_c_ur_neg = -torch.relu(-accum_c.ur)
        
        lr =  self.constraints.lr @ accum_c_lr_pos +  self.constraints.ur @ accum_c_lr_neg
        ur =  self.constraints.ur @ accum_c_ur_pos +  self.constraints.lr @ accum_c_ur_neg

        lo = self.constraints.lo @ accum_c_lr_pos + self.constraints.uo @ accum_c_lr_neg
        uo = self.constraints.uo @ accum_c_ur_pos + self.constraints.lo @ accum_c_ur_neg
        uo = uo + accum_c.uo
        lo = lo + accum_c.lo

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
        lr[mask_crossing] = torch.where(-bounds.lb < bounds.ub, 0.0, 0.0)[mask_crossing]
        lr[mask_lower] = 1
        lo = torch.zeros_like(bounds.lb)

        self.constraints = DpConstraints(torch.diag(lr.flatten()), torch.diag(ur.flatten()), lo, uo)

    def compute_bound(self, bounds: DpBounds):

        self.compute_constraints(bounds)
        
        lr_pos = torch.relu(self.constraints.lr)
        lr_neg = -torch.relu(-self.constraints.lr)
        ur_pos = torch.relu(self.constraints.ur)
        ur_neg = -torch.relu(-self.constraints.ur)
        
        lb = bounds.lb @ lr_pos + bounds.ub @ lr_neg + self.constraints.lo
        ub = bounds.ub @ ur_pos + bounds.lb @ ur_neg + self.constraints.uo
        
        self.bounds = DpBounds(lb, ub)

    def backsub(self, accum_c: DpConstraints):
        
        accum_c_lr_pos = torch.relu(accum_c.lr)
        accum_c_lr_neg = -torch.relu(-accum_c.lr)
        accum_c_ur_pos = torch.relu(accum_c.ur)
        accum_c_ur_neg = -torch.relu(-accum_c.ur)
        
        lr =  self.constraints.lr @ accum_c_lr_pos +  self.constraints.ur @ accum_c_lr_neg
        ur =  self.constraints.ur @ accum_c_ur_pos +  self.constraints.lr @ accum_c_ur_neg

        lo = self.constraints.lo @ accum_c_lr_pos + self.constraints.uo @ accum_c_lr_neg
        uo = self.constraints.uo @ accum_c_ur_pos + self.constraints.lo @ accum_c_ur_neg
        uo = uo + accum_c.uo
        lo = lo + accum_c.lo

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
    min_interval = ub.max() - lb.min()
    
    out = True
    for i in range(ub.shape[0]):
        if i != target and ub[i] >= target_lb:
            out = False
        if i != target:
            min_interval = min(min_interval, target_lb - ub[i])
    logger.info(f'Certification Distance: {min_interval}\n')
    return out


# Function to get the 0th deepoly object with the initial bounds
# and the upper + lower identity constra
def get_input_bounds(x: torch.Tensor, eps: float, min_val=0, max_val=1):
    lb = (x - eps).to(torch.float)
    lb.clamp_(min=min_val, max=max_val)

    ub = (x + eps).to(torch.float)
    ub.clamp_(min=min_val, max=max_val)

    return DpBounds(lb, ub)

def deeppoly_backsub(dp_layers):
    constraints_acc = dp_layers[-1].constraints
    logger.info("BACKWARD PROPAGATION")
    logger.debug(f'Last Layer:\n{str(constraints_acc)}')
    for i, layer in enumerate(reversed(dp_layers[:-1])):
        if isinstance(layer, DpLinear):
            constraints_acc = layer.backsub(constraints_acc)
        elif isinstance(layer, DpFlatten):
            constraints_acc = layer.backsub(constraints_acc)
        elif isinstance(layer, DpRelu):
            constraints_acc = layer.backsub(constraints_acc)
        elif isinstance(layer, DpConv):
            pass
        elif isinstance(layer, DpInput):
            constraints_acc_ur_pos = torch.relu(constraints_acc.ur)
            constraints_acc_ur_neg = -torch.relu(-constraints_acc.ur)
            constraints_acc_lr_neg = -torch.relu(-constraints_acc.lr)
            constraints_acc_lr_pos = torch.relu(constraints_acc.lr)
            lb_in = dp_layers[0].bounds.lb
            ub_in = dp_layers[0].bounds.ub
            lb = lb_in @ constraints_acc_lr_pos + ub_in @ constraints_acc_lr_neg + constraints_acc.lo
            ub = ub_in @ constraints_acc_ur_pos + lb_in @ constraints_acc_ur_neg + constraints_acc.uo 
        logger.debug(f'Layer {len(dp_layers) - 2 - i} [{layer.layer}]:')
        logger.debug(str(constraints_acc))
    return DpBounds(lb, ub)

def deeppoly_backsub_aux(dp_layers, constraints_acc: DpConstraints):
    logger.info("BACKWARD PROPAGATION")
    logger.debug(f'Last Layer:\n{constraints_acc}')
    for i, layer in enumerate(reversed(dp_layers[:-1])):
        if isinstance(layer, DpLinear):
            constraints_acc = layer.backsub(constraints_acc)
        elif isinstance(layer, DpFlatten):
            constraints_acc = layer.backsub(constraints_acc)
        elif isinstance(layer, DpRelu):
            constraints_acc = layer.backsub(constraints_acc)
        elif isinstance(layer, DpConv):
            pass
        elif isinstance(layer, DpInput):
            constraints_acc_ur_pos = torch.relu(constraints_acc.ur)
            constraints_acc_ur_neg = -torch.relu(-constraints_acc.ur)
            constraints_acc_lr_neg = -torch.relu(-constraints_acc.lr)
            constraints_acc_lr_pos = torch.relu(constraints_acc.lr)
            lb_in = dp_layers[0].bounds.lb
            ub_in = dp_layers[0].bounds.ub
            lb = lb_in @ constraints_acc_lr_pos + ub_in @ constraints_acc_lr_neg + constraints_acc.lo
            ub = ub_in @ constraints_acc_ur_pos + lb_in @ constraints_acc_ur_neg + constraints_acc.uo 
        logger.debug(f'Layer {len(dp_layers) - 2 - i} [{layer.layer}]:')
        logger.debug(str(constraints_acc))

    return DpBounds(lb, ub)

def propagate_sample(model, x, eps, min_val=0, max_val=1):

    bounds = get_input_bounds(x, eps, min_val, max_val)
    input_layer = DpInput(bounds)
    dp_layers = [input_layer]
    logger.info("FORWARD PROPAGATION")
    #logger.debug(f'Input Layer:\n\t{input_layer.bounds.lb.numpy()} {input_layer.bounds.ub.numpy()}')
    logger.debug(f'Input Layer')
    logger.debug(f'lb: shape [{input_layer.bounds.lb.shape}], min: {input_layer.bounds.lb.min()}, max: {input_layer.bounds.lb.max()}')
    logger.debug(f'ub: shape [{input_layer.bounds.ub.shape}], min: {input_layer.bounds.ub.min()}, max: {input_layer.bounds.ub.max()}')
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Flatten):
            dp_layer = DpFlatten(layer)
            dp_layer.compute_bound(dp_layers[-1].bounds)
            dp_layers.append(dp_layer)
        elif isinstance(layer, nn.Linear):
            dp_layer = DpLinear(layer)
            dp_layer.compute_bound(dp_layers[-1].bounds)
            dp_layers.append(dp_layer)
            #dp_layer.bounds = deeppoly_backsub_aux(dp_layers, dp_layer.constraints)
        elif isinstance(layer, nn.ReLU):
            dp_layer = DpRelu(layer)
            dp_layer.compute_bound(dp_layers[-1].bounds)
            dp_layers.append(dp_layer)
            #dp_layer.bounds = deeppoly_backsub_aux(dp_layers, dp_layer.constraints)
        #logger.debug(f'Layer {i + 1} {layer}:\n\t{dp_layer.bounds.lb.numpy()} {dp_layer.bounds.ub.numpy()}\t\t\t')
        logger.debug(f'Layer {i + 1} {layer}')
        logger.debug(f'lb: shape [{dp_layer.bounds.lb.shape}], min: {dp_layer.bounds.lb.min()}, max: {dp_layer.bounds.lb.max()}')
        logger.debug(f'ub: shape [{dp_layer.bounds.ub.shape}], min: {dp_layer.bounds.ub.min()}, max: {dp_layer.bounds.ub.max()}')
    return dp_layers

def certify_sample(model, x, y, eps) -> bool:
    dp_layers = propagate_sample(model, x, eps)
    if check_postcondition(y, dp_layers[-1].bounds):
        return True
    bounds = deeppoly_backsub(dp_layers)
    return check_postcondition(y, bounds)

if __name__ == "__main__":
    
    import logging
    # Configure logging. Set level to [NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL] (in order) to control verbosity.
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', datefmt='%X')
    logger = logging.getLogger(__name__)
    
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
    
    def sample():
        """
        linear = nn.Linear(3, 2)
        linear.weight.data = torch.tensor([[2, 1, -7], [1, 3, 1]], dtype=torch.float)
        linear.bias.data = torch.tensor([3, -5], dtype=torch.float)
        leaky = nn.LeakyReLU(negative_slope=0.5)
        flatten = nn.Flatten()
        model = nn.Sequential(linear, leaky, flatten)
        x = torch.tensor([[[1, 0, 0], [1, 0, 1], [0, 1, 1]]], dtype=torch.float)
        """
        # The example we did in class
        linear1 = nn.Linear(2, 2)
        linear1.weight.data = torch.tensor([[1, 1], [1, -1]], dtype=torch.float)
        linear1.bias.data = torch.tensor([0, 0], dtype=torch.float)
        relu1 = nn.ReLU()
        linear2 = nn.Linear(2, 2)
        linear2.weight.data = torch.tensor([[1, 1], [1, -1]], dtype=torch.float)
        linear2.bias.data = torch.tensor([-0.5, 0], dtype=torch.float)
        relu2 = nn.ReLU()
        linear3 = nn.Linear(2, 2)
        linear3.weight.data = torch.tensor([[-1, 1], [0, 1]], dtype=torch.float)
        linear3.bias.data = torch.tensor([3, 0], dtype=torch.float)
        flatten = nn.Flatten()
        model = nn.Sequential(linear1, relu1, linear2, relu2, linear3)
        model.eval()

    def sample():
        """
        linear = nn.Linear(3, 2)
        linear.weight.data = torch.tensor([[2, 1, -7], [1, 3, 1]], dtype=torch.float)
        linear.bias.data = torch.tensor([3, -5], dtype=torch.float)
        leaky = nn.LeakyReLU(negative_slope=0.5)
        flatten = nn.Flatten()
        model = nn.Sequential(linear, leaky, flatten)
        x = torch.tensor([[[1, 0, 0], [1, 0, 1], [0, 1, 1]]], dtype=torch.float)
        """
        # The example we did in class
        linear1 = nn.Linear(2, 2)
        linear1.weight.data = torch.tensor([[1, 1], [1, -1]], dtype=torch.float)
        linear1.bias.data = torch.tensor([0, 0], dtype=torch.float)
        relu1 = nn.ReLU()
        linear2 = nn.Linear(2, 2)
        linear2.weight.data = torch.tensor([[1, 1], [1, -1]], dtype=torch.float)
        linear2.bias.data = torch.tensor([-0.5, 0], dtype=torch.float)
        relu2 = nn.ReLU()
        linear3 = nn.Linear(2, 2)
        linear3.weight.data = torch.tensor([[-1, 1], [0, 1]], dtype=torch.float)
        linear3.bias.data = torch.tensor([3, 0], dtype=torch.float)
        flatten = nn.Flatten()
        model = nn.Sequential(linear1, relu1, linear2, relu2, linear3)
        model.eval()
        
        x = torch.tensor([[0, 0]])
        eps = 1.0

        dp_layers = propagate_sample(model, x, eps, min_val=-1, max_val=1)
        if check_postcondition(0, dp_layers[-1].bounds):
            print("Verified")
        else:
            print("Failed")
        bounds = deeppoly_backsub(dp_layers)
        if check_postcondition(0, bounds):
            print("Verified")
        else:
            print("Failed")

    sample()
    
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

        x = torch.tensor([[0]])
        eps = 1.0

        print()
        dp_layers = propagate_sample(model, x, eps, -1, 1)
        bounds = deeppoly_backsub(dp_layers)
        
        lb = bounds.lb.flatten()
        ub = bounds.ub.flatten()
        return ub[1].item()

    # import numpy as np
    # import matplotlib.pyplot as plt

    # w_vals = np.linspace(-3, 5, 500)
    # ub_vals = []
    # for w in w_vals:
    #     ub_vals.append(simulate(w))
    # plt.plot(w_vals, ub_vals)
    # plt.show()
    
    # simulate(weight)