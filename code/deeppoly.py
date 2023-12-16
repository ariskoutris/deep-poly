import torch
import torch.nn as nn
from dp_utils import *
import numpy as np

from line_profiler import profile
from time import perf_counter
import logging
logger = logging.getLogger(__name__)

DTYPE = torch.float32

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
        self.constraints_trans = DpConstraints(lr.t(), ur.t(), lo, uo)
        self.inp_shape = None
        self.out_last_dim = lo.shape[-1]

    def compute_bound(self, bounds: DpBounds):
        self.inp_shape = bounds.shape
        self.bounds = bounds_mul_constraints(self.constraints_trans, bounds)
        self.bounds.lb = self.bounds.lb.view(*self.inp_shape[:-1], self.out_last_dim)
        self.bounds.ub = self.bounds.ub.view(*self.inp_shape[:-1], self.out_last_dim)

    def backsub(self, accum_c: DpConstraints):
        return constraints_mul(self.constraints, accum_c)


class DpFlatten():
    def __init__(self, layer : nn.Flatten):
        self.layer = layer

    def compute_bound(self, bounds: DpBounds):
        self.input_shape = bounds.lb.shape
        lb = self.layer(bounds.lb)
        ub = self.layer(bounds.ub)
        self.bounds = DpBounds(lb, ub)

    def backsub(self, constraints: DpConstraints):
        lr = constraints.lr.reshape((*self.input_shape, *constraints.lr.shape[-1:]))
        ur = constraints.ur.reshape((*self.input_shape, *constraints.ur.shape[-1:]))
        return DpConstraints(lr, ur, constraints.lo, constraints.uo)


class DpRelu():
    def __init__(self, layer : nn.ReLU, is_leaky = True):
        self.layer = layer
        self.relu_neg_slope = 0.0 if not is_leaky else layer.negative_slope
        self.bounds = None
        self.inp_shape = None
        self.pos_slope = None
        self.neg_slope = None
        self.bias_upper = None
        self.bias_lower = None
        self.constraints = None
        self.alphas = None

    def compute_constraints(self, bounds: DpBounds):
        bound_diff = bounds.ub - bounds.lb
        slope_common = (bounds.ub - self.relu_neg_slope * bounds.lb)/bound_diff
        bias_common = - (1 - self.relu_neg_slope) * bounds.ub * bounds.lb / bound_diff
        #  0 >= ub >= lb
        mask_upper = bounds.ub <= 0
        # ub >= lb >= 0
        mask_lower = bounds.lb >= 0
        # ub >= 0 >= lb
        mask_crossing =  ~(mask_lower | mask_upper)
        assert (mask_crossing & mask_upper & mask_lower == False).all()

        if self.relu_neg_slope <= 1.0:
            self.pos_slope = slope_common
            self.bias_upper = bias_common

            if self.alphas == None:
                self.neg_slope = torch.where(-bounds.lb < bounds.ub, 1.0, self.relu_neg_slope).to(dtype=DTYPE)
            else:
                self.neg_slope = self.alphas
            self.bias_lower = torch.zeros_like(self.bias_upper)
        else:
            self.neg_slope = slope_common
            self.bias_lower = bias_common

            if self.alphas == None:
                self.pos_slope = torch.where(-bounds.lb < bounds.ub, 1.0, self.relu_neg_slope).to(dtype=DTYPE)
            else:
                self.pos_slope = self.alphas
            self.bias_upper = torch.zeros_like(self.bias_lower)

        ur = torch.zeros_like(bounds.ub)
        ur[mask_crossing] = self.pos_slope[mask_crossing]
        ur[mask_lower] = 1
        ur[mask_upper] = self.relu_neg_slope

        uo = torch.zeros_like(bounds.lb)
        uo[mask_crossing] = self.bias_upper[mask_crossing]

        lr = torch.zeros_like(bounds.lb)
        lr[mask_crossing] = self.neg_slope[mask_crossing]
        lr[mask_lower] = 1
        lr[mask_upper] = self.relu_neg_slope

        lo = torch.zeros_like(bounds.lb)
        lo[mask_crossing] = self.bias_lower[mask_crossing]

        self.constraints = DpConstraints(torch.diag(lr.flatten()), torch.diag(ur.flatten()), lo.flatten(), uo.flatten())

    def compute_bound(self, bounds: DpBounds):
        self.inp_shape = bounds.shape
        self.compute_constraints(bounds)
        self.bounds = bounds_mul_constraints(self.constraints, bounds)
        self.bounds.lb = self.bounds.lb.view(self.inp_shape)
        self.bounds.ub = self.bounds.ub.view(self.inp_shape)

    def backsub(self, accum_c: DpConstraints):
        acc_lr = accum_c.lr.flatten(0, -2)
        acc_ur = accum_c.ur.flatten(0, -2)
        constraints_out = constraints_mul(self.constraints, DpConstraints(acc_lr, acc_ur, accum_c.lo, accum_c.uo))
        out_lr = constraints_out.lr.view(*self.inp_shape, acc_lr.shape[-1])
        out_ur = constraints_out.ur.view(*self.inp_shape, acc_ur.shape[-1])
        return DpConstraints(out_lr, out_ur, constraints_out.lo, constraints_out.uo)
    
    def set_alphas(self, alphas: torch.Tensor):
        self.alphas = alphas


class DpConv():
    def __init__(self, layer : nn.Conv2d):
        self.layer = layer
        self.constraints = None
        self.constraints_trans = None
        self.inp_shape = None
        self.out_shape = None
    
    @profile
    def compute_weight_matrix(self, inp_shape):
        
        @profile
        def get_weight(inp_shape, conv_row, conv_col, kernel):
            temp = torch.zeros(inp_shape)
            end_row = conv_row + kernel.shape[1]
            end_col = conv_col + kernel.shape[2]
            temp[:, conv_row:end_row, conv_col:end_col] = kernel
            return temp
        
        @profile
        def get_weight_matrix(conv, inp_shape):
            #TODO: Improve efficiency. Remove loops
            kernel = conv.weight.data.detach()
            C_out, C_in, K_h, K_w = kernel.shape
            N_in, C_in, H_i, W_i = inp_shape
            H_o = ((inp_shape[-2] + conv.padding[-2] + conv.padding[-2] - conv.kernel_size[-2]) // conv.stride[-2] + 1)
            W_o = ((inp_shape[-1] + conv.padding[-1] + conv.padding[-1] - conv.kernel_size[-1]) // conv.stride[-1] + 1)
            out_shape = N_in, C_out, H_o, W_o

            H_grd, W_grd = H_o, H_i
            H_blk, W_blk = W_o, W_i

            W_conv = torch.zeros((C_out, H_grd, H_blk, C_in, W_grd, W_blk), dtype=DTYPE)

            for c in range(C_out):
                for i in range(H_o):
                    for j in range(W_o):
                        padded_H_i, padded_W_i = H_i + 2 * conv.padding[0], W_i + 2 * conv.padding[1]
                        conv_row, conv_col = i * conv.stride[0], j * conv.stride[1]
                        if conv_row >= padded_H_i | conv_col >= padded_W_i:
                            continue
                        temp_weight = get_weight((C_in, padded_H_i, padded_W_i), conv_row, conv_col, kernel[c])
                        W_conv[c, i, j] = temp_weight[:, conv.padding[0] : H_i+conv.padding[0], conv.padding[1] : W_i + conv.padding[1]]

            B_conv = conv.bias.data.detach()
            B_conv = torch.ones(H_o*W_o, C_out) * B_conv
            B_conv = B_conv.t()

            return W_conv, B_conv, out_shape
        
        # conv = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(2, 2), stride=(2, 2))
        # conv.double()
        # # conv.bias.data = torch.zeros_like(conv.bias)
        # inp_shape = (1, 1, 28, 28)

        W, B, out_shape = get_weight_matrix(self.layer, inp_shape)

        W = torch.flatten(W, start_dim=0, end_dim=2)
        W = torch.flatten(W, start_dim=1, end_dim=3)

        B = B.flatten()
        W.shape[1] == B.shape[0]

        # def test_weight(T, B, conv, inp_shape):
        #     i = torch.randn(*inp_shape, dtype = DTYPE)
        #     out = i.flatten() @ T.t() + B
        #     print(torch.allclose(conv(i).flatten(), out.flatten(), atol=1e-06))

        # for i in range(100):
        #     test_weight(W, B, conv, inp_shape)
        return W, B, out_shape

    @profile 
    def compute_bound(self, bounds: DpBounds):
        if self.inp_shape != None and self.inp_shape == bounds.shape:
            r = self.constraints.lr
            o = self.constraints.lo
        else:
            self.inp_shape = bounds.shape
            r, o, out_shape = self.compute_weight_matrix(bounds.shape)
            self.constraints = DpConstraints(r, r, o, o)
            self.out_shape = out_shape

        self.bounds = bounds_mul_constraints(DpConstraints(r.t(), r.t(), o, o), bounds)
        self.bounds.lb = self.bounds.lb.view(self.out_shape)
        self.bounds.ub = self.bounds.ub.view(self.out_shape)
        return self.bounds
    
    def backsub(self, accum_c: DpConstraints):
        acc_lr = accum_c.lr.flatten(0, -2)
        acc_ur = accum_c.ur.flatten(0, -2)
        constraints_out = constraints_mul(self.constraints, DpConstraints(acc_lr, acc_ur, accum_c.lo, accum_c.uo))
        out_lr = constraints_out.lr.view(*self.inp_shape, acc_lr.shape[-1])
        out_ur = constraints_out.ur.view(*self.inp_shape, acc_ur.shape[-1])
        return DpConstraints(out_lr, out_ur, constraints_out.lo, constraints_out.uo)


class DiffLayer():
    def __init__(self, target: int, n_classes: int):
        self.layer = None
        self.target = target
        self.n_classes = n_classes
        self.constraints = self.compute_constraints(target, n_classes)

    def compute_constraints(self, target: int = None, n_classes: int = None):
        I = [i for i in range(n_classes) if i != target]
        C = torch.eye(n_classes, dtype=DTYPE)[target].unsqueeze(dim=0) - torch.eye(n_classes, dtype=DTYPE)[I]
        lr, ur = C, C
        lo = torch.zeros_like(lr[:, 0])
        uo = torch.zeros_like(ur[:, 0])
        self.bounds = DpBounds(torch.zeros_like(lo), torch.zeros_like(uo))
        return DpConstraints(lr, ur, lo, uo)

    def backsub(self, accum_c: DpConstraints):
        return constraints_mul(self.constraints, accum_c)

@profile
def deeppoly_backsub(dp_layers):
    constraints_acc = dp_layers[-1].constraints.copy()
    constraints_acc.ur = constraints_acc.ur.t()
    constraints_acc.lr = constraints_acc.lr.t()
    logger.debug('[BACKSUBSTITUTION START]')
    logger.debug(f'Current Layer [{dp_layers[-1].layer}]:\n{str(constraints_acc)}')
    for i, layer in enumerate(reversed(dp_layers[1:-1])):
        constraints_acc = layer.backsub(constraints_acc)
        logger.debug(f'Layer {len(dp_layers) - 2 - i} [{layer.layer}]:')
        logger.debug(str(constraints_acc))
        
    ur = constraints_acc.ur.flatten(0, -2)
    assert ur.dim() == 2
    lr = constraints_acc.lr.flatten(0, -2)
    lb_in = dp_layers[0].bounds.lb.flatten(1)
    ub_in = dp_layers[0].bounds.ub.flatten(1)
    b_curr = bounds_mul_constraints(DpConstraints(lr, ur, constraints_acc.lo, constraints_acc.uo), DpBounds(lb_in, ub_in))
    lb = b_curr.lb.view(dp_layers[-1].bounds.shape)
    ub = b_curr.ub.view(dp_layers[-1].bounds.shape)
    logger.debug(f'Input Layer:')
    logger.debug(str(constraints_acc))
    logger.debug('[BACKSUBSTITUTION END]')
    return DpBounds(lb, ub)

@profile
def propagate_sample(model, x, eps, le_layer=None, min_val=0, max_val=1, layers=None):
    bounds = get_input_bounds(x, eps, min_val, max_val)
    input_layer = DpInput(bounds)
    dp_layers = [input_layer] if layers == None else layers
    log_layer_bounds(logger, input_layer, 'Input Layer')
    for i, layer in enumerate(model):
        dp_layer = None
        if layers != None:
            dp_layer = dp_layers[i + 1] # i + 1 as the first element is DpInput
            dp_layer.compute_bound(dp_layers[i].bounds)
        elif isinstance(layer, nn.Flatten):
            dp_layer = DpFlatten(layer)
            dp_layer.compute_bound(dp_layers[i].bounds)
        elif isinstance(layer, nn.Linear):
            dp_layer = DpLinear(layer)
            dp_layer.compute_bound(dp_layers[i].bounds)
        elif isinstance(layer, nn.ReLU):
            dp_layer = DpRelu(layer, False)
            dp_layer.compute_bound(dp_layers[i].bounds)
        elif isinstance(layer, nn.LeakyReLU):
            dp_layer = DpRelu(layer)
            dp_layer.compute_bound(dp_layers[i].bounds)
        elif isinstance(layer, nn.Conv2d):
            dp_layer = DpConv(layer)
            dp_layer.compute_bound(dp_layers[i].bounds)

        # Uncomment this line after optimization of DpConv.compute_bound is complete
        # dp_layer.compute_bound(dp_layers[i].bounds)
        
        if layers == None:
            dp_layers.append(dp_layer)
        
        # Backsubstitution is called on the layer before the ReLU
        if not isinstance(layer, nn.Flatten):
            if i + 2 <= len(model) and (isinstance(model[i + 1], nn.ReLU) or isinstance(model[i + 1], nn.LeakyReLU)):
                dp_layer.bounds = deeppoly_backsub(dp_layers[:i+2]) # i + 2 as the first element is DpInput
                
        log_layer_bounds(logger, dp_layer, f'Layer {i + 1} [{layer}]')

    if le_layer is not None:
        if layers is None:
            dp_layers.append(le_layer)
            le_layer.bounds = deeppoly_backsub(dp_layers)
        else:
            dp_layers[-1] = le_layer
            le_layer.bounds = deeppoly_backsub(dp_layers)
        log_layer_bounds(logger, le_layer, f'Layer {len(dp_layers) - 1} [{le_layer}]:')

    return dp_layers

def assign_alphas_to_relus(dp_layers, alphas):
    for i, layer in enumerate(dp_layers):
        if (i-1) in alphas:
            layer.set_alphas(alphas[i - 1].value) # i - 1 as the first element of dp_layers is DpInput
    return dp_layers

@profile
def certify_sample(model, x, y, eps, use_le=True, use_slope_opt=True) -> bool:
    model = model.to(dtype=DTYPE)
    x = x.to(dtype=DTYPE)

    if x.dim() == 3:
        x = x.unsqueeze(0)

    if use_le:
        n_classes = model[-1].out_features
        le_layer = DiffLayer(y, n_classes)
        dp_layers = propagate_sample(model, x, eps, le_layer=le_layer)
    else:
        dp_layers = propagate_sample(model, x, eps)

    bounds = dp_layers[-1].bounds

    verified = check_postcondition_le(bounds) if use_le else check_postcondition(y, bounds)
    if verified:
        logger.warning(f'Certification Distance: {bounds.get_certification_distance()}')
        return True

    if not use_slope_opt:
        return False
    
    num_restarts = 1
    for _ in range(num_restarts):
        verified = certify_with_alphas(model, dp_layers, x, y, eps, 30, use_le)
        if verified:
            break
        
    return verified

@profile
def certify_with_alphas(model, dp_layers, x, y, eps, num_epochs, use_le=True):

    alphas_dict = init_alphas(model, x.shape)
    if len(alphas_dict) == 0:
        return False
    dp_layers = assign_alphas_to_relus(dp_layers, alphas_dict)

    loss_func = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam([alphas_dict[key].value for key in alphas_dict], lr=2)

    # Early Stopping Parameters
    num_epochs = 30
    min_epochs = 3
    window_size = 3
    cd_window = []
    cd_max = -1000
    patience = 2
    pi_window = []
    min_pi = 0.04
    
    for epoch in range(num_epochs):
        
        if use_le:
            n_classes = model[-1].out_features
            le_layer = DiffLayer(y, n_classes)
            dp_layers = propagate_sample(model, x, eps, le_layer=le_layer, layers=dp_layers)
        else:
            dp_layers = propagate_sample(model, x, eps, layers=dp_layers)
            
        bounds = dp_layers[-1].bounds
        
        if use_le:
            loss = torch.sum(-bounds.lb[bounds.lb < 0])
        else:
            loss = loss_func(bounds.get_loss_tensor(y), torch.tensor(y).view(1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        for alpha_param in alphas_dict.values():
            alpha_param.value.data.clamp_(alpha_param.lb, alpha_param.ub)

        verified = check_postcondition_le(bounds) if use_le else check_postcondition(y, bounds)
        cert_dist = bounds.get_certification_distance()

        if verified:
            logger.warning(f'Certification Distance: {cert_dist}\n')
            return True
        
        if len(cd_window) == window_size:
            cd_window.pop(0)
            pi_window.pop(0)
            
        cd_window.append(cert_dist)
        perc_improvement = (cd_max - cert_dist) / cd_max
        pi_window.append(perc_improvement)
        
        cd_mean = np.mean(cd_window)
        cd_std = np.std(cd_window)
        upper_confidence_bound = cd_mean + 2 * cd_std
        in_upper_confidence_bound = upper_confidence_bound >= 0
        
        pi_mean = np.mean(pi_window)
        pi_std = np.std(pi_window)
        
        new_cd_max = np.max(cd_window)
        if new_cd_max > cd_max:
            cd_max = new_cd_max
            
        if perc_improvement > min_pi:
            patience_counter = patience
        else:
            patience_counter -= 1
        
        logger.warning(f'Epoch: {epoch} | Certification Distance: {cert_dist:4f} | Mean: {cd_mean:4f} | Std: {cd_std:4f} | Upper Confidence Bound: {upper_confidence_bound:4f} | In Upper Confidence Bound: {in_upper_confidence_bound} | Max: {cd_max:4f} | Patience: {patience_counter} | Percentage Improvement {perc_improvement:4f} | Mean PI {pi_mean:4f} | Std PI: {pi_std:4f}\n')
        if epoch >= min_epochs:
            if not in_upper_confidence_bound and patience_counter <= 0:
                logger.warning(f'Certification Distance: {cert_dist}\n')
                return False
        
    return False
