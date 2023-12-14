import torch

def toeplitz_torch(c, r=None):
    """
    Construct a Toeplitz matrix in PyTorch.

    Parameters
    ----------
    c : tensor
        First column of the matrix.
    r : tensor, optional
        First row of the matrix. If None, `r = conjugate(c)` is assumed.

    Returns
    -------
    A : (len(c), len(r)) tensor
        The Toeplitz matrix.
    """
    c = torch.as_tensor(c).flatten()
    if r is None:
        r = torch.conj(c)
    else:
        r = torch.as_tensor(r).flatten()

    # Create a large tensor that includes both c (reversed) and r[1:]
    vals = torch.cat((c.flip(dims=(0,)), r[1:]))

    # The output shape of the Toeplitz matrix
    out_shp = len(c), len(r)

    # To replicate `as_strided`, we'll use a combination of `unfold` and `transpose`
    n = vals.stride()[0]
    return vals[len(c) - 1:].unfold(0, out_shp[1], 1).transpose(0, 1).clone()

def toeplitz_1_ch_torch(kernel, input_size):
    # shapes
    k_h, k_w = kernel.shape
    i_h, i_w = input_size
    o_h, o_w = i_h - k_h + 1, i_w - k_w + 1

    # construct 1d conv toeplitz matrices for each row of the kernel
    toeplitz = []
    for r in range(k_h):
        first_col = torch.cat((kernel[r, 0:1], torch.zeros(i_w - k_w)))
        first_row = torch.cat((kernel[r], torch.zeros(i_w - k_w)))
        toeplitz.append(toeplitz_torch(first_col, first_row))

    # construct toeplitz matrix of toeplitz matrices (just for padding=0)
    h_blocks, w_blocks = o_h, i_h
    h_block, w_block = toeplitz[0].shape

    W_conv = torch.zeros((h_blocks, h_block, w_blocks, w_block))

    for i, B in enumerate(toeplitz):
        for j in range(o_h):
            W_conv[j, :, i + j, :] = B

    W_conv = W_conv.reshape(h_blocks * h_block, w_blocks * w_block)

    return W_conv

def toeplitz_mult_ch_torch(kernel, input_size):
    """Compute toeplitz matrix for 2d conv with multiple in and out channels in PyTorch.
    Args:
        kernel: shape=(n_out, n_in, H_k, W_k)
        input_size: (n_in, H_i, W_i)"""

    kernel_size = kernel.shape
    output_size = (kernel_size[0], input_size[1] - (kernel_size[2] - 1), input_size[2] - (kernel_size[3] - 1))
    T = torch.zeros((output_size[0], int(torch.prod(torch.tensor(output_size[1:]))), input_size[0], int(torch.prod(torch.tensor(input_size[1:])))))

    for i, ks in enumerate(kernel):  # loop over output channel
        for j, k in enumerate(ks):  # loop over input channel
            T_k = toeplitz_1_ch_torch(k, input_size[1:])
            T[i, :, j, :] = T_k

    T = T.reshape(int(torch.prod(torch.tensor(output_size))), int(torch.prod(torch.tensor(input_size))))

    return T