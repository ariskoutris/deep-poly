from deeppoly import *


def sample():
    # Example from DeepPoly Lecture
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

def simulate(w):
    # Example from Exercise 5
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
    dp_layers = propagate_sample(model, x, eps, min_val=-1, max_val=1)
    bounds = deeppoly_backsub(dp_layers)

    ub = bounds.ub.flatten()
    return ub[1].item()

def plot_upper_bounds():
    w_vals = np.linspace(-3, 5, 500)
    ub_vals = []
    for w in w_vals:
        ub_vals.append(simulate(w))
    plt.plot(w_vals, ub_vals)
    plt.show()


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    import argparse

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', datefmt='%X')
    logger = logging.getLogger(__name__)

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

    sample()
    simulate(weight)
