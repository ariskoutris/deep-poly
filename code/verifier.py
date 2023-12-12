import argparse
import torch

from networks import get_network
from utils.loading import parse_spec
import deeppoly, box

import logging

# Configure logging. Set level to [NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL] (in order) to control verbosity.
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', datefmt='%X')

DEVICE = "cpu"
LOG = True

def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int
) -> bool:
    return deeppoly.certify_sample(net, inputs, true_label, eps)


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    parser.add_argument("--labels", type=str, required=False, help="Path to file containing the labels of the dataset.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    logging.info(f"Verifying {args.spec} (model={args.net}, epsilon={eps}, true_label={true_label})")

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)
    logging.info(net)

    verified_status = None
    if args.labels:
        labels_file = args.labels
        image_name = args.spec.split('/')[-1]
        with open(labels_file, 'r') as file:
            for line in file:
                row = line.strip().split(',')
                if row[0] == args.net and row[1] == image_name:
                    verified_status = (row[2] == 'verified')
                    break

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))
    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        status_msg = "verified"
        if verified_status != None: status_msg += '\t✅' if verified_status else '\t🛑 (❗️)'
        print(status_msg)
    else:
        status_msg = "not verified"
        if verified_status != None: status_msg += u'\t🛑' if verified_status else '\t✅'
        print(status_msg)


if __name__ == "__main__":
    main()
