import argparse

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=4, help='Batch size')
    parser.add_argument('-c', type=int, default=5, help='Number of classes in dataset')
    parser.add_argument('-t', type=check_network_type, default=18, help='ResNet18 (18) or ResNet50 (50)')
    parser.add_argument('-f', type=bool, default=True, help='Feature extract or not')
    parser.add_argument('-e', type=int, default=5, help='Number of epochs')
    parser.add_argument('-l', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-m', type=float, default=0.9, help='Momentum')
    parser.add_argument('-w', type=float, default=5e-4, help='Weight decay')
    
    return parser.parse_args()

def check_network_type(input):
    int_value = int(input)

    if input != 18 and input != 50:
        raise argparse.ArgumentTypeError(f"Network type should be 18 or 50")
    
    return int_value