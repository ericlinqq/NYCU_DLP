import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=1, help='The number of cycle for kl annealing during training (if use cyclical mode)')

    args = parser.parse_args()
    return args

class kl_annealing():
    def __init__(self, args):
        super().__init__()
        period = args.niter / args.kl_anneal_cycle
        step = 1. / (period * args.kl_anneal_ratio)
        self.L = np.ones(args.niter)
        self.idx = 0
        
        for c in range(args.kl_anneal_cycle):
            v, i = 0.0, 0
            while v <= 1.0 and (int(i+c*period) < args.niter):
                self.L[int(i+c*period)] = v
                v += step
                i += 1

    def update(self):
        self.idx += 1

    def get_beta(self):
        beta = self.L[self.idx]
        self.update()
        return beta

def main():
    args = parse_args()
    kla = kl_annealing(args)
    for i in range(args.niter):
        print(i, kla.get_beta())

if __name__ == '__main__':
    main()
    
