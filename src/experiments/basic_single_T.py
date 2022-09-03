from enum import Enum
import argparse as ap
import matplotlib.pyplot as plt
import numpy as np
from ..model import XYModel

class StepType(Enum):
    METROPOLIS = 0
    WOLFF = 1

if __name__ == "__main__":

    parser = ap.ArgumentParser("Basic experiment with user-defined temperature")

    parser.add_argument("-T", type=float, default=1.0, help="Temperature")
    parser.add_argument("-s", type=int, default=1000, help="Number of steps")
    parser.add_argument("-N", type=int, default=10, help="System size")
    parser.add_argument("-dphis", type=float, default=0.1, help="Step size")
    parser.add_argument("-seed", type=int, default=None, help="Random seed")
    parser.add_argument("-step", type=int, default=0, help="Step type (0 = Metropolis, 1 = Wolff)")
    
    args = parser.parse_args()

    m = XYModel(args.N, args.T, XYModel.InitialState.RANDOM, args.seed)

    accepted = 0
    traj = []

    stype = StepType(args.step)
    
    for i in range(args.s):
        if stype == StepType.METROPOLIS:
            for j in range(args.N**2):
                accepted += 1 if m.metropolis_step(args.dphis) else 0
        elif stype == StepType.WOLFF:
            accepted += m.wolff_step()
        traj.append([m.E, m.M])  

    print(f"Spin flips per step: {accepted/args.s}")

    traj = np.array(traj)
    plt.plot(traj[:,0], label="E")
    plt.legend()
    plt.show()    
    
    plt.plot(traj[:,1], label="M")
    plt.legend()
    plt.show()    

    nrng = np.arange(m.N)
    field = m.state/2
    plt.quiver(np.tile(nrng, m.N), np.repeat(nrng, m.N), field[:,0], field[:,1])    
    plt.show()