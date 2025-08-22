import numpy as np
from mpi4py import MPI
import time

def compute_forces(pos, mass, G, softening):
    """Compute forces on all particles (parallelized over MPI ranks)."""
    N = pos.shape[0]
    forces = np.zeros_like(pos)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Split work across ranks
    chunk = N // size
    start = rank * chunk
    end = (rank + 1) * chunk if rank != size - 1 else N

    # Local force computation
    for i in range(start, end):
        for j in range(N):
            if i == j:
                continue
            r = pos[j] - pos[i]
            dist = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2 + softening**2)
            forces[i] += G * mass[i] * mass[j] * r / (dist**3)

    # Sum forces across all ranks
    forces_global = np.zeros_like(forces)
    comm.Allreduce(forces, forces_global, op=MPI.SUM)
    return forces_global

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Simulation parameters
    N = 1000  # Number of particles
    G = 1.0   # Gravitational constant
    softening = 0.1  # Softening length to avoid singularities
    dt = 0.01  # Time step
    steps = 100  # Number of steps

    # Initialize particles (same on all ranks)
    np.random.seed(42)
    pos = np.random.randn(N, 3)
    vel = np.zeros((N, 3))
    mass = np.ones(N)

    # Main loop
    for _ in range(steps):
        forces = compute_forces(pos, mass, G, softening)
        vel += forces * dt / mass[:, np.newaxis]
        pos += vel * dt

        # Optional: Save snapshot (rank 0 only)
        if rank == 0 and _ % 10 == 0:
            np.savez(f"snapshot_{_}.npz", pos=pos, vel=vel)

    if rank == 0:
        print("Simulation complete.")

if __name__ == "__main__":
    main()
