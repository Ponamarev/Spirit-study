import spirit
import numpy as np

def get_magnetization_length(p_state, idx_image=-1, idx_chain=-1):
    """
    Возвращает длину вектора средней намагниченности |M|.
    """
    M = spirit.quantities.get_magnetization(p_state, idx_image=idx_image, idx_chain=idx_chain)
    return np.linalg.norm(M)

with spirit.state.State("input/input.cfg") as p_state:


    # geometry

    # Hamiltonian

    # Boundary conditions
        # spirit.hamiltonian.set_boundary_conditions(p_state, [0, 0, 0], idx_image=-1, idx_chain=-1) # 0 - открытые, 1 - периодические

    # Set system size

    # starting configuration

    # parameters for tasks 4, 5
        # spirit.parameters.llg.set_temperature(p_state, temperature)
        # spirit.parameters.llg.set_timestep(p_state, dt)

    iteration = 0
    step = 1000
    for n in range(100):
        spirit.simulation.start(p_state, spirit.simulation.METHOD_LLG, spirit.simulation.SOLVER_DEPONDT, step)
        iteration += step
