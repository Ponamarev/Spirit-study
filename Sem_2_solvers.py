import spirit
import matplotlib.pyplot as plt
import numpy as np
import time

### Import Spirit modules
from spirit import state
from spirit import configuration
from spirit import simulation
from spirit import system
from spirit import io
from spirit import hamiltonian
from spirit import quantities
from spirit import log
from spirit import parameters
from spirit import geometry


def get_magnetization_length(p_state, idx_image=-1, idx_chain=-1):
    """
    Возвращает длину вектора средней намагниченности |M|.
    """
    M = spirit.quantities.get_magnetization(p_state, idx_image=idx_image, idx_chain=idx_chain)
    return np.linalg.norm(M)


def test_solver_performance(p_state, solver_type, E_stable, max_steps=200):
    """
    Для коректной работы кода нужно предварительно дать системе сойтись.
    :param p_state:         Система спирит
    :param solver_type:     Тестируемый решатель
    :param E_stable:        Энергия системы в со
    :param max_steps:       Пробное количество шагов для IPS (для минимизаторов может потребоваться меньше шагов)
    :return:                dt_crit, ips
    """
    # 1. Поиск критического шага dt
    dt = 0.001
    dt_crit = dt

    while dt < 1.0:  # защита от бесконечного цикла
        spirit.parameters.llg.set_timestep(p_state, dt)
        # запускаем немного шагов, чтобы проверить стабильность
        spirit.simulation.start(p_state, spirit.simulation.METHOD_LLG, solver_type, 1000)
        # M = spirit.quantities.get_magnetization(p_state)
        E = spirit.quantities.get_energy(p_state)

        # Проверка на расходимость по энергии
        if abs((E-E_stable)/E_stable) > 0.15:
            break

        dt *= 1.5  # увеличиваем шаг

    dt_crit = dt


    print(f"Критический шаг для {solver_type}: {dt_crit:.4f} ps")

    # 2. Замер скорости (IPS)

    # Сбрасываем состояние после теста
    spirit.configuration.random(p_state)
    spirit.parameters.llg.set_timestep(p_state, 0.001)
    start_time = time.perf_counter()
    # Важно, чтобы система не успела сойтись
    spirit.simulation.start(p_state, spirit.simulation.METHOD_LLG, solver_type, max_steps)
    end_time = time.perf_counter()

    real_time = end_time - start_time
    ips = max_steps / real_time

    return dt_crit, ips



with spirit.state.State("input/input.cfg") as p_state:


    # geometry
    # System size
    spirit.geometry.set_bravais_lattice_type(p_state, spirit.geometry.BRAVAIS_LATTICE_FCC)
    spirit.geometry.set_n_cells(p_state, n_cells=[20, 20, 20], idx_image=-1, idx_chain=-1)

    # Hamiltonian
    spirit.hamiltonian.set_ddi(p_state, spirit.hamiltonian.DDI_METHOD_FFT, n_periodic_images=[4, 4, 4], radius=0.0, pb_zero_padding=True,
                               idx_image=-1, idx_chain=-1)
    spirit.hamiltonian.set_exchange(p_state, 1, [14], idx_image=-1, idx_chain=-1)
    spirit.hamiltonian.set_dmi(p_state, 1, [0], chirality=1, idx_image=-1, idx_chain=-1)

    # Boundary conditions
    spirit.hamiltonian.set_boundary_conditions(p_state, [1, 1, 1]) # 0 - открытые, 1 - периодические


    # starting configuration
    spirit.configuration.plus_z(p_state)
    # spirit.configuration.random(p_state)

    # parameters for tasks 4, 5
    spirit.parameters.llg.set_temperature(p_state, 10)
    spirit.parameters.llg.set_timestep(p_state, 0.005)
    spirit.parameters.llg.set_damping(p_state, 0.4)

    M_data = []
    E_data = []
    It_arg = []
    iteration = 0
    step = 100
    for item in range(100):
        iteration += step
        It_arg.append(iteration)
        spirit.simulation.start(p_state, spirit.simulation.METHOD_LLG, spirit.simulation.SOLVER_DEPONDT, step)

        E_data.append(spirit.system.get_energy(p_state))
        M_data.append(get_magnetization_length(p_state))



    plt.plot(It_arg, E_data)
    # naming the x axis
    plt.xlabel('I [I]')
    # naming the y axis
    plt.ylabel('Energy [a.u.]')
    # giving a title to my graph
    plt.title('E(I)')
    # function to show the plot
    plt.show()

    plt.plot(It_arg, M_data)
    # naming the x axis
    plt.xlabel('I [I]')
    # naming the y axis
    plt.ylabel('M [a.u.]')
    # giving a title to my graph
    plt.title('M(I)')
    # function to show the plot
    plt.show()

