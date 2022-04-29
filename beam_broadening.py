from numpy.random import default_rng
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import itertools
import pandas
from scipy.spatial.transform import Rotation as R


landing_energy = '200000'
elements = ['C', 'Os']

DCS_data = {element: pandas.read_csv('DCS_' + element + '_' + landing_energy + '_eV.csv', skiprows=9)
            for element in elements}

a02 = 2.8002852E-21
totals = {'C': a02 * 1.787180E-2, 'Os': a02 * 5.178653E-1}

base_bremsstrahlung = 1.425

# number of carbon atoms in 1nm * 100nm * 100nm
density_carbon = 2.1E3  # kg/m^3
mass_carbon_nucleus = 12 * 1.66E-27
N_carbon = 1E-9 * 100E-9 * 100E-9 * density_carbon / mass_carbon_nucleus
print(N_carbon)
prob_scatter = {'C': totals['C'] * N_carbon / 100E-9 / 100E-9, 'Os': 0.002}
print(prob_scatter['C'])

# decay fun: e^(-prob_scatter*dist)
#plt.plot(np.arange(1000),np.exp(-prob_scatter['C']*np.arange(1000)))
#plt.show()
# lifetime distribution prob_scatter['C']*e^(-prob_scatter['C']*dist)
#plt.plot(np.arange(1000),prob_scatter['C']*np.exp(-prob_scatter['C']*np.arange(1000)))
#plt.show()
rays = []
xx=[]
rng = default_rng()
random.seed()
thickness = 200
weights = []


def np_access(element_, idx_):
    return DCS_data[element_].to_numpy()[:, idx_]


cum_weights = {element: np.cumsum(  np.sin(np.radians(np_access(element, 0)[1:]))
                                  * np_access(element, 1)[1:]
                                  * np.diff(np_access(element, 0))).tolist()
               for element in elements}


def make_ray(mylist, start, dir, energy):
    depth = rng.exponential(size=(1), scale=1/(sum(prob_scatter.values())))[0]
    end = start + dir * depth
    if 0 < end[2] < thickness:
        scattered_at = random.choices(elements, weights=[prob_scatter[element] for element in elements])[0]
        angle_todo = np.radians(random.choices(np_access(scattered_at, 0)[1:],
                                               cum_weights=cum_weights[scattered_at])[0])
        # create vector orthogonal to dir
        ortho = np.cross(dir, [1, 0, 0])
        assert(np.linalg.norm(ortho) > 0)
        dir_todo = (R.from_rotvec(random.uniform(0, 2*np.pi)*dir)*R.from_rotvec(angle_todo*ortho)).apply(dir)
        energy_todo = energy - base_bremsstrahlung * np.linalg.norm(dir-dir_todo)
        make_ray(mylist,
                 start=end,
                 dir=dir_todo,
                 energy=energy_todo)
    else:
        end = start + dir * 10_000
    mylist.append({'start': start, 'end': end, 'dir': dir, 'energy': energy})


for idx in range(160):
    make_ray(rays,
             start=[idx/40-2, 0, 0],
             dir=np.array([0, 0, 1]),
             energy=random.normalvariate(mu=float(landing_energy), sigma=0.9))
plt.figure(figsize=(12, 6))
for ray in rays:
    plt.plot([ray['start'][0], ray['end'][0]],
             [ray['start'][2], ray['end'][2]], 'k', linewidth=0.5)
plt.ylim([0, 100])
plt.xlim([-4, 4])
plt.xlabel('nm')
plt.ylabel('nm')
plt.gca().invert_yaxis()
plt.legend(['n=160'])
plt.savefig('test'+str(random.random())+'.pdf')
plt.show()
