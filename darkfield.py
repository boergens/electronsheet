import pandas
import matplotlib.pyplot as plt
from numpy.random import default_rng
import itertools
import numpy as np
import random
landing_energy = '200000'
elements = ['C', 'Os']

DCS_data = {element: pandas.read_csv('DCS_' + element + '_' + landing_energy + '_eV.csv', skiprows=9)
            for element in elements}

def np_access(element_, idx_):
    return DCS_data[element_].to_numpy()[:, idx_]


cum_weights = {element: np.cumsum(  np.sin(np.radians(np_access(element, 0)[1:]))
                                  * np_access(element, 1)[1:]
                                  * np.diff(np_access(element, 0))).tolist()
               for element in elements}
print(len(cum_weights['C']))
plt.figure(figsize=(6,6))
rng = default_rng()
numbers = {'Os':1500,'C':2635}
for scattered_at in elements:
    angle_todo = np.array(random.choices(np_access(scattered_at, 0)[1:],
                                         cum_weights=cum_weights[scattered_at],
                                         k=numbers[scattered_at]))
    print(scattered_at)
    print(np.logical_and(0.1 < angle_todo, angle_todo < 0.75).sum())
    angle_secondary = rng.uniform(0,2*np.pi, size=(numbers[scattered_at]))
    plt.scatter(angle_todo*np.sin(angle_secondary),
                angle_todo*np.cos(angle_secondary),
                label=scattered_at,
                s=1)

plt.scatter(rng.normal(scale=0.01,size=(100)),
            rng.normal(scale=0.01,size=(100)),
            label="primary",
            s=1)
fig = plt.gcf()
ax = fig.gca()
circle1 = plt.Circle((0, 0), 0.1, color='k', fill=False)
circle2 = plt.Circle((0, 0), 0.75, color='k', fill=False)
plt.legend()
plt.xlim(-1,1)
plt.ylim(-1,1)
ax.add_patch(circle1)
ax.add_patch(circle2)
plt.show()

