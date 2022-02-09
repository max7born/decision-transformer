import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

datasets = [
    'hopper-expert-v2',
    'mujoco-pendulum-expert'
]

for dataset in datasets:
    print(dataset)
    with open(f'{dataset}.pkl', 'rb') as f:
        data = pickle.load(f)

    actions = np.array([d['actions'] for d in data])
    plt.plot(range(1000), actions[0], linewidth=0.1)
    plt.show()
    actions = np.concatenate(actions)


    print('Num traj: ', len(data))
    print('Max: ', np.amax(actions))
    print('Min: ', np.amin(actions))
    print('Closest to 0: ', np.amin(np.abs(actions)))

    print('-'*50)
