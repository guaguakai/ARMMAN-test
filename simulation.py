import numpy as np

if __name__ == '__main__':
    P00, P10 = 0.5, 0.4
    A00, A10 = 0.4, 0.2
    P_passive = np.array([
            [P00, 1 - P00],
            [P10, 1 - P10]
            ])
    P_active = np.array([
            [A00, 1 - A00],
            [A10, 1 - A10]
            ])

    gamma = 0.99

    # value iteration
    def value_iteration(m, P_passive, P_active, num_iterations=100, gamma=0.9):
        V = np.zeros(2)
        Q = np.zeros((2,2))
        lr = 0.1
        for _ in range(num_iterations):
            new_Q = np.zeros((2,2))
            new_Q[0,0] = m + gamma * (P_passive[0,0] * V[0] + P_passive[0,1] * V[1])
            new_Q[0,1] = gamma * (P_active[0,0] * V[0] + P_active[0,1] * V[1])
            new_Q[1,0] = m + 1 + gamma * (P_passive[1,0] * V[0] + P_passive[1,1] * V[1]) 
            new_Q[1,1] = 1 + gamma * (P_active[1,0] * V[0] + P_active[1,1] * V[1])

            Q = Q * (1 - lr) + new_Q * lr
            V[0] = max(Q[0,0], Q[0,1])
            V[1] = max(Q[1,0], Q[1,1])

        return V, Q

    # Binary search
    step_size = 1
    m = 0
    delta = 1e-4
    for iteration in range(100):
        V, Q = value_iteration(m, P_passive, P_active, gamma=gamma)
        print('Iteration {}, V: {}, Q: {}'.format(iteration, V, Q))
        if Q[0,0] > Q[0,1] + delta: # the value of passive is larger than the value of active
            m -= step_size
        elif Q[0,0] < Q[0,1] - delta: # the value of passive is smaller than the value of active
            m += step_size
        else:
            break
        step_size /= 2

    print('Whittle index: {}'.format(m))

    # Using matrix form to solve
    M1 = np.array([
        [1, gamma * P_passive[0,0] - 1, gamma * P_passive[0,1]], 
        [0, gamma * P_active[0,0] - 1, gamma * P_active[0,1]], 
        [1, gamma * P_passive[1,0], gamma * P_passive[1,1] - 1]
        ])

    M2 = np.array([
        [1, gamma * P_passive[0,0] - 1, gamma * P_passive[0,1]], 
        [0, gamma * P_active[0,0] - 1, gamma * P_active[0,1]], 
        [0, gamma * P_active[1,0], gamma * P_active[1,1] - 1]
        ])

    v = np.array([0, 0, -1])

    v1 = np.linalg.solve(M1, v)
    v2 = np.linalg.solve(M2, v)
    print('Whittle index candidate: {}, {}'.format(v1[0], v2[0]))

