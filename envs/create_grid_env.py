import numpy as np

Ns = 25
Na = 4

def env_step(s,a, eps=0.1):
    row = s // 5
    col = s % 5

    next_states = []

    if a == 0: # Up.

        # W.p. 1-eps. action up suceeds.
        new_row = max(row - 1, 0)
        new_col = col
        new_state = new_row*5 + new_col
        next_states.append((new_state, 1-eps))

        # W.p. eps./2 action right happens.
        new_row = row
        new_col = min(col + 1, 4)
        new_state = new_row*5 + new_col
        next_states.append((new_state, eps/2))

        # W.p. eps./2 action left happens.
        new_row = row
        new_col = max(col - 1, 0)
        new_state = new_row*5 + new_col
        next_states.append((new_state, eps/2))

    elif a == 1: # Down.

        # W.p. 1-eps. action down suceeds.
        new_row = min(row + 1, 4)
        new_col = col
        new_state = new_row*5 + new_col
        next_states.append((new_state, 1-eps))

        # W.p. eps./2 action right happens.
        new_row = row
        new_col = min(col + 1, 4)
        new_state = new_row*5 + new_col
        next_states.append((new_state, eps/2))

        # W.p. eps./2 action left happens.
        new_row = row
        new_col = max(col - 1, 0)
        new_state = new_row*5 + new_col
        next_states.append((new_state, eps/2))

    elif a == 2: # Left.

        # W.p. 1-eps. action left suceeds.
        new_row = row
        new_col = max(col - 1, 0)
        new_state = new_row*5 + new_col
        next_states.append((new_state, 1-eps))

        # W.p. eps./2 action up happens.
        new_row = max(row - 1, 0)
        new_col = col
        new_state = new_row*5 + new_col
        next_states.append((new_state, eps/2))

        # W.p. eps./2 action down happens.
        new_row = min(row + 1, 4)
        new_col = col
        new_state = new_row*5 + new_col
        next_states.append((new_state, eps/2))

    elif a == 3: # Right.

        # W.p. 1-eps. action right suceeds.
        new_row = row
        new_col = min(col + 1, 4)
        new_state = new_row*5 + new_col
        next_states.append((new_state, 1-eps))

        # W.p. eps./2 action up happens.
        new_row = max(row - 1, 0)
        new_col = col
        new_state = new_row*5 + new_col
        next_states.append((new_state, eps/2))

        # W.p. eps./2 action down happens.
        new_row = min(row + 1, 4)
        new_col = col
        new_state = new_row*5 + new_col
        next_states.append((new_state, eps/2))

    return next_states

# print(env_step(4,0))
# print(env_step(4,1))
# print(env_step(4,2))
# print(env_step(4,3))

def get_grid_env_transition_matrix(eps):
    P_matrix = np.zeros((Na, Ns, Ns))
    for action in range(Na):
        for state in range(Ns):
            next_states_info = env_step(state, action, eps)
            probs = np.zeros(Ns)
            for next_state, prob in next_states_info:
                probs[next_state] += prob
            P_matrix[action, state] = probs
    return P_matrix

# P_matrix = get_grid_env_transition_matrix()
# print("P_matrix", P_matrix)
# print(P_matrix[0,8])
# print(P_matrix[1,8])
# print(P_matrix[2,8])
# print(P_matrix[3,8])
