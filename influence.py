import numpy as np

def influence_theta(agent, history):
    hessian = agent.hessian(history)
    inv = np.linalg.pinv(hessian)
    delta_thetas = np.array([
        -inv @ agent.grad_single_loss(trans) for trans in history
    ])

    return delta_thetas
