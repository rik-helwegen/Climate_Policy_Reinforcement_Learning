# standard imports
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
# project imports
from dice2007cjl import Dice2007cjl as Dice
import pickle
def find_optimal_policy(stepsize=10):
    """Find optimal policy for our model.

    Args
      stepsize: keep policy fixed for number of years
    """
    # create environment
    env = Dice()

    # determine numbers of steps
    assert int(env.t_max/stepsize) == env.t_max/stepsize  # check stepsize
    n_steps = int(env.t_max/stepsize)

    # create map from actionvec to total utility
    def fullrun(muvec):
        # initialize
        env.reset()
        utility = 0
        # execute run
        list_of_rewards = []
        for step in range(n_steps):
            for _ in range(stepsize):
                _, r, _ = env.step(muvec[step])
                utility += r
                list_of_rewards.append(r)
        # return minus utility (since scipy offers minimize)
        return -(utility/194)+381800  # scale to ease optimization

    # set initial guess and bounds for actions
    initialguess = 0.99*np.ones(n_steps)
    bounds = [(0,1)]*n_steps

    # perform minimization
    print('Starting optimization...')
    res = minimize(fullrun, initialguess,
                   bounds=bounds,
                   method='slsqp',
                   options={'ftol': 1e-10, 'maxiter': 500, 'disp':True}
                   )

    return res

if __name__ == '__main__':
    # settings
    stepsize = 10
    # get data and vector with years
    res = find_optimal_policy(stepsize)
    timevec = list(range(2005, 2005+len(res.x)*stepsize, stepsize))
    f = open('results_200.pckl', 'wb')
    pickle.dump(res, f)
    f.close()
    # plot result
    plt.plot(timevec, res.x)
    plt.xlabel('Year')
    plt.ylabel('Abatement (ratio of output)')
    plt.show()
