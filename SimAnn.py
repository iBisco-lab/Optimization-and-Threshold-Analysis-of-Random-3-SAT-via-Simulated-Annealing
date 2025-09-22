import numpy as np

## Stochastically determine whether to accept a move according to the
## Metropolis rule (valid for symmetric proposals)
def accept(delta_c, beta):
    ## If the cost doesn't increase, we always accept
    if delta_c <= 0:
        return True
    ## If the cost increases and beta is infinite, we always reject
    if beta == np.inf:
        return False
    ## Otherwise the probability is going to be somewhere between 0 and 1
    p = np.exp(-beta * delta_c)
    ## Returns True with probability p
    return np.random.rand() < p

## The simulated annealing generic solver.
## Assumes that the proposals are symmetric.
## The `probl` object must implement these methods:
##    init_config()               # returns None [changes internal config]
##    cost()                      # returns a real number
##    propose_move()              # returns a (problem-dependent) move - must be symmetric!
##    compute_delta_cost(move)    # returns a real number
##    accept_move(move)           # returns None [changes internal config]
##    copy()                      # returns a new, independent object
## NOTE: The default beta0 and beta1 are arbitrary.
def simann(probl,
           anneal_steps=10, mcmc_steps=100,
           beta0=0.1, beta1=10.0,
           seed=None, debug_delta_cost=False):
    ## Optionally set up the random number generator state
    if seed is not None:
        np.random.seed(seed)

    # Set up the list of betas.
    # First allocate an array with the required number of steps
    beta_list = np.zeros(anneal_steps)
    # All but the last one are evenly spaced between beta0 and beta1 (included)
    beta_list[:-1] = np.linspace(beta0, beta1, anneal_steps - 1)
    # The last one is set to infinity
    beta_list[-1] = np.inf

    # Set up the initial configuration, compute and print the initial cost
    probl.init_config()
    c = probl.cost()
    #print(f"initial cost = {c}")
    #print(probl.x)
    
    ## Keep the best cost seen so far, and its associated configuration.
    best = probl.copy()
    best_c = c

    #array of acceptance rate to keep track of the evolution of acceptance rate
    acc_rates = []


    # Main loop of the annealing: Loop over the betas
    for beta in beta_list:
        ## At each beta, we want to record the acceptance rate, so we need a
        ## counter for the number of accepted moves
        accepted = 0
        # For each beta, perform a number of MCMC steps
        for t in range(mcmc_steps):
            move = probl.propose_move()
            #print(f"Proposed move: {move}")
            delta_c = probl.compute_delta_cost(move)
            #print(f"Delta cost: {delta_c}")
            ## Optional (expensive) check that `compute_delta_cost` works
            if debug_delta_cost:
                probl_copy = probl.copy()
                probl_copy.accept_move(move)
                assert abs(c + delta_c - probl_copy.cost()) < 1e-10
            ## Metropolis rule
            #print(probl.x, c, move, delta_c, accept(delta_c, beta))
            
            if accept(delta_c, beta):
                #print(f"Accepted move: {move} with delta cost: {delta_c}")
                probl.accept_move(move)
                #print(probl.x)
                c += delta_c
                accepted += 1
                if c <= best_c:
                    best_c = c
                    best = probl.copy()

                #This break is needed to speed up running time when investigating 
                #solving probabilities
                if best_c == 0:
                    return best_c, np.array(acc_rates)
        
            
        acc_rates.append(accepted/mcmc_steps)
        #print(f"acc.rate={accepted/mcmc_steps} beta={beta} c={c} [best={best_c}]")
    
        
    ## Return the best instance and acc_rates, wihch are necessary for the assignment
    return best_c, np.array(acc_rates)
