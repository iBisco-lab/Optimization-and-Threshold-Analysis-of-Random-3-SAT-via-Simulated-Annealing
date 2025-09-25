#%%

###Performance of Simulated Annealing algorithm###

import matplotlib.pyplot as plt
import SimAnn
import KSAT

#(To make this function work the return statement inside the beta 
#loop in SimAnn must be commented)

#Function to plot acceptance rates

def display(mcmc_steps,anneal_steps,
                   beta0, beta1, N=400, M=200, K=3, seed=123):
    
    ksat = KSAT.KSAT(N, M, K, seed=seed)
    
    best, acc_rates = SimAnn.simann(ksat, mcmc_steps=mcmc_steps, 
                  anneal_steps=anneal_steps,
                  beta0=beta0, beta1=beta1,
                  seed=123)

    # Plot acceptance rates
    
    plt.plot(anneal_steps, acc_rates, '-o')
    plt.xlabel('Annealing schedule')
    plt.ylabel('Acceptance Rate')
    plt.title(f'Acceptance Rates (mcmc={mcmc_steps}, anneal={anneal_steps}, β0={beta0}, β1={beta1})')
    plt.grid(True)
    plt.show()

display(400, 20, 1, 10)

#%%

#Plotting acceprance rate for different M

import matplotlib.pyplot as plt
import SimAnn
import KSAT
import time
from tqdm import tqdm

def solving_probability(totinst, seed = 123,):
    
    #Useful to keep track of running time
    start_time = time.time()

    #Choosing values of M to analyze
    listM = [400, 600, 800, 1000]

    #Adjust mcmc_steps accordingly
    mcmc_steps = [400, 1200, 1600, 2000]

    #Keep track of success rate
    probArr = []

    #Outer loop going through every M we chose
    for j in tqdm(range(len(listM))):

        solved = 0

        #Running the program multiple times to collect a meaningful sample
        for i in range(totinst):

            ksat = KSAT.KSAT(N = 200, M = listM[j], K = 3, seed=seed)

            best, _ = SimAnn.simann(ksat, mcmc_steps = mcmc_steps[j], 
                                            anneal_steps = 20,
                                            beta0 = 1, beta1 = 10,
                                            seed = seed)

            #Keep track of solved instances
            if best == 0:
                solved += 1

        probArr.append(solved/totinst)
    
    total_time = time.time() - start_time  # Calculate total time
    print(f"\nTotal runtime: {total_time:.2f} seconds")
    
    
    #Plot results
    print(f'probArr = {probArr}')
    plt.plot(listM, probArr,  '-o')
    plt.xlabel('M values')
    plt.ylabel('P(N, M)')
    plt.title('3-SAT Solving Probability vs Number of Clauses (N=200)')
    plt.grid(True)
    plt.show()

solving_probability(30)

#%%

###General 3-SAT Properties###

#Given the scope of this task, I made another change to SimAnn 
# to make the code go faster:
# when the cost = 0, the code is stopped

import matplotlib.pyplot as plt
import SimAnn
import KSAT
import time
from tqdm import tqdm

def solving_probability(totinst, seed = None,):
    
    start_time = time.time()
    #Choosing values of M to analyze
    #and adjust parameters accordingly
    listM = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    mcmc_steps = [200, 200, 400, 600, 800, 1000, 1100, 1400, 1700]
    anneal_steps = [20, 20, 20, 20, 20, 20, 20, 20, 20]
    beta0 = [1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    beta1 = [10, 10, 10, 10, 15, 20, 25, 30, 30]

    #Keeping track of success rate
    probArr = []

    #Outer loop going through every M we chose
    for j in tqdm(range(len(listM))):

        solved = 0

        #Running the program multiple times to collect a meaningful sample
        for i in range(totinst):

            ksat = KSAT.KSAT(N = 200, M = listM[j], K = 3, seed=seed)

            best, _ = SimAnn.simann(ksat, mcmc_steps = mcmc_steps[j], 
                                            anneal_steps = anneal_steps[j],
                                            beta0 = beta0[j], beta1 = beta1[j],
                                            seed = seed)

            #Keep track of solved instances
            if best == 0:
                solved += 1

        probArr.append(solved/totinst)
    
    total_time = time.time() - start_time  # Calculate total time
    print(f"\nTotal runtime: {total_time:.2f} seconds")
    
    
    #Plot results
    print(f'probArr = {probArr}')
    plt.plot(listM, probArr,  '-o')
    plt.xlabel('M values')
    plt.ylabel('P(N, M)')
    plt.title('3-SAT Solving Probability vs Number of Clauses (N=200)')
    plt.grid(True)
    plt.show()

solving_probability(30)



#%%

#Identifying the algorithmic threshold M(Alg)(N = 200)
#Using the binary search algorithm
#We'll use the same parameters mcmc, anneal_steps, beta0, beta1, seed
#for each M because the threshold changes on different parameters
#From the previous question we know that the threshold is between 
#M = 725 and M = 775

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import KSAT
import SimAnn

def find_threshold_detailed(N=200, M_min=600, M_max=700, num_points=20, instances_per_point=30, seed=None):

    # Creating evenly spaced M values
    M_values = np.linspace(M_min, M_max, num_points, dtype=int)
    probabilities = []
    
    # For each M, compute success probability
    for M in tqdm(M_values, desc="Testing M values"):
        solved = 0
        mcmc_steps = int(M*2)  # Scale MCMC steps with M
        
        for i in range(instances_per_point):
            
            ksat = KSAT.KSAT(N=N, M=M, K=3, seed=seed)
            
            best, _ = SimAnn.simann(
                ksat,
                mcmc_steps=mcmc_steps,
                anneal_steps=30,
                beta0=0.1,
                beta1=25,
                seed=seed
            )
            
            #Keep track of solved instances
            if best == 0:
                solved += 1
        
        #Compute solving probability for each M
        prob = solved / instances_per_point
        probabilities.append(prob)
    
    # Find the threshold (M where probability ≈ 0.5)
    #This might need manual adjustments afterward
    threshold_idx = np.argmin(np.abs(np.array(probabilities) - 0.5))
    threshold_M = M_values[threshold_idx]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(M_values, probabilities, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=threshold_M, color='r', linestyle='--', label=f'Threshold ≈ {threshold_M}')
    plt.axhline(y=0.5, color='g', linestyle='--', label='P = 0.5')
    
    plt.xlabel('Number of Clauses (M)')
    plt.ylabel('Solving Probability')
    plt.title('3-SAT Algorithmic Threshold Detection (N=200)')
    plt.grid(True)
    plt.legend()
    
    return M_values, probabilities, threshold_M

# Run the program
M_values, probs, threshold = find_threshold_detailed(
    M_min=650,
    M_max=750,
    num_points=20,
    instances_per_point=30,
    seed=None
)

print(f"Estimated threshold M(Alg) ≈ {threshold}")


# %%
