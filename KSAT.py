import numpy as np
from copy import deepcopy

class KSAT:
    def __init__(self, N, M, K, seed = None):
        if not (isinstance(K, int) and K >= 2):
            raise Exception("k must be an int greater or equal than 2")
        self.K = K
        self.M = M
        self.N = N

        ## Optionally set up the random number generator state
        if seed is not None:
            np.random.seed(seed)
    
        
        # s is the sign matrix
        s = np.random.choice([-1,1], size=(M,K))
        
        # index is the matrix reporting the index of the K variables of the m-th clause 
        index = np.zeros((M,K), dtype = int)        
        for m in range(M):
            index[m] = np.random.choice(N, size=(K), replace=False)
            
        # Dictionary for keeping track of literals in clauses
        clauses = []   
        for n in range(N):
            clauses.append([i for i, row in enumerate(index) if n in row])
        
        self.s, self.index, self.clauses = s, index, clauses        
        
        ## Inizializza la configurazione
        x = np.ones(N, dtype=int)
        self.x = x
        self.init_config()


    ## Initialize (or reset) the current configuration
    def init_config(self):
        N = self.N 
        self.x[:] = np.random.choice([-1,1], size=(N))
    
    def __repr__(self):
        return f'{str(self.x)}'
    
    def __getitem__(self,i):
        return self.x[i]
        
    ## Definition of the cost function
    # Here you need to complete the function computing the cost using eq.(4) of pdf file
    def cost(self, inf_clauses = None):
        #Computing a matrix with the values of x at position index[ij]
        if inf_clauses is None:
            x, s = self.x[self.index], self.s
        else:

            x, s = self.x[self.index[inf_clauses]], self.s[inf_clauses]

        #With this formula we verify how many clauses are unsatisfied
        satisfiability = np.prod((1-s * x)/2, axis = 1)

        #Sum of unsatisfied clauses (total cost)
        return np.sum(satisfiability)
    
    ## Propose a valid random move. 
    def propose_move(self):
        N = self.N
        move = np.random.choice(N)
        #print(f'move = {move}')
        return move
    
    ## Modifying the current configuration, accepting the proposed move
    def accept_move(self, move):
        self.x[move] *= -1

    ## Compute the extra cost of the move (new-old, negative means convenient)
    
    def compute_delta_cost(self, move):

        #First we compute only the clauses affected by the move
        influenced_clauses = self.clauses[move]

        #we compute the cost of these clauses before the move
        old_cost = self.cost(influenced_clauses)

    
        #temporarely flip x

        self.accept_move(move)

        #we compute the cost of the clauses after the move
        new_cost = self.cost(influenced_clauses)

        #take back x to inital value
        self.accept_move(move)

        #Computing the difference
        delta_cost = new_cost-old_cost

        return delta_cost

    ## Make an entirely independent duplicate of the current object.
    def copy(self):
        return deepcopy(self)
    
    ## The display function should not be implemented
    def display(self):
        pass

