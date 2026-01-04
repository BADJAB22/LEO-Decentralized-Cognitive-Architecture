import numpy as np

class ADMMEngine:
    """
    Implements the Byzantine-Resilient ADMM for Collective Reasoning
    as described in Section 1 of the LEO Whitepaper.
    """
    def __init__(self, num_nodes, dimension, rho=1.0, gamma=0.1):
        self.N = num_nodes
        self.d = dimension
        self.rho = rho
        self.gamma = gamma
        
        # Local states (theta_i)
        self.theta = [np.zeros(dimension) for _ in range(num_nodes)]
        # Global consensus state (w)
        self.w = np.zeros(dimension)
        # Dual variables (u_i)
        self.u = [np.zeros(dimension) for _ in range(num_nodes)]
        # Trust coefficients (T_i)
        self.trust = np.ones(num_nodes)

    def local_update(self, node_id, local_gradient, alpha_i=1.0):
        """
        Step 1: Local Computation
        theta_i^{k+1} = prox(f_i + (rho/2)||theta - w + u||^2)
        Simplified as a gradient step for the prototype.
        """
        # Gradient descent step towards local objective and consensus
        consensus_term = self.theta[node_id] - self.w + self.u[node_id]
        self.theta[node_id] = self.theta[node_id] - 0.01 * (alpha_i * local_gradient + self.rho * consensus_term)
        return self.theta[node_id]

    def global_update(self):
        """
        Step 3: Global Consensus using Byzantine-resilient aggregation (Median)
        W^{k+1} = Median({T_i * (theta_i + u_i)})
        """
        aggregated_states = []
        for i in range(self.N):
            aggregated_states.append(self.trust[i] * (self.theta[i] + self.u[i]))
        
        # Using median for Byzantine resilience as suggested in the paper
        self.w = np.median(aggregated_states, axis=0)
        
        # Apply global coherence/safety potential (simplified)
        self.w = self.w - self.gamma * 0.01 * self.w # Simplified safety gradient
        return self.w

    def dual_update(self):
        """
        Step 4: Dual Variable Update
        u_i^{k+1} = u_i^k + theta_i^{k+1} - w^{k+1}
        """
        for i in range(self.N):
            self.u[i] = self.u[i] + self.theta[i] - self.w
        return self.u

    def run_iteration(self, local_gradients):
        """Runs one full ADMM cycle."""
        for i in range(self.N):
            self.local_update(i, local_gradients[i])
        self.global_update()
        self.dual_update()
        return self.w

if __name__ == "__main__":
    # Simple test: 5 nodes trying to agree on a value [10, 20]
    engine = ADMMEngine(num_nodes=5, dimension=2)
    target = np.array([10.0, 20.0])
    
    print("Starting ADMM Consensus Simulation...")
    for k in range(100):
        # Each node has a slightly noisy gradient towards the target
        gradients = [ (engine.theta[i] - target) + np.random.normal(0, 0.1, 2) for i in range(5)]
        w = engine.run_iteration(gradients)
        if k % 20 == 0:
            print(f"Iteration {k}: Global State w = {w}")
    
    print(f"Final Consensus: {w}")
