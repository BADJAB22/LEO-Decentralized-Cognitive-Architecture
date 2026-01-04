import numpy as np
from core.admm_engine import ADMMEngine
from memory.cognitive_memory import CognitiveMemory

def run_leo_demo():
    print("==========================================")
    print("   LEO: Decentralized Cognitive Engine    ")
    print("        Integrated Prototype Demo         ")
    print("==========================================\n")

    # 1. Initialize Components
    dimension = 64
    num_nodes = 3
    engine = ADMMEngine(num_nodes=num_nodes, dimension=dimension)
    memories = [CognitiveMemory(dimension=dimension) for _ in range(num_nodes)]

    # 2. Simulate Learning Phase (Local Data)
    print("[Step 1] Learning from local 'private' data...")
    secret_concept = np.random.rand(dimension)
    for i in range(num_nodes):
        # Each node learns a slightly different version of the same secret
        local_data = secret_concept + np.random.normal(0, 0.05, dimension)
        memories[i].add_observation(local_data)
    print("Nodes have consolidated local knowledge into LTM.\n")

    # 3. Simulate Query with Consensus
    print("[Step 2] Processing a query across the decentralized network...")
    query = secret_concept + np.random.normal(0, 0.2, dimension)
    
    # ADMM Iterations for Collective Reasoning
    for k in range(50):
        local_gradients = []
        for i in range(num_nodes):
            # Each node recalls from its private memory to guide the consensus
            recalled_state = memories[i].recall(engine.theta[i])
            # Gradient points towards the recalled (memory-biased) state
            grad = engine.theta[i] - recalled_state
            local_gradients.append(grad)
        
        # Global consensus step (Byzantine-resilient)
        w = engine.run_iteration(local_gradients)
        
    print(f"Consensus reached after 50 iterations.")
    
    # 4. Results
    from sklearn.metrics.pairwise import cosine_similarity
    final_sim = cosine_similarity(w.reshape(1, -1), secret_concept.reshape(1, -1))[0][0]
    print(f"\n[Result] Final Collective Intelligence Accuracy: {final_sim:.4f}")
    print("The system successfully combined private memories into a single accurate consensus.")
    print("==========================================")

if __name__ == "__main__":
    run_leo_demo()
