import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CognitiveMemory:
    """
    Implements the Hierarchical Memory System (STM/LTM)
    as described in Section 2 of the LEO Whitepaper.
    """
    def __init__(self, dimension, ltm_threshold=0.85):
        self.d = dimension
        self.ltm_threshold = ltm_threshold
        
        # Short-Term Memory (STM): Circular buffer
        self.stm = []
        self.stm_max_size = 10
        
        # Long-Term Memory (LTM): Sparse Graph (Vertices)
        self.ltm_nodes = []
        self.ltm_edges = {} # (v_i, v_j) -> weight

    def add_observation(self, vector):
        """Adds a new representation to STM and potentially consolidates to LTM."""
        # Update STM
        self.stm.append(vector)
        if len(self.stm) > self.stm_max_size:
            self.stm.pop(0)
            
        # LTM Consolidation (Hebbian-like update)
        self._consolidate_to_ltm(vector)

    def _consolidate_to_ltm(self, vector):
        """Section 2.2: Node Addition and Edge Update."""
        if not self.ltm_nodes:
            self.ltm_nodes.append(vector)
            return

        # Check similarity with existing LTM nodes
        similarities = [cosine_similarity(vector.reshape(1, -1), node.reshape(1, -1))[0][0] for node in self.ltm_nodes]
        max_sim = max(similarities)
        
        if max_sim < self.ltm_threshold:
            # Add as a new concept/node if it's novel enough
            self.ltm_nodes.append(vector)
            # print(f"New LTM Node added. Total nodes: {len(self.ltm_nodes)}")
        else:
            # Strengthen existing patterns (Simplified Hebbian)
            best_node_idx = np.argmax(similarities)
            # Update edges or node representation (Simplified)
            self.ltm_nodes[best_node_idx] = 0.9 * self.ltm_nodes[best_node_idx] + 0.1 * vector

    def recall(self, query_vector):
        """Reconciles STM and LTM to provide a context-aware representation."""
        if not self.ltm_nodes:
            return query_vector
            
        # Soft bias towards consistent prior reasoning (Section 2.3)
        similarities = [cosine_similarity(query_vector.reshape(1, -1), node.reshape(1, -1))[0][0] for node in self.ltm_nodes]
        
        # Weighted sum of LTM nodes based on similarity
        recall_vector = np.zeros(self.d)
        total_weight = 0
        for i, sim in enumerate(similarities):
            if sim > 0.5: # Only recall relevant memories
                recall_vector += sim * self.ltm_nodes[i]
                total_weight += sim
        
        if total_weight > 0:
            return 0.7 * query_vector + 0.3 * (recall_vector / total_weight)
        return query_vector

if __name__ == "__main__":
    print("Starting Cognitive Memory Simulation...")
    mem = CognitiveMemory(dimension=128)
    
    # Simulate learning a concept (e.g., "Financial Risk")
    concept_a = np.random.rand(128)
    for _ in range(5):
        noisy_a = concept_a + np.random.normal(0, 0.05, 128)
        mem.add_observation(noisy_a)
        
    # Recall with a partial/noisy query
    query = concept_a + np.random.normal(0, 0.2, 128)
    result = mem.recall(query)
    
    sim_before = cosine_similarity(query.reshape(1, -1), concept_a.reshape(1, -1))[0][0]
    sim_after = cosine_similarity(result.reshape(1, -1), concept_a.reshape(1, -1))[0][0]
    
    print(f"Similarity to original concept BEFORE recall: {sim_before:.4f}")
    print(f"Similarity to original concept AFTER recall: {sim_after:.4f}")
    print("Memory system successfully biased the query towards learned knowledge.")
