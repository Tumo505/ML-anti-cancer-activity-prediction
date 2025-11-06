"""
Genetic Algorithm for drug combination generation.
Uses CNN predictions as fitness function to optimize efficacy and minimize toxicity.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Callable, Dict, Optional, TYPE_CHECKING
from deap import base, creator, tools, algorithms
import random
import pandas as pd
from dataclasses import dataclass


@dataclass
class DrugCombination:
    """Represents a drug combination."""
    drug_indices: List[int]
    efficacy_score: float
    toxicity_score: float
    fitness_score: float
    

class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm for optimizing drug combinations.
    Maximizes efficacy while minimizing toxicity.
    """
    
    def __init__(
        self,
        n_drugs: int,
        min_combination_size: int = 2,
        max_combination_size: int = 5,
        efficacy_weight: float = 0.7,
        toxicity_weight: float = 0.3,
        population_size: int = 100,
        n_generations: int = 50,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        tournament_size: int = 3
    ):
        """
        Initialize Genetic Algorithm.
        
        Args:
            n_drugs: Total number of available drugs
            min_combination_size: Minimum drugs in combination
            max_combination_size: Maximum drugs in combination
            efficacy_weight: Weight for efficacy in fitness (0-1)
            toxicity_weight: Weight for toxicity penalty (0-1)
            population_size: Number of individuals in population
            n_generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            tournament_size: Size of tournament selection
        """
        self.n_drugs = n_drugs
        self.min_combination_size = min_combination_size
        self.max_combination_size = max_combination_size
        self.efficacy_weight = efficacy_weight
        self.toxicity_weight = toxicity_weight
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        
        self.predictor = None
        self.best_combinations = []
        
        # Initialize DEAP framework
        self._setup_deap()
    
    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework."""
        # Create fitness class (maximize efficacy, minimize toxicity)
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMulti", base.Fitness, weights=(1.0,))  # Maximize combined fitness
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
    
    def _create_individual(self):
        """
        Create a random individual (drug combination).
        
        Returns:
            Binary array representing drug selection
        """
        individual = [0] * self.n_drugs
        n_selected = random.randint(self.min_combination_size, self.max_combination_size)
        selected_indices = random.sample(range(self.n_drugs), n_selected)
        
        for idx in selected_indices:
            individual[idx] = 1
        
        return creator.Individual(individual)
    
    def _crossover(self, ind1, ind2):
        """
        Crossover operation: exchange drug selections between two individuals.
        
        Args:
            ind1: First parent
            ind2: Second parent
        
        Returns:
            Two offspring
        """
        # Two-point crossover
        size = len(ind1)
        cx_point1 = random.randint(1, size - 1)
        cx_point2 = random.randint(cx_point1 + 1, size)
        
        ind1[cx_point1:cx_point2], ind2[cx_point1:cx_point2] = \
            ind2[cx_point1:cx_point2], ind1[cx_point1:cx_point2]
        
        # Ensure combination size constraints
        ind1 = self._enforce_size_constraint(ind1)
        ind2 = self._enforce_size_constraint(ind2)
        
        return ind1, ind2
    
    def _mutate(self, individual):
        """
        Mutation operation: randomly add/remove drugs.
        
        Args:
            individual: Individual to mutate
        
        Returns:
            Mutated individual
        """
        # Randomly flip a drug selection
        if random.random() < 0.5:
            # Add a drug
            zero_indices = [i for i, x in enumerate(individual) if x == 0]
            if zero_indices and sum(individual) < self.max_combination_size:
                idx = random.choice(zero_indices)
                individual[idx] = 1
        else:
            # Remove a drug
            one_indices = [i for i, x in enumerate(individual) if x == 1]
            if len(one_indices) > self.min_combination_size:
                idx = random.choice(one_indices)
                individual[idx] = 0
        
        return (individual,)
    
    def _enforce_size_constraint(self, individual):
        """
        Ensure individual meets size constraints.
        
        Args:
            individual: Individual to check
        
        Returns:
            Valid individual
        """
        n_selected = sum(individual)
        
        if n_selected < self.min_combination_size:
            # Add drugs
            zero_indices = [i for i, x in enumerate(individual) if x == 0]
            n_to_add = self.min_combination_size - n_selected
            if zero_indices:
                add_indices = random.sample(zero_indices, min(n_to_add, len(zero_indices)))
                for idx in add_indices:
                    individual[idx] = 1
        
        elif n_selected > self.max_combination_size:
            # Remove drugs
            one_indices = [i for i, x in enumerate(individual) if x == 1]
            n_to_remove = n_selected - self.max_combination_size
            remove_indices = random.sample(one_indices, n_to_remove)
            for idx in remove_indices:
                individual[idx] = 0
        
        return individual
    
    def set_predictor(self, predictor: Callable):
        """
        Set the prediction model for fitness evaluation.
        
        Args:
            predictor: Function that takes drug combination and returns 
                      (efficacy_score, toxicity_score)
        """
        self.predictor = predictor
    
    def _evaluate_fitness(self, individual) -> Tuple[float]:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual: Drug combination to evaluate
        
        Returns:
            Fitness tuple
        """
        if self.predictor is None:
            raise ValueError("Predictor not set! Use set_predictor() first.")
        
        # Get drug indices
        drug_indices = [i for i, x in enumerate(individual) if x == 1]
        
        if len(drug_indices) == 0:
            return (-1000.0,)  # Invalid combination
        
        # Get prediction from model
        efficacy, toxicity = self.predictor(drug_indices)
        
        # Calculate combined fitness
        # Maximize efficacy, minimize toxicity
        fitness = (self.efficacy_weight * efficacy) - (self.toxicity_weight * toxicity)
        
        return (fitness,)
    
    def optimize(self, verbose: bool = True) -> List[DrugCombination]:
        """
        Run genetic algorithm optimization.
        
        Args:
            verbose: Whether to print progress
        
        Returns:
            List of best drug combinations
        """
        print("\n" + "="*80)
        print("GENETIC ALGORITHM OPTIMIZATION")
        print("="*80)
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.n_generations}")
        print(f"Combination size: {self.min_combination_size}-{self.max_combination_size} drugs")
        print(f"Efficacy weight: {self.efficacy_weight}")
        print(f"Toxicity weight: {self.toxicity_weight}")
        print("="*80)
        
        # Register fitness function
        self.toolbox.register("evaluate", self._evaluate_fitness)
        
        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        
        # Statistics
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)
        
        # Hall of fame to store best individuals
        hof = tools.HallOfFame(20)
        
        # Run evolution
        population, logbook = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.n_generations,
            stats=stats,
            halloffame=hof,
            verbose=verbose
        )
        
        # Extract best combinations
        self.best_combinations = []
        for individual in hof:
            drug_indices = [i for i, x in enumerate(individual) if x == 1]
            efficacy, toxicity = self.predictor(drug_indices)
            fitness = individual.fitness.values[0]
            
            combo = DrugCombination(
                drug_indices=drug_indices,
                efficacy_score=efficacy,
                toxicity_score=toxicity,
                fitness_score=fitness
            )
            self.best_combinations.append(combo)
        
        print("\nOPTIMIZATION COMPLETE")
        print(f"Found {len(self.best_combinations)} optimal combinations")
        print("="*80)
        
        return self.best_combinations
    
    def get_top_combinations(self, n: int = 10) -> List[DrugCombination]:
        """
        Get top N drug combinations.
        
        Args:
            n: Number of top combinations to return
        
        Returns:
            List of top combinations
        """
        return self.best_combinations[:n]
    
    def export_results(self, filepath: str, drug_names: Optional[List[str]] = None):
        """
        Export optimization results to CSV.
        
        Args:
            filepath: Path to save results
            drug_names: Optional list of drug names
        """
        results = []
        
        for i, combo in enumerate(self.best_combinations):
            if drug_names:
                drugs = [drug_names[idx] for idx in combo.drug_indices]
                drug_str = ", ".join(drugs)
            else:
                drug_str = ", ".join([f"Drug_{idx}" for idx in combo.drug_indices])
            
            results.append({
                'rank': i + 1,
                'combination': drug_str,
                'n_drugs': len(combo.drug_indices),
                'efficacy_score': combo.efficacy_score,
                'toxicity_score': combo.toxicity_score,
                'fitness_score': combo.fitness_score
            })
        
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
        
        print(f"Results exported to {filepath}")


if __name__ == "__main__":
    print("="*80)
    print("GENETIC ALGORITHM - TEST")
    print("="*80)
    
    # Mock predictor function
    def mock_predictor(drug_indices: List[int]) -> Tuple[float, float]:
        """Mock predictor for testing."""
        # Simulate efficacy: more drugs = higher efficacy (with diminishing returns)
        efficacy = 0.5 + 0.1 * len(drug_indices) - 0.01 * len(drug_indices)**2
        
        # Simulate toxicity: more drugs = higher toxicity
        toxicity = 0.2 + 0.05 * len(drug_indices)
        
        # Add some randomness
        efficacy += np.random.normal(0, 0.05)
        toxicity += np.random.normal(0, 0.03)
        
        return max(0, min(1, efficacy)), max(0, min(1, toxicity))
    
    # Initialize GA
    ga = GeneticAlgorithmOptimizer(
        n_drugs=100,
        min_combination_size=2,
        max_combination_size=4,
        population_size=50,
        n_generations=20
    )
    
    # Set predictor
    ga.set_predictor(mock_predictor)
    
    # Run optimization
    best_combos = ga.optimize(verbose=False)
    
    # Display top 5 combinations
    print("\nTOP 5 DRUG COMBINATIONS:")
    print("-" * 80)
    for i, combo in enumerate(best_combos[:5], 1):
        print(f"{i}. Drugs: {combo.drug_indices}")
        print(f"   Efficacy: {combo.efficacy_score:.3f} | Toxicity: {combo.toxicity_score:.3f} | Fitness: {combo.fitness_score:.3f}")
        print()
    
    print("Genetic Algorithm test complete!")
