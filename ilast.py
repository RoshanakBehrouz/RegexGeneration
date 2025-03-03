import streamlit as st
import re
import random
import numpy as np
from deap import base, creator, tools, gp
from multiprocessing import Pool
from tqdm import tqdm

def eval_regex(individual, training_data):
    """Evaluate fitness of a regex individual."""
    regex_pattern = "|".join(map(str, individual))  # Ensure all elements are strings
    correct, total, expected, unique_patterns = 0, 0, 0, set()
    
    for text, pos_labels, neg_labels in training_data:
        matches = set(re.findall(regex_pattern, text))
        expected += len(pos_labels)
        total += len(matches)
        correct += len(set(pos_labels) & matches)
        unique_patterns.update(matches)  # Track diversity of matched patterns
    
    precision = correct / total if total else 0
    recall = correct / expected if expected else 0
    complexity = len(regex_pattern)
    diversity = len(unique_patterns)  # Count unique matched patterns
    
    return precision, recall, complexity, diversity

# Streamlit UI
st.title("Regex Generator using Genetic Programming")

# User Input
st.write("Enter positive and negative examples for regex inference:")

positive_examples = st.text_area("Positive Examples (one per line)").split("\n")
negative_examples = st.text_area("Negative Examples (one per line)").split("\n")

data = [(txt, [txt], []) for txt in positive_examples if txt] + [(txt, [], [txt]) for txt in negative_examples if txt]

if st.button("Generate Regex"):
    if not data:
        st.error("Please provide examples before running.")
    else:
        # Define regex components
        regex_parts = [
            r"\b\w+@\w+\.\w+\b",
            r"\b\d{3}-\d{3}-\d{4}\b",
            r"\bhttps?://\S+\b",
            r"\b[A-Za-z]+\b",
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
        ]
        
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -0.5, 0.5))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_item", lambda: random.choice(regex_parts))
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, n=2)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", lambda ind: ind.append(random.choice(regex_parts)) if random.random() < 0.5 else ind.pop() if len(ind) > 1 else ind)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("evaluate", lambda ind: eval_regex(ind, data))
        
        pop_size = 52
        pop = toolbox.population(n=pop_size)
        ngen, cxpb, mutpb = 30, 0.5, 0.2
        
        for gen in tqdm(range(ngen), desc="Evolving regex patterns"):
            offspring = tools.selNSGA2(pop, pop_size)
            offspring = [toolbox.clone(ind) for ind in offspring]
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb and len(child1) > 1 and len(child2) > 1:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values
            
            for mutant in offspring:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            invalids = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalids:
                ind.fitness.values = toolbox.evaluate(ind)
            
            pop[:] = offspring
        
        best = tools.selBest(pop, 1)[0]
        best_regex = "|".join(map(str, best))
        
        st.success(f"Best Regex: {best_regex}")
