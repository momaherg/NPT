#!/usr/bin/env python3
"""
Demonstration of Modulation Arithmetic Concepts.

This script shows the mathematical intuition behind modulation arithmetic
in NPT models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Simulate modulation vectors
def create_random_modulation(name: str, seed: int = None, dim: int = 8):
    """Create a simulated modulation vector."""
    if seed is not None:
        torch.manual_seed(seed)
    
    v_a = torch.randn(1, 1, dim) * 0.1
    v_b = torch.randn(1, 1, dim * 2) * 0.1  # FFN dimension typically larger
    
    return {'name': name, 'v_a': v_a, 'v_b': v_b}

def modulation_magnitude(mod):
    """Calculate magnitude of a modulation."""
    return (mod['v_a'].norm().item() + mod['v_b'].norm().item()) / 2

def visualize_modulations(mods, title="Modulation Vectors"):
    """Visualize modulation vectors."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for ax, key in zip(axes, ['v_a', 'v_b']):
        ax.set_title(f"{key} vectors")
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Modulation")
        
        for mod in mods:
            values = mod[key].squeeze().numpy()
            ax.plot(values, label=mod['name'], alpha=0.7, linewidth=2)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def demonstrate_subtraction():
    """Show what subtraction reveals."""
    print("\n" + "="*60)
    print("MODULATION SUBTRACTION: Isolating Differences")
    print("="*60)
    
    # Create modulations for different contexts
    paris = create_random_modulation("paris", seed=42)
    berlin = create_random_modulation("berlin", seed=43)
    
    # Add some common component (European capital)
    common = create_random_modulation("european", seed=100)
    paris['v_a'] += common['v_a'] * 0.5
    paris['v_b'] += common['v_b'] * 0.5
    berlin['v_a'] += common['v_a'] * 0.5
    berlin['v_b'] += common['v_b'] * 0.5
    
    # Compute difference
    diff = {
        'name': 'paris - berlin',
        'v_a': paris['v_a'] - berlin['v_a'],
        'v_b': paris['v_b'] - berlin['v_b']
    }
    
    print(f"Paris magnitude:        {modulation_magnitude(paris):.4f}")
    print(f"Berlin magnitude:       {modulation_magnitude(berlin):.4f}")
    print(f"Difference magnitude:   {modulation_magnitude(diff):.4f}")
    print("\nThe difference isolates what makes Paris distinct from Berlin,")
    print("removing their common 'European capital' component.")
    
    visualize_modulations([paris, berlin, diff], 
                          "Subtraction: Isolating Differences")

def demonstrate_averaging():
    """Show how averaging creates prototypes."""
    print("\n" + "="*60)
    print("MODULATION AVERAGING: Creating Prototypes")
    print("="*60)
    
    # Create modulations for similar concepts
    cat = create_random_modulation("cat", seed=1)
    dog = create_random_modulation("dog", seed=2)
    rabbit = create_random_modulation("rabbit", seed=3)
    
    # Average them
    avg = {
        'name': 'avg(cat,dog,rabbit)',
        'v_a': (cat['v_a'] + dog['v_a'] + rabbit['v_a']) / 3,
        'v_b': (cat['v_b'] + dog['v_b'] + rabbit['v_b']) / 3
    }
    
    print(f"Cat magnitude:     {modulation_magnitude(cat):.4f}")
    print(f"Dog magnitude:     {modulation_magnitude(dog):.4f}")
    print(f"Rabbit magnitude:  {modulation_magnitude(rabbit):.4f}")
    print(f"Average magnitude: {modulation_magnitude(avg):.4f}")
    print("\nThe average creates a 'pet animal' prototype,")
    print("capturing common features while reducing specific noise.")
    
    visualize_modulations([cat, dog, rabbit, avg], 
                          "Averaging: Creating Prototypes")

def demonstrate_negation():
    """Show how negation creates opposites."""
    print("\n" + "="*60)
    print("MODULATION NEGATION: Creating Opposites")
    print("="*60)
    
    # Create a modulation
    positive = create_random_modulation("positive_sentiment", seed=10)
    
    # Negate it
    negative = {
        'name': '-1 * positive',
        'v_a': -positive['v_a'],
        'v_b': -positive['v_b']
    }
    
    print(f"Original magnitude: {modulation_magnitude(positive):.4f}")
    print(f"Negated magnitude:  {modulation_magnitude(negative):.4f}")
    print("\nNegation creates an 'anti-modulation' that would:")
    print("- Suppress tokens promoted by the original")
    print("- Promote tokens suppressed by the original")
    
    visualize_modulations([positive, negative], 
                          "Negation: Creating Opposites")

def demonstrate_composition():
    """Show complex modulation composition."""
    print("\n" + "="*60)
    print("COMPLEX COMPOSITION: Building Nuanced Modulations")
    print("="*60)
    
    # Create base modulations
    technical = create_random_modulation("technical", seed=20)
    friendly = create_random_modulation("friendly", seed=21)
    formal = create_random_modulation("formal", seed=22)
    
    # Create a composed modulation: technical + friendly - formal
    composed = {
        'name': 'tech + friend - formal',
        'v_a': technical['v_a'] + friendly['v_a'] - formal['v_a'],
        'v_b': technical['v_b'] + friendly['v_b'] - formal['v_b']
    }
    
    print("Creating: Technical + Friendly - Formal")
    print(f"Technical magnitude:  {modulation_magnitude(technical):.4f}")
    print(f"Friendly magnitude:   {modulation_magnitude(friendly):.4f}")
    print(f"Formal magnitude:     {modulation_magnitude(formal):.4f}")
    print(f"Composed magnitude:   {modulation_magnitude(composed):.4f}")
    print("\nThis creates a modulation for 'technical but approachable' style,")
    print("removing formality while keeping technical accuracy.")
    
    visualize_modulations([technical, friendly, formal, composed], 
                          "Composition: Building Complex Modulations")

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║         MODULATION ARITHMETIC IN NPT MODELS                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Modulations encode how attention and MLP transformations   ║
║  modify the model's behavior. By performing arithmetic on   ║
║  these modulations, we can:                                 ║
║                                                              ║
║  • SUBTRACT to isolate differences                          ║
║  • ADD to combine knowledge                                 ║
║  • AVERAGE to create prototypes                             ║
║  • NEGATE to create opposites                               ║
║  • SCALE to control strength                                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Note: Set to False if matplotlib is not available
    SHOW_PLOTS = False
    
    if not SHOW_PLOTS:
        # Monkey patch to disable plotting
        plt.show = lambda: None
    
    demonstrate_subtraction()
    demonstrate_averaging()
    demonstrate_negation()
    demonstrate_composition()
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("""
1. Subtraction reveals contrastive features:
   paris - berlin = "what makes Paris unique"

2. Addition combines complementary knowledge:
   technical + friendly = "technical but approachable"

3. Averaging creates stable prototypes:
   avg(cat, dog, rabbit) = "generic pet animal"

4. Negation creates suppressors:
   -positive_sentiment = "anti-positive sentiment"

5. Scaling controls effect strength:
   0.5 * modulation = "subtle effect"
   2.0 * modulation = "strong effect"
    """)
    
    print("\nThese operations enable fine-grained control over")
    print("knowledge transfer and behavior modification in NPT models.")
    print("\nUse the interactive tool to experiment with real modulations!")