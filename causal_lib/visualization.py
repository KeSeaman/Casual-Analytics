import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

def plot_ate_comparison(results: Dict[str, float], title: str = "ATE Comparison"):
    """
    Plots a bar chart comparing ATE estimates from different methods.
    Returns a Marimo UI element.
    """
    methods = list(results.keys())
    values = list(results.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, values, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f'])
    
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Average Treatment Effect (ATE)", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
                
    plt.tight_layout()
    return fig

def plot_propensity_score(ps_treated: np.ndarray, ps_control: np.ndarray):
    """
    Plots the distribution of propensity scores for treated vs control.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ps_control, bins=30, alpha=0.5, label='Control', density=True, color='blue')
    ax.hist(ps_treated, bins=30, alpha=0.5, label='Treated', density=True, color='red')
    
    ax.set_title("Propensity Score Distribution (Overlap)", fontsize=16)
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Density")
    ax.legend()
    
    plt.tight_layout()
    return fig
