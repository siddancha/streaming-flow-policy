import matplotlib.pyplot as plt
import numpy as np
import random
import torch

def set_random_seed(seed: int):
    # https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def SetLatexInMatplotlib():
    """Configure Matplotlib to use LaTeX for labels and titles only.
    
    Requires: 
    - A LaTeX distribution (e.g., TexLive, MikTeX)
    - STIXGeneral fonts.
    """
    try:
        plt.rcParams.update({
            # Only enable LaTeX for specific text elements
            'axes.titlesize': 12,
            'axes.labelsize': 12,
            # 'font.serif': ['Computer Modern Roman'],
            'mathtext.fontset': 'cm',
            'font.size': 12,
            # Set smaller tick label size
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            # LaTeX settings for labels and titles
            'axes.titley': True,
            'axes.labelpad': 5,
            'axes.formatter.use_mathtext': True,
            'axes.labelcolor': 'black',
        })
    except Exception as e:
        print(f"Failed to set LaTeX rendering: {e}")
        print("Please ensure LaTeX and required fonts are installed.")
