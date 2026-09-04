import numpy as np
import matplotlib.pyplot as plt

def simulate_kelly_growth(win_prob, payoff_ratio, num_steps, num_paths, starting_wealth=100.0):
    """
    Simulates geometric wealth growth using different betting fractions.
    win_prob: Probability of winning the bet (p)
    payoff_ratio: Odds received on a win (b) 
    """
    # 1. Calculate the theoretical optimal Kelly fraction: f* = (bp - q) / b
    loss_prob = 1.0 - win_prob
    kelly_fraction = (payoff_ratio * win_prob - loss_prob) / payoff_ratio
    
    # Define our test strategies: Half Kelly, Full Kelly, and Over-betting (Risk of Ruin)
    fractions = {
        'Half Kelly (Safe)': kelly_fraction / 2,
        'Full Kelly (Optimal)': kelly_fraction,
        'Double Kelly (Over-betting)': kelly_fraction * 2
    }
    
    results = {}
    
    # 2. Run Monte Carlo simulation for each fraction
    for label, f in fractions.items():
        # Initialize wealth matrix: rows = paths, cols = time steps
        wealth = np.zeros((num_paths, num_steps + 1))
        wealth[:, 0] = starting_wealth
        
        # Generate random coin flips for all paths and steps at once (Vectorized)
        # 1 means win, 0 means loss
        flips = np.random.binomial(1, win_prob, size=(num_paths, num_steps))
        
        # Calculate multipliers: (1 + f*b) on win, (1 - f) on loss
        multipliers = np.where(flips == 1, 1 + f * payoff_ratio, 1 - f)
        
        # Cumulative product calculates geometric compounding over time
        wealth[:, 1:] = starting_wealth * np.cumprod(multipliers, axis=1)
        
        # Store the median path to plot typical performance
        results[label] = np.median(wealth, axis=0)
        
    return results, kelly_fraction

# --- Execution & Plotting ---
if __name__ == "__main__":
    np.random.seed(42)
    steps = 300
    paths = 5000
    
    # Example: A game with a 55% win rate and 1:1 payout
    sim_results, optimal_f = simulate_kelly_growth(win_prob=0.55, payoff_ratio=1.0, num_steps=steps, num_paths=paths)
    
    print(f"Optimal Kelly Fraction: {optimal_f:.2%}")
    
    plt.figure(figsize=(10, 6))
    for label, median_wealth in sim_results.items():
        plt.plot(median_wealth, label=label, linewidth=2)
        
    plt.yscale('log') # Log scale because wealth compounds geometrically
    plt.title('Monte Carlo: Wealth Trajectories via Kelly Criterion (5000 paths)')
    plt.xlabel('Number of Bets')
    plt.ylabel('Median Wealth (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('wealth_trajectories.png')
