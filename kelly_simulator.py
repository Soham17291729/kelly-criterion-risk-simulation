import numpy as np
import matplotlib.pyplot as plt

def simulate_kelly_growth(win_prob, payoff_ratio, num_steps, num_paths, starting_wealth=100.0):
    """
    Simulates geometric wealth growth across 10^6 paths using vectorized NumPy.
    Calculates median trajectories and 90% confidence intervals (5th-95th percentiles).
    """
    # 1. Theoretical optimal Kelly fraction: f* = (bp - q) / b
    loss_prob = 1.0 - win_prob
    kelly_fraction = (payoff_ratio * win_prob - loss_prob) / payoff_ratio
    
    fractions = {
        'Half Kelly (Safe, Low Variance)': kelly_fraction / 2,
        'Full Kelly (Optimal Growth)': kelly_fraction,
        'Double Kelly (Over-betting / Ruin)': kelly_fraction * 2
    }
    
    results = {}
    
    for label, f in fractions.items():
        # 2. Vectorized Monte Carlo: 10^6 paths x 300 steps simultaneously
        # Generate 1s (win) and 0s (loss)
        flips = np.random.binomial(1, win_prob, size=(num_paths, num_steps))
        
        # Apply multipliers: (1 + fb) on win, (1 - f) on loss
        multipliers = np.where(flips == 1, 1 + f * payoff_ratio, 1 - f)
        
        # Calculate cumulative geometric growth
        wealth_paths = starting_wealth * np.cumprod(multipliers, axis=1)
        
        # Prepend the starting wealth at step 0 for graphing
        initial_wealth_col = np.full((num_paths, 1), starting_wealth)
        wealth_paths = np.hstack([initial_wealth_col, wealth_paths])
        
        # 3. Extract statistical bounds (prevents plotting 1 million individual lines)
        p50 = np.median(wealth_paths, axis=0)
        p05 = np.percentile(wealth_paths, 5, axis=0)
        p95 = np.percentile(wealth_paths, 95, axis=0)
        
        results[label] = {'median': p50, 'lower_bound': p05, 'upper_bound': p95}
        
    return results, kelly_fraction

# --- Execution & Plotting ---
if __name__ == "__main__":
    np.random.seed(42) # For reproducibility 
    steps = 300
    paths = 1000000 # Scaled to 10^6 paths for rigorous convergence
    
    # Example: 55% win rate, 1:1 payout
    sim_results, optimal_f = simulate_kelly_growth(win_prob=0.55, payoff_ratio=1.0, num_steps=steps, num_paths=paths)
    
    print(f"Optimal Kelly Fraction: {optimal_f:.2%}")
    
    # 4. Generate Professional Rigorous Plot
    plt.figure(figsize=(10, 6))
    
    colors = {'Half Kelly (Safe, Low Variance)': 'blue', 
              'Full Kelly (Optimal Growth)': 'green', 
              'Double Kelly (Over-betting / Ruin)': 'red'}
    
    for label, data in sim_results.items():
        color = colors[label]
        # Plot the median trajectory
        plt.plot(data['median'], label=label, color=color, linewidth=2)
        
        # Fill the 90% confidence interval to show variance and drawdown risk
        # (We skip shading Double Kelly to keep the chart clean, as it converges to 0)
        if "Double" not in label:
            plt.fill_between(range(steps + 1), data['lower_bound'], data['upper_bound'], 
                             color=color, alpha=0.15)
        
    plt.yscale('log')
    plt.title(f'Kelly Criterion: Geometric Wealth Growth & Variance ($10^6$ Paths)')
    plt.xlabel('Number of Bets')
    plt.ylabel('Wealth (Log Scale) with 90% Confidence Intervals')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3, which="both", ls="--")
    plt.tight_layout()
    plt.savefig('wealth_trajectories.png', dpi=300) # High-res export
