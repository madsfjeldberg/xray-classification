import numpy as np
import matplotlib.pyplot as plt


def plot_loss_flattening(losses, window_size=3, title="Loss Flattening Detection", save_path=None):
    """
    Plots loss over epochs and highlights flattening regions.
    
    Flattening is detected by computing the gradient (rate of change) 
    and identifying when it becomes small (close to flat).
    
    Parameters:
    - losses: list of loss values per epoch
    - window_size: size of moving average window to smooth the curve
    - title: plot title
    - save_path: optional path to save the figure
    """
    losses = np.asarray(losses, dtype=float)
    if len(losses) == 0:
        raise ValueError("losses must not be empty.")
    
    epochs = np.arange(1, len(losses) + 1)
    
    # Compute rate of change (gradient)
    gradient = np.abs(np.gradient(losses))
    
    # Smooth gradient with moving average
    smoothed_gradient = np.convolve(gradient, np.ones(window_size)/window_size, mode='same')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Loss over epochs
    ax1.plot(epochs, losses, marker='o', linewidth=2, label='Loss', color='steelblue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Rate of change (gradient) - shows flattening
    ax2.plot(epochs, gradient, marker='s', linewidth=1, alpha=0.5, label='Gradient (|Δloss|)', color='orange')
    ax2.plot(epochs, smoothed_gradient, linewidth=2, label=f'Smoothed gradient (window={window_size})', color='red')
    ax2.axhline(y=np.mean(smoothed_gradient) * 0.1, color='green', linestyle='--', label='Flattening threshold (10% of avg)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Absolute Change in Loss')
    ax2.set_title('Rate of Change - Lower = Flatter (Converged)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    
    # Print flattening analysis
    threshold = np.mean(smoothed_gradient) * 0.1
    flat_epochs = np.where(smoothed_gradient < threshold)[0]
    if len(flat_epochs) > 0:
        first_flat = flat_epochs[0] + 1
        print(f"\n✓ Loss appears to flatten starting around epoch {first_flat}")
    else:
        print("\n✗ Loss did not flatten within the training period (still improving)")