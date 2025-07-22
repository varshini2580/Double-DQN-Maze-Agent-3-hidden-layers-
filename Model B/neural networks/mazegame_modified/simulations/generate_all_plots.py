import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Set up paths
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

def create_plot(x_data, y_data, title, xlabel, ylabel, filename, plot_type='line', yticks=None):
    """Helper function to create and save a plot"""
    plt.figure(figsize=(12, 6))
    
    if plot_type == 'line':
        plt.plot(x_data, y_data, 'b-', linewidth=1.5)
    elif plot_type == 'bar':
        plt.bar(x_data, y_data, color='b', alpha=0.7)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if yticks:
        plt.yticks(yticks)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()

def generate_plots():
    """Generate all specified plots from training_data.xlsx"""
    try:
        # Load data
        df = pd.read_excel('training_data.xlsx')
        
        # 1. Temporal Difference per Episode
        create_plot(
            x_data=df['Episode'],
            y_data=df['TD_Error'],
            title='Temporal Difference per Episode',
            xlabel='Episodes',
            ylabel='Temporal Difference',
            filename='temporal_difference.png'
        )
        
        # 2. Episode Length per Episode
        create_plot(
            x_data=df['Episode'],
            y_data=df['Episode_Length'],
            title='Episode Length per Episode',
            xlabel='Episodes',
            ylabel='Episode Length',
            filename='episode_length.png',
            plot_type='bar'
        )
        
        # 3. Returns per Episodes
        create_plot(
            x_data=df['Episode'],
            y_data=df['Episode_Reward'],
            title='Returns per Episodes',
            xlabel='Episodes',
            ylabel='Returns',
            filename='returns.png'
        )
        
        # 4. Epsilon Decay per Episode
        create_plot(
            x_data=df['Episode'],
            y_data=df['Epsilon'],
            title='Epsilon Decay per Episode',
            xlabel='Episodes',
            ylabel='Epsilon Decay',
            filename='epsilon_decay.png'
        )
        
        # 5. Cumulative Reward per Episodes
        create_plot(
            x_data=df['Episode'],
            y_data=df['Cumulative_Reward'],
            title='Cumulative Reward per Episodes',
            xlabel='Episodes',
            ylabel='Cumulative Returns',
            filename='cumulative_reward.png'
        )
        
        print("All plots have been generated successfully!")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_plots()
    input("\nPress Enter to exit...")
