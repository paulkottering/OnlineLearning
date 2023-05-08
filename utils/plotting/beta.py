import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
import os

# Parameter pairs for the Beta distributions
params = [(1, 1), (1, 3), (3, 1), (0.5, 0.5), (5, 5)]
labels = ['Uniform', 'Negatively Skewed', 'Positively Skewed', 'Tail Skewed', 'Symmetrical Unimodal']

# Create x-axis values
x = np.linspace(0, 1, 100)

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.set_style('whitegrid')

# Plot each Beta distribution with its label
lines = []
for p, label in zip(params, labels):
    alpha, beta_ = p
    y = beta.pdf(x, alpha, beta_)
    line, = ax.plot(x, y, label=label)
    lines.append(line)

# Customize plot
ax.set_title('Beta Distributions with Different Shape Parameters')
ax.set_xlabel('Potential Function Value')
ax.set_ylabel('Probability Density')

# Create custom legend with textbox above the plot
legend = ax.legend(lines, labels, title='Distributions', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=len(lines), frameon=True, fancybox=True)

# Adjust subplot
plt.subplots_adjust(bottom=0.2)

# Save the plot
if not os.path.exists("Figures"):
    os.makedirs("Figures")
title = f"Figures/5_5.png"
plt.savefig(title, bbox_inches="tight")
print("Plot saved to", title)
