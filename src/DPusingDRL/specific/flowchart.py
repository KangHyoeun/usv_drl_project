import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def add_box(ax, text, xy, boxstyle="round,pad=0.3", box_kwargs=None, text_kwargs=None):
    if box_kwargs is None:
        box_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}
    box = FancyBboxPatch(xy, 3, 1, boxstyle=boxstyle, **box_kwargs)
    ax.add_patch(box)
    ax.text(xy[0] + 1.5, xy[1] + 0.5, text, ha="center", va="center", fontsize=10, **text_kwargs)

fig, ax = plt.subplots(figsize=(10, 15))
ax.set_xlim(0, 6)
ax.set_ylim(0, 15)
ax.axis('off')

steps = [
    "1. Research Problem Definition",
    "2. Literature Review",
    "3. System Modeling",
    "4. Environment Setup",
    "5. Reinforcement Learning Algorithm Selection",
    "6. Design Reward System",
    "7. Train Model",
    "8. Evaluate Model Performance",
    "9. Fine-Tuning & Hyperparameter Optimization",
    "10. Validation & Testing",
    "11. Results Analysis",
    "12. Documentation & Reporting"
]

feedback_steps = [
    "8. Evaluate Model Performance",
    "9. Fine-Tuning & Hyperparameter Optimization",
    "10. Validation & Testing",
    "11. Results Analysis"
]

# Add main steps
for i, step in enumerate(steps):
    add_box(ax, step, (1, 14-i), box_kwargs={"edgecolor": "black", "facecolor": "lightgrey"})

# Add arrows for main flow
for i in range(1, len(steps)):
    ax.annotate('', xy=(2.5, 13.5-i), xytext=(2.5, 14.5-i),
                arrowprops=dict(arrowstyle="->", lw=1.5))

# Add feedback loop
feedback_y = [13-steps.index(step) for step in feedback_steps]
for i in range(len(feedback_y) - 1):
    ax.annotate('', xy=(2.5, feedback_y[i] - 0.5), xytext=(2.5, feedback_y[i+1] + 0.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='blue'))

# Add loop back to "Train Model"
ax.annotate('', xy=(2.5, feedback_y[-1] - 0.5), xytext=(2.5, 7.5),
            arrowprops=dict(arrowstyle="->", lw=1.5, color='blue'))

# Add continuous improvement box
add_box(ax, "Continuous Improvement", (4, 7), box_kwargs={"edgecolor": "black", "facecolor": "lightblue"})
ax.annotate('', xy=(3.5, 7.5), xytext=(5.5, 7.5),
            arrowprops=dict(arrowstyle="->", lw=1.5, color='blue'))
ax.annotate('', xy=(5.5, 7.5), xytext=(3.5, 8.5),
            arrowprops=dict(arrowstyle="->", lw=1.5, color='blue'))

plt.tight_layout()
plt.show()
