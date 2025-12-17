from src.viz.static import plot_snapshot
from src.viz.animation import save_animation
# Import interactive conditionally to avoid crashing if dependencies are missing
try:
    from src.viz.interactive import InteractiveSimulation
except ImportError:
    pass