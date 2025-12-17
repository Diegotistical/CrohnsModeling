"""
Interactive Research Dashboard (Panel + HoloViews).

Responsibility:
    - Provide a reactive GUI for parameter exploration.
    - Stream simulation results without re-initializing the kernel.
    - Separate UI logic from physics logic.
"""
try:
    import param
    import panel as pn
    import holoviews as hv
    import numpy as np
    from holoviews import opts
except ImportError:
    raise ImportError("Interactive viz requires: pip install panel holoviews param")

# Initialize extension for Jupyter environment
hv.extension('bokeh')

class InteractiveSimulation(param.Parameterized):
    """
    A reactive controller for the Reaction-Diffusion system.
    
    Architecture:
    [ Params ] --triggers--> [ _update_cache ] --updates--> [ internal_data ]
                                                                    |
    [ View Methods ] <---reads--- [ time_step param ] <-------------+
    """
    
    # --- 1. Control Parameters ---
    Du = param.Number(0.16, bounds=(0.01, 0.5), doc="Diffusion Coeff U (Substrate)")
    Dv = param.Number(0.08, bounds=(0.01, 0.5), doc="Diffusion Coeff V (Inflammation)")
    F  = param.Number(0.0545, bounds=(0.01, 0.1), doc="Feed Rate")
    k  = param.Number(0.062, bounds=(0.03, 0.07), doc="Kill Rate")
    
    # Simulation Settings
    steps = param.Integer(1000, bounds=(100, 5000), doc="Simulation Steps")
    
    # Playback Control
    time_step = param.Integer(0, bounds=(0, 0), doc="Current Frame")
    
    def __init__(self, runner_callback, **params):
        """
        Args:
            runner_callback: A function fn(Du, Dv, F, k, steps) -> List[Tuple[U, V]]
                             This decouples the UI from the solver implementation.
        """
        super().__init__(**params)
        self._runner = runner_callback
        self._cache = []
        
        # Initial run
        self._run_simulation()

    @param.depends('Du', 'Dv', 'F', 'k', 'steps', watch=True)
    def _run_simulation(self):
        """Triggered automatically when physics parameters change."""
        if not self._runner:
            return

        # Status indicator (could be added to UI)
        print(f"Simulating with F={self.F}, k={self.k}...")
        
        # Execute Solver
        self._cache = self._runner(
            Du=self.Du, Dv=self.Dv, F=self.F, k=self.k, steps=self.steps
        )
        
        # Update playback slider bounds
        self.param.time_step.bounds = (0, max(0, len(self._cache) - 1))
        self.time_step = 0 # Reset to start

    # --- 2. View Components ---

    @param.depends('time_step')
    def view_u(self):
        """Reactive view for Substrate U."""
        if not self._cache:
            return hv.Image(np.zeros((10, 10)), bounds=(0,0,1,1))
            
        U, _ = self._cache[self.time_step]
        
        return hv.Image(
            U.T, bounds=(0, 0, 1, 1), label="Substrate U"
        ).opts(
            cmap='Viridis', colorbar=True, xaxis=None, yaxis=None, 
            title=f"Substrate (t={self.time_step})"
        )

    @param.depends('time_step')
    def view_v(self):
        """Reactive view for Inflammation V."""
        if not self._cache:
            return hv.Image(np.zeros((10, 10)), bounds=(0,0,1,1))
            
        _, V = self._cache[self.time_step]
        
        return hv.Image(
            V.T, bounds=(0, 0, 1, 1), label="Inflammation V"
        ).opts(
            cmap='Inferno', colorbar=True, xaxis=None, yaxis=None,
            title=f"Inflammation (t={self.time_step})"
        )

    # --- 3. Main Layout ---
    
    def view(self):
        """Returns the full dashboard object."""
        controls = pn.Column(
            "## Parameters",
            self.param.Du,
            self.param.Dv,
            self.param.F,
            self.param.k,
            "## Simulation",
            self.param.steps,
            "## Playback",
            self.param.time_step,
            background='#f0f0f0', width=300
        )
        
        plots = pn.Row(
            self.view_u,
            self.view_v
        )
        
        return pn.Row(controls, plots)