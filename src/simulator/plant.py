"""
The true plant: integrates the nonlinear model of the segway.

Role in the system
------------------
Stands in for the real robot. Uses the same nonlinear model as the
controllers and the observer, but is built independently of them, so
changing a controller's tuning can never change or break the physics.
"""

from pathlib import Path

from acados_template import AcadosModel
from acados_template.acados_sim import AcadosSim
from acados_template.acados_sim_solver import AcadosSimSolver

from ..models import MODEL

BUILD_DIR = Path(__file__).parent / "acados_build"


def segway_plant_model() -> AcadosModel:
    model = AcadosModel()
    model.f_expl_expr = MODEL.plant_explicit_acados
    model.f_impl_expr = MODEL.plant_implicit_acados
    model.x = MODEL.x
    model.u = MODEL.u
    model.xdot = MODEL.dx
    model.name = "SegwayPlant"  # unique name, so the generated C code doesn't clash with the OCPs
    return model


def segway_integrator(physics_dt: float) -> AcadosSimSolver:
    """One call to simulate() advances the true state by physics_dt."""
    sim = AcadosSim()
    sim.model = segway_plant_model()
    sim.solver_options.T = physics_dt
    sim.solver_options.integrator_type = "IRK"
    sim.solver_options.num_steps = 2
    sim.code_export_directory = str(BUILD_DIR / "c_generated_code_plant")
    return AcadosSimSolver(sim, json_file=str(BUILD_DIR / "acados_sim_plant.json"))
