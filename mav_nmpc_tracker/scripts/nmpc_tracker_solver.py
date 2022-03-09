import numpy as np
from dataclasses import dataclass
import casadi as cd
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from pathlib import Path
FILE_THIS = Path(__file__).resolve()
PARENT = FILE_THIS.parent
GPARENT = FILE_THIS.parents[1]

# NMPC trajectory tracking formulation using acados

# Constants
g = 9.8066


@dataclass
class MPC_Formulation_Param:
    # The following are by default, can be changed later
    mass = 1.56
    thrust_scale = 20.0
    # horizon
    dt = 0.05
    N = 20
    Tf = N * dt
    # dynamics
    roll_time_constant = 0.3
    roll_gain = 1.0
    pitch_time_constant = 0.3
    pitch_gain = 1.0
    drag_coefficient_x = 0.01
    drag_coefficient_y = 0.01
    # control bound
    roll_max = np.deg2rad(25)
    pitch_max = np.deg2rad(25)
    thrust_min = 0.5 * g            # mass divided
    thrust_max = 1.5 * g
    # cost weights
    q_x = 80
    q_y = 80
    q_z = 120
    q_vx = 80
    q_vy = 80
    q_vz = 100
    r_roll = 50
    r_pitch = 50
    r_thrust = 1


def acados_mpc_solver_generation(mpc_form_param):
    # Acados model
    model = AcadosModel()
    model.name = "mav_nmpc_tracker_model"

    # state
    px = cd.MX.sym('px')
    py = cd.MX.sym('py')
    pz = cd.MX.sym('pz')
    vx = cd.MX.sym('vx')
    vy = cd.MX.sym('vy')
    vz = cd.MX.sym('vz')
    roll = cd.MX.sym('roll')
    pitch = cd.MX.sym('pitch')
    yaw = cd.MX.sym('yaw')
    x = cd.vertcat(px, py, pz, vx, vy, vz, roll, pitch, yaw)

    # control
    roll_cmd = cd.MX.sym('roll_cmd')
    pitch_cmd = cd.MX.sym('pitch_cmd')
    thrust_cmd = cd.MX.sym('thrust_cmd')        # mass divided
    u = cd.vertcat(roll_cmd, pitch_cmd, thrust_cmd)

    # state derivative
    px_dot = cd.MX.sym('px_dot')
    py_dot = cd.MX.sym('py_dot')
    pz_dot = cd.MX.sym('pz_dot')
    vx_dot = cd.MX.sym('vx_dot')
    vy_dot = cd.MX.sym('vy_dot')
    vz_dot = cd.MX.sym('vz_dot')
    roll_dot = cd.MX.sym('roll_dot')
    pitch_dot = cd.MX.sym('pitch_dot')
    yaw_dot = cd.MX.sym('yaw_dot')
    x_dot = cd.vertcat(px_dot, py_dot, pz_dot, vx_dot, vy_dot, vz_dot, roll_dot, pitch_dot, yaw_dot)

    # drag
    drag_acc_x = np.cos(pitch)*np.cos(yaw)*mpc_form_param.drag_coefficient_x*thrust_cmd*vx \
                 - np.cos(pitch)*np.sin(yaw)*mpc_form_param.drag_coefficient_x*thrust_cmd*vy \
                 + np.sin(pitch)*mpc_form_param.drag_coefficient_x*thrust_cmd*vz
    drag_acc_y = (np.cos(roll)*np.sin(yaw) - np.cos(yaw)*np.sin(pitch)*np.sin(roll))*mpc_form_param.drag_coefficient_y*thrust_cmd*vx \
                 - (np.cos(roll)*np.cos(yaw) + np.sin(pitch)*np.sin(roll)*np.sin(yaw))*mpc_form_param.drag_coefficient_y*thrust_cmd*vy \
                 - np.cos(pitch)*np.sin(roll)*mpc_form_param.drag_coefficient_y*thrust_cmd*vz

    # dynamics
    dyn_f_expl = cd.vertcat(
        vx,
        vy,
        vz,
        (cd.cos(roll) * cd.cos(yaw) * cd.sin(pitch) + cd.sin(roll) * cd.sin(yaw)) * thrust_cmd  - drag_acc_x,
        (cd.cos(roll) * cd.sin(pitch) * cd.sin(yaw) - cd.cos(yaw) * cd.sin(roll)) * thrust_cmd  - drag_acc_y,
        -g + cd.cos(pitch) * cd.cos(roll) * thrust_cmd,
        (mpc_form_param.roll_gain * roll_cmd - roll) / mpc_form_param.roll_time_constant,
        (mpc_form_param.pitch_gain * pitch_cmd - pitch) / mpc_form_param.pitch_time_constant,
        0
    )
    dyn_f_impl = x_dot - dyn_f_expl

    # acados mpc model
    model.x = x
    model.u = u
    model.xdot = x_dot
    model.f_expl_expr = dyn_f_expl
    model.f_impl_expr = dyn_f_impl

    # Acados ocp
    ocp = AcadosOcp()
    ocp.model = model

    # ocp dimension
    ocp.dims.N = mpc_form_param.N
    nx = 9
    nu = 3
    ny = 9  # tracking pos, vel, and making u smaller
    ny_e = 6  # tracking terminal pos, vel

    # initial condition, can be changed in real time
    ocp.constraints.x0 = np.zeros(nx)

    # cost terms
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    Vx = np.zeros((ny, nx))
    Vx[:6, :6] = np.eye(6)
    ocp.cost.Vx = Vx
    Vu = np.zeros((ny, nu))
    Vu[6, 0] = 1.0
    Vu[7, 1] = 1.0
    Vu[8, 2] = 1.0
    ocp.cost.Vu = Vu
    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:6, :6] = np.eye(6)
    ocp.cost.Vx_e = Vx_e
    # weights, changed in real time
    ocp.cost.W = np.diag([mpc_form_param.q_x, mpc_form_param.q_y, mpc_form_param.q_z,
                          mpc_form_param.q_vx, mpc_form_param.q_vy, mpc_form_param.q_vz,
                          mpc_form_param.r_roll, mpc_form_param.r_pitch, mpc_form_param.r_thrust])
    # ocp.cost.W = np.diag([0.0, 0.0, 0.0,
    #                     0.0, 0.0, 0.0,
    #                     mpc_form_param.r_roll, mpc_form_param.r_pitch, mpc_form_param.r_thrust])
    ocp.cost.W_e = np.diag([mpc_form_param.q_x, mpc_form_param.q_y, mpc_form_param.q_z,
                            mpc_form_param.q_vx, mpc_form_param.q_vy, mpc_form_param.q_vz])

    # reference for tracking, changed in real time
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # set control bound
    ocp.constraints.lbu = np.array([-mpc_form_param.roll_max, -mpc_form_param.pitch_max, mpc_form_param.thrust_min])
    ocp.constraints.ubu = np.array([mpc_form_param.roll_max, mpc_form_param.pitch_max, mpc_form_param.thrust_max])
    ocp.constraints.idxbu = np.array(range(nu))

    # solver options
    # horizon
    ocp.solver_options.tf = mpc_form_param.Tf
    # qp solver
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'    # PARTIAL_CONDENSING_HPIPM
    ocp.solver_options.qp_solver_cond_N = 5
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.qp_solver_warm_start = 1
    # nlp solver
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.nlp_solver_tol_eq = 1E-3
    ocp.solver_options.nlp_solver_tol_ineq = 1E-3
    ocp.solver_options.nlp_solver_tol_comp = 1E-3
    ocp.solver_options.nlp_solver_tol_stat = 1E-3
    # hessian
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    # integrator
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    # print
    ocp.solver_options.print_level = 0
    # solver generation
    ocp.code_export_directory = str(GPARENT) + '/solver/'

    # Acados solver
    print("Starting solver generation...")
    solver = AcadosOcpSolver(ocp, json_file='ACADOS_nmpc_tracker_solver.json')
    print("Solver generated.")

    return solver


if __name__ == "__main__":
    param = MPC_Formulation_Param()
    acados_mpc_solver_generation(param)
