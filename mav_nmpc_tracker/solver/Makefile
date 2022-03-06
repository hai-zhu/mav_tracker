#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#






ACADOS_FLAGS = -fPIC -std=c99   #-fno-diagnostics-show-line-numbers -g
ACADOS_FLAGS += -DACADOS_WITH_QPOASES
# # Debugging
# ACADOS_FLAGS += -g3

MODEL_OBJ=
MODEL_OBJ+= mav_nmpc_tracker_model_model/mav_nmpc_tracker_model_expl_ode_fun.o
MODEL_OBJ+= mav_nmpc_tracker_model_model/mav_nmpc_tracker_model_expl_vde_forw.o


OCP_OBJ=
OCP_OBJ+= acados_solver_mav_nmpc_tracker_model.o


SIM_OBJ=
SIM_OBJ+= acados_sim_solver_mav_nmpc_tracker_model.o

EX_OBJ=
EX_OBJ+= main_mav_nmpc_tracker_model.o

EX_SIM_OBJ=
EX_SIM_OBJ+= main_sim_mav_nmpc_tracker_model.o

OBJ=
OBJ+= $(MODEL_OBJ)
OBJ+= $(SIM_OBJ)
OBJ+= $(OCP_OBJ)

EXTERNAL_DIR=
EXTERNAL_LIB=

INCLUDE_PATH = /home/hai/ROS/mav_navi_ws/src/mav_tracker/external/acados/include
LIB_PATH = /home/hai/ROS/mav_navi_ws/src/mav_tracker/external/acados/lib
all: clean casadi_fun example_sim example
shared_lib: bundled_shared_lib ocp_shared_lib sim_shared_lib

CASADI_MODEL_SOURCE=
CASADI_MODEL_SOURCE+= mav_nmpc_tracker_model_expl_ode_fun.c
CASADI_MODEL_SOURCE+= mav_nmpc_tracker_model_expl_vde_forw.c

casadi_fun:
	( cd mav_nmpc_tracker_model_model ; gcc $(ACADOS_FLAGS) -c $(CASADI_MODEL_SOURCE))

main:
	gcc $(ACADOS_FLAGS) -c main_mav_nmpc_tracker_model.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/ \
	-I $(INCLUDE_PATH)/qpOASES_e/

main_sim:
	gcc $(ACADOS_FLAGS) -c main_sim_mav_nmpc_tracker_model.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/

ocp_solver:
	gcc $(ACADOS_FLAGS) -c acados_solver_mav_nmpc_tracker_model.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/ \
	-I $(INCLUDE_PATH)/qpOASES_e/

sim_solver:
	gcc $(ACADOS_FLAGS) -c acados_sim_solver_mav_nmpc_tracker_model.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/ \
	-I $(INCLUDE_PATH)/qpOASES_e/

example: ocp_solver main
	gcc $(ACADOS_FLAGS) -o main_mav_nmpc_tracker_model $(EX_OBJ) $(OBJ) -L $(LIB_PATH) \
	-lacados -lhpipm -lblasfeo \
	-lqpOASES_e \
	-lm \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) \
	-I $(INCLUDE_PATH)/acados/ \
	-I $(INCLUDE_PATH)/qpOASES_e/


example_sim: sim_solver main_sim
	gcc $(ACADOS_FLAGS) -o main_sim_mav_nmpc_tracker_model $(EX_SIM_OBJ) $(MODEL_OBJ) $(SIM_OBJ) -L $(LIB_PATH) \
	-lacados -lhpipm -lblasfeo \
	-lqpOASES_e \
	-lm \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/acados/ \

bundled_shared_lib: casadi_fun ocp_solver sim_solver
	gcc $(ACADOS_FLAGS) -shared -o libacados_solver_mav_nmpc_tracker_model.so $(OBJ) \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) \
	-L $(LIB_PATH) \
	-lacados -lhpipm -lblasfeo \
	-lqpOASES_e \
	-lm \

ocp_shared_lib: casadi_fun ocp_solver
	gcc $(ACADOS_FLAGS) -shared -o libacados_ocp_solver_mav_nmpc_tracker_model.so $(OCP_OBJ) $(MODEL_OBJ) \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) \
	-L$(EXTERNAL_DIR) -l$(EXTERNAL_LIB) \
	-L $(LIB_PATH) -lacados -lhpipm -lblasfeo \
	-lqpOASES_e \
	-lm \

sim_shared_lib: casadi_fun sim_solver
	gcc $(ACADOS_FLAGS) -shared -o libacados_sim_solver_mav_nmpc_tracker_model.so $(SIM_OBJ) $(MODEL_OBJ) -L$(EXTERNAL_DIR) -l$(EXTERNAL_LIB) \
	-L $(LIB_PATH) -lacados -lhpipm -lblasfeo \
	-lqpOASES_e \
	-lm \

clean:
	rm -f *.o
	rm -f *.so
	rm -f main_mav_nmpc_tracker_model

clean_ocp_shared_lib:
	rm -f libacados_ocp_solver_mav_nmpc_tracker_model.so
	rm -f acados_solver_mav_nmpc_tracker_model.o
