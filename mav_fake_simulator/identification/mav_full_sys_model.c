/*    
   Use this file to create a MEX function that specifies the model
   structure and equations. The MEX file syntax is
      [dx, y] = mymodel(t, x, u, p1, p2, ..., pn, auxvar)
   where
      * t is the time (scalar).
      * x is the state vector at time t (column vector).
      * u is the vector of inputs at time t (column vector).
      * p1, p2,... pn: values of the estimated parameters specified
        in the IDNLGREY model.
      * auxvar: a cell array containing auxiliary data in any format
        (optional).
      * dx is the vector of state derivatives at time t (column vector).
      * y is the vector of outputs at time t.
   
   To create the MEX file "mav_sys_model", do the following:
      1) Save this file as "mav_sys_model.c".
      2) Define the number NY of outputs below.
      3) Specify the state derivative equations in COMPUTE_DX below.
      4) Specify the output equations in COMPUTE_Y below.
      5) Build the MEX file using
            >> mex mav_sys_model.c
*/

/* Include libraries. */
#include "mex.h"
#include "math.h"

/* Specify the number of outputs here. */
#define NY 5    // px, py, pz, roll, pitch
#define G 9.8066 // gravity constant

/* State equations. */
void compute_dx(
    double *dx,  /* Vector of state derivatives (length nx). */
    double t,    /* Time t (scalar). */
    double *x,   /* State vector (length nx). */
    double *u,   /* Input vector (length nu). */
    double **p,  /* p[j] points to the j-th estimated model parameters (a double array). */
    const mxArray *auxvar  /* Cell array of additional data. */
   )
{
    /*
      Define the state equation dx = f(t, x, u, p[0],..., p[np-1], auvar)
      in the body of this function.
    */
    /*
      Accessing the contents of auxvar:
      
      Use mxGetCell to fetch pointers to individual cell elements, e.g.:
          mxArray* auxvar1 = mxGetCell(auxvar, 0);
      extracts the first cell element. If this element contains double
      data, you may obtain a pointer to the double array using mxGetPr:
          double *auxData = mxGetPr(auxvar1);
      
      See MATLAB documentation on External Interfaces for more information
      about functions that manipulate mxArrays.
    */
    
    /* Example code from ODE function for DCMOTOR example
       used in idnlgreydemo1 (dcmotor_c.c) follows.
    */
    
    /* Estimated model parameters. */
    double *roll_time_constant, *roll_gain, *pitch_time_constant, *pitch_gain;
    double *drag_coefficient_x, *drag_coefficient_y;
    roll_time_constant = p[0];
    roll_gain = p[1];
    pitch_time_constant = p[2];
    pitch_gain = p[3];
    drag_coefficient_x = p[4];
    drag_coefficient_y = p[5];
    
    /* State. */
    double px, py, pz, vx, vy, vz, roll, pitch;
    px = x[0];
    py = x[1];
    pz = x[2];
    vx = x[3];
    vy = x[4];
    vz = x[5];
    roll = x[6];
    pitch = x[7];
    
    /* Control. */
    double roll_cmd, pitch_cmd, thrust_cmd;
    roll_cmd = u[0];
    pitch_cmd = u[1];
    thrust_cmd = u[2];      // already divided by mass 
    
    /* Aux data. */
    //mxArray* auxvar1 = mxGetCell(auxvar, 0);
    //double *auxData = mxGetPr(auxvar1);
    //double yaw = auxData[0];
    double yaw = 0.0;
    
    /* Drag. */
    double drag_acc_x, drag_acc_y;
    drag_acc_x = cos(pitch)*cos(yaw)*drag_coefficient_x[0]*thrust_cmd*vx - cos(pitch)*sin(yaw)*drag_coefficient_x[0]*thrust_cmd*vy + sin(pitch)*drag_coefficient_x[0]*thrust_cmd*vz;
    drag_acc_y = (cos(roll)*sin(yaw) - cos(yaw)*sin(pitch)*sin(roll))*drag_coefficient_y[0]*thrust_cmd*vx - (cos(roll)*cos(yaw) + sin(pitch)*sin(roll)*sin(yaw))*drag_coefficient_y[0]*thrust_cmd*vy - cos(pitch)*sin(roll)*drag_coefficient_y[0]*thrust_cmd*vz;
            
    /* Derivatives. */
    dx[0] = vx;
    dx[1] = vy;
    dx[2] = vz;
    dx[3] = (cos(roll) * cos(yaw) * sin(pitch) + sin(roll) * sin(yaw)) * thrust_cmd - drag_acc_x;
    dx[4] = (cos(roll) * sin(pitch) * sin(yaw) - cos(yaw) * sin(roll)) * thrust_cmd - drag_acc_y;
    dx[5] = -G + cos(pitch) * cos(roll) * thrust_cmd;
    dx[6] = (roll_gain[0] * roll_cmd - roll) / roll_time_constant[0];
    dx[7] = (pitch_gain[0] * pitch_cmd - pitch) / pitch_time_constant[0];
       
}

/* Output equations. */
void compute_y(
    double *y,   /* Vector of outputs (length NY). */
    double t,    /* Time t (scalar). */
    double *x,   /* State vector (length nx). */
    double *u,   /* Input vector (length nu). */
    double **p,  /* p[j] points to the j-th estimated model parameters (a double array). */
    const mxArray *auxvar  /* Cell array of additional data. */
   )
{
    /*
      Define the output equation y = h(t, x, u, p[0],..., p[np-1], auvar)
      in the body of this function.
    */
    y[0] = x[0]; /* px. */
    y[1] = x[1]; /* py. */
    y[2] = x[2]; /* pz. */
    y[3] = x[6]; /* r. */
    y[4] = x[7]; /* p. */
}



/*----------------------------------------------------------------------- *
   DO NOT MODIFY THE CODE BELOW UNLESS YOU NEED TO PASS ADDITIONAL
   INFORMATION TO COMPUTE_DX AND COMPUTE_Y
 
   To add extra arguments to compute_dx and compute_y (e.g., size
   information), modify the definitions above and calls below.
 *-----------------------------------------------------------------------*/

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    /* Declaration of input and output arguments. */
    double *x, *u, **p, *dx, *y, *t;
    int     i, np, nu, nx;
    const mxArray *auxvar = NULL; /* Cell array of additional data. */
    
    if (nrhs < 3) {
        mexErrMsgIdAndTxt("IDNLGREY:ODE_FILE:InvalidSyntax",
        "At least 3 inputs expected (t, u, x).");
    }
    
    /* Determine if auxiliary variables were passed as last input.  */
    if ((nrhs > 3) && (mxIsCell(prhs[nrhs-1]))) {
        /* Auxiliary variables were passed as input. */
        auxvar = prhs[nrhs-1];
        np = nrhs - 4; /* Number of parameters (could be 0). */
    } else {
        /* Auxiliary variables were not passed. */
        np = nrhs - 3; /* Number of parameters. */
    }
    
    /* Determine number of inputs and states. */
    nx = mxGetNumberOfElements(prhs[1]); /* Number of states. */
    nu = mxGetNumberOfElements(prhs[2]); /* Number of inputs. */
    
    /* Obtain double data pointers from mxArrays. */
    t = mxGetPr(prhs[0]);  /* Current time value (scalar). */
    x = mxGetPr(prhs[1]);  /* States at time t. */
    u = mxGetPr(prhs[2]);  /* Inputs at time t. */
    
    p = mxCalloc(np, sizeof(double*));
    for (i = 0; i < np; i++) {
        p[i] = mxGetPr(prhs[3+i]); /* Parameter arrays. */
    }
    
    /* Create matrix for the return arguments. */
    plhs[0] = mxCreateDoubleMatrix(nx, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(NY, 1, mxREAL);
    dx      = mxGetPr(plhs[0]); /* State derivative values. */
    y       = mxGetPr(plhs[1]); /* Output values. */
    
    /*
      Call the state and output update functions.
      
      Note: You may also pass other inputs that you might need,
      such as number of states (nx) and number of parameters (np).
      You may also omit unused inputs (such as auxvar).
      
      For example, you may want to use orders nx and nu, but not time (t)
      or auxiliary data (auxvar). You may write these functions as:
          compute_dx(dx, nx, nu, x, u, p);
          compute_y(y, nx, nu, x, u, p);
    */
    
    /* Call function for state derivative update. */
    compute_dx(dx, t[0], x, u, p, auxvar);
    
    /* Call function for output update. */
    compute_y(y, t[0], x, u, p, auxvar);
    
    /* Clean up. */
    mxFree(p);
}
