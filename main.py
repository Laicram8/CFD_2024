# Development of a Python code to solve the two-dimensional
# Navier-Stokes equations on a rectangular domain.
# Import some relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.animation
import math
import scipy.sparse as sp
import scipy.linalg as scl
from scipy.sparse.linalg import splu
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-R',default=25,help="Choose from different model")
parser.add_argument('-Lx',default=1,type=int,help="Number of samples used for prediction")
parser.add_argument('-Ly',default=1,type=int,help="Number of samples used for prediction")
parser.add_argument('-Nx',default=30,type=int,help="Number of samples used for prediction")
parser.add_argument('-Ny',default=30,type=int,help="Number of samples used for prediction")
parser.add_argument('-t',default=0.01,type=int,help="Number of samples used for prediction")
args = parser.parse_args()

params = {'legend.fontsize': 12,
          'legend.loc':'best',
          'figure.figsize': (8,5),
          'lines.markerfacecolor':'none',
          'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize':12,
          'ytick.labelsize':12,
          'grid.alpha':0.6}
pylab.rcParams.update(params)

# Some useful functions:
def avg(A,axis=0):
    """
    Averaging function to go from cell centres (pressure nodes)
    to cell corners (velocity nodes) and vice versa.
    avg acts on index idim; default is idim=1.
    """
    if (axis==0):
        B = (A[:-1,:] + A[1:,:])/2.
    elif (axis==1):
        B = (A[:,:-1] + A[:,1:])/2.
    else:
        raise ValueError('Wrong value for axis')
    return B

def DD(n,h):
    """
    One-dimensional finite-difference derivative matrix
    of size n times n for second derivative:
    h^2 * f’’(x_j) = -f(x_j-1) + 2*f(x_j) - f(x_j+1)

    Homogeneous Neumann boundary conditions on the boundaries
    are imposed, i.e.
    f(x_0) = f(x_1)
    if the wall lies between x_0 and x_1. This gives then
    h^2 * f’’(x_j) = + f(x_0) - 2*f(x_1) + f(x_2)
    = + f(x_1) - 2*f(x_1) + f(x_2)
    = f(x_1) + f(x_2)

    For n=5 and h=1 the following result is obtained:

    A =
    -1 1 0 0 0
    1 -2 1 0 0
    0 1 -2 1 0
    0 0 1 -2 1
    0 0 0 1 -1
    """
    data = np.concatenate( (-np.ones((n-1,1)), 2*np.ones((n,1)), -np.ones((n-1,1))))
    diags = np.array([-1,0,1])
    A = sp.spdiags(data.T, diags, n, n) / h**2
    return A

# Homemade version of Matlab tic and toc functions
def tic():
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

# Simulation parameters:
Pr = 0.71
Re = args.R
# Ri = 0.
dt = args.t
Tf = 20
Lx = 1.
Ly = 1.
Nx = args.Nx
Ny = args.Ny
namp = 0.
ig = 200

# Discretisation in space and time, and definition of boundary conditions:
# number of iterations
Nit = None
# edge coordinates
x = np.linspace(None)
y = np.linspace(None)
# grid spacing
hx = args.Lx/args.Nx
hy = args.Ly/args.Ny

# boundary conditions
Utop = 1.; Ttop = 1.; Tbottom = 0.;
uN = x*0 + Utop; uN = uN[:,np.newaxis];
vN = avg(x)*0; vN = vN[:,np.newaxis];
uS = None; uS = uS[:,np.newaxis];
vS = None; vS = vS[:,np.newaxis];
uW = avg(y)*0; uW = uW[np.newaxis,:];
vW = None; vW = vW[np.newaxis,:];
uE = avg(y)*0; uE = uE[np.newaxis,:];
vE = y*0; vE = vE[np.newaxis,:];

tN = None; tS =None;

# Pressure correction and pressure Poisson equation:
# Compute system matrices for pressure
# Laplace operator on cell centres: Fxx + Fyy
# First set homogeneous Neumann condition all around
Lp = np.kron(DD(Nx-1,hx).toarray(),DD(Ny-1,hy).toarray()) + np.kron(sp.eye(Nx-1).toarray(),sp.eye(Ny-1).toarray());
# Set one Dirichlet value to fix pressure in that point
Lp[:,0] = None; Lp[0,:] = None; Lp[0,0] = None;
Lp_lu, Lp_piv = scl.lu_factor(Lp)
Lps = sp.csc_matrix(Lp)
Lps_lu = splu(Lps)

# Initial conditions
U = np.zeros((Nx-1,Ny))
V = np.zeros((Nx,Ny-1))
T = None + \
    namp*(np.random.rand(Nx,Ny)-0.5);

# Main time-integration loop. Write output file "cavity.mp4" if
if (ig>0):
    metadata = dict(title='Lid-driven cavity', artist='SG2212')
    writer = matplotlib.animation.FFMpegWriter(fps=15, metadata=metadata)
    matplotlib.use("Agg")
    fig=plt.figure()
    writer.setup(fig,"cavity.mp4",dpi=200)

    # progress bar
    print('[ | | | | ]')
    tic()
    for k in range(Nit):
        # print("Iteration k=%i time=%.2e" % (k,k*dt))

        # include all boundary points for u and v (linear extrapolation
        # for ghost cells) into extended array (Ue,Ve)
        Ue = np.vstack((uW, U, uE)); Ue = np.hstack( (2*uS-Ue[:,0,np.newaxis], Ue, 2*uN-Ue[:,-1,np.newaxis]));
        Ve = None

        # averaged (Ua,Va) of u and v on corners
        Ua = None
        Va = None

        # construct individual parts of nonlinear terms
        dUVdx = None
        dUVdy = None
        Ub = avg( Ue[:,1:-1],0);
        Vb = None
        dU2dx = np.diff( None )/hx;
        dV2dy = None

        # treat viscosity explicitly
        viscu = np.diff( None,axis=1,n=2 )/hy**2;
        viscv = np.diff( None,axis=0,n=2 )/hx**2;

        # compose final nonlinear term + explicit viscous terms
        U = None
        V = None

        # pressure correction, Dirichlet P=0 at (1,1)
        rhs = (np.diff( None)/hx + np.diff( None,axis=1)/hy)/dt;
        rhs = np.reshape(None);
        rhs[0] = 0;

        # different ways of solving the pressure-Poisson equation:
        P = Lps_lu.solve(None)

        P = np.reshape(None

        # apply pressure correction
        U = U - dt*np.diff(None)/hx;
        V = V - dt*np.diff(None)/hy;

        # Temperature equation
        None

        # do postprocessing to file
        if (ig>0 and np.floor(k/ig)==k/ig):
            plt.clf()
            plt.contourf(avg(x),avg(y),T.T,levels=np.arange(0,1.05,0.05))
            plt.gca().set_aspect(1.)
            plt.colorbar()
            plt.title(f'Temperature at t={k*dt:.2f}')
            writer.grab_frame()

        # update progress bar
        if np.floor(51*k/Nit)>np.floor(51*(k-1)/Nit):
            print('.',end='')

    # finalise progress bar
    print(' done. Iterations k=%i time=%.2f' % (k,k*dt))
    toc()

    if (ig>0):
        writer.finish()

# Visualisation of the flow field at the end time:
%matplotlib notebook

Ua = np.hstack( (uS,avg(np.vstack((uW,U,uE)),1),uN));
Va = np.vstack((vW,avg(np.hstack((vS,V, vN)),0),vE));
plt.figure()
plt.contourf(x,y,np.sqrt(Ua**2+Va**2).T,20)
plt.quiver(x,y,Ua.T,Va.T)
plt.gca().set_aspect(1.)
plt.colorbar()
plt.title(f'Velocity at t={k*dt:.2f}')
plt.show()

# compute divergence on cell centres
div = (np.diff( np.vstack( (uW,U, uE)),axis=0)/hx + np.diff(
    np.hstack(( vS, V, vN)),axis=1)/hy)
plt.figure()
plt.pcolor(avg(x),avg(y),div.T,shading='nearest')
plt.gca().set_aspect(1.)
plt.colorbar()
plt.title(f'Divergence at t={k*dt:.2f}')
plt.show()

# Analysis of the pressure Poisson equation:
# Matrix structure
plt.figure()
plt.spy(Lp)
plt.show()
# Size, rank and null space:
Lp.shape
np.linalg.matrix_rank(Lp)
scl.null_space
