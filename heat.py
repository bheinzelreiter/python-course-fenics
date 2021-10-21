import dolfinx
import dolfinx.plot
from dolfinx import plot
from dolfinx.io import XDMFFile
import ufl
from petsc4py import PETSc
from mpi4py import MPI

import numpy as np

import pyvista
from pyvistaqt import BackgroundPlotter
from mesh_generation import generate_mesh, radiator_marker, wall_marker

# Define temporal parameters
t = 0                  # Start time
T = 40.0               # Final time
num_steps = 1000
dt = T / num_steps     # Time step size
conductivity = 1.0              # Window conductivity

# Define mesh
mesh, ft = generate_mesh(0.5)

# Plot mesh
topology, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
grid.plot(show_edges=True)

# Define problem on mesh
V = dolfinx.FunctionSpace(mesh, ("CG", 1))

# Create initial condition
def initial_condition(x):
    return x[0] * 0
u_n = dolfinx.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)
u_n.x.scatter_forward()

# Create boundary condition
# Define Dirichlet boundary conditions
def boundary_D(x):
    return np.logical_and( \
        np.logical_and(x[0] > 4 - 1e-5, x[0] < 6 + 1e-5), \
        np.logical_and(x[1] > 4 - 1e-5, x[1] < 6 + 1e-5))

dofs_D = dolfinx.fem.locate_dofs_geometrical(V, boundary_D)
u_D = dolfinx.Function(V)
with u_D.vector.localForm() as loc:
    loc.set(5.0)
fdim = mesh.topology.dim - 1
facets_D = np.array(ft.indices[ft.values == radiator_marker])
dofs_D = dolfinx.fem.locate_dofs_topological(V, fdim, facets_D)
bc = dolfinx.DirichletBC(u_D, dofs_D)

# Define Neumann boundary conditions
def boundary_R(x):
    values = []
    for xvalue in x[0]:
        if (2 < xvalue) and (xvalue < 4):
            values.append(-10 * conductivity)
        else:
            values.append(0.0)
    return values
u_R = dolfinx.Function(V)
u_R.interpolate(boundary_R)

def boundary_k(x):
    values = []
    for xvalue in x[0]:
        if (2 < xvalue) and (xvalue < 4):
            values.append(conductivity)
        else:
            values.append(0.0)
    return values

k = dolfinx.Function(V)
k.interpolate(boundary_k)

# Define output file
xdmf = XDMFFile(MPI.COMM_WORLD, "diffusion.xdmf", "w")
xdmf.write_mesh(mesh)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = dolfinx.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
uh.x.scatter_forward()
xdmf.write_function(uh, t)

ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = dolfinx.Constant(mesh, 0)
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + dt * k * u * v * ds(wall_marker)
L = (u_n + dt * f) * v * ufl.dx + dt * u_R * v * ds(wall_marker)

A = dolfinx.fem.assemble_matrix(a, bcs=[bc])
A.assemble()
b = dolfinx.fem.create_vector(L)

solver = PETSc.KSP().create(mesh.mpi_comm())
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# Prepare animated plot
topology, cell_types = plot.create_vtk_topology(mesh, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
grid.point_data["u"] = uh.compute_point_values().real
grid.set_active_scalars("u")
plotter = BackgroundPlotter(title="concentration", auto_update=True)
plotter.add_mesh(grid, clim=[-1, 2])
plotter.view_xy(True)

for i in range(num_steps):
    t += dt

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    dolfinx.fem.assemble_vector(b, L)
    
    # Apply Dirichlet boundary condition to the vector
    dolfinx.fem.apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    with uh.vector.localForm() as loc, u_n.vector.localForm() as loc_n:
        loc.copy(result=loc_n)

    # Write solution to file
    xdmf.write_function(uh, t)
    
    # Plot solution
    plotter.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
    grid.point_data["u"] = uh.compute_point_values().real
    grid.set_active_scalars("u")

xdmf.close()