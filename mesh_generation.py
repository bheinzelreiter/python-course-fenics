import gmsh
import numpy as np
from gmsh_helpers import gmsh_model_to_mesh

radiator_marker, wall_marker = 2, 3

def generate_mesh(h):
    gmsh.initialize()
    gdim = 2  # The dimension of the considered domain

    # Define the structure of the domain
    rectangle = gmsh.model.occ.addRectangle(0,0,0, 10, 10)
    radiator = gmsh.model.occ.addRectangle(4,4,0, 2, 2)
    domain = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, radiator)])
    gmsh.model.occ.synchronize()

    # Assign markers to volume elements and boundary elements
    volumes = gmsh.model.getEntities(dim=gdim)
    room_marker = 1
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], room_marker)
    gmsh.model.setPhysicalName(volumes[0][0], room_marker, "Room")

    radiator_boundary, wall_boundary = [], []
    boundaries = gmsh.model.getBoundary(volumes)
    for boundary in boundaries:
        bounding_box = gmsh.model.occ.getBoundingBox(boundary[0], boundary[1])
        if (np.logical_or( \
            np.allclose([bounding_box[0]], [0], atol=1e-4), \
            np.allclose([bounding_box[3]], [10], atol=1e-4))):
            wall_boundary.append(boundary[1])
        else:
            radiator_boundary.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, wall_boundary, wall_marker)
    gmsh.model.addPhysicalGroup(1, radiator_boundary, radiator_marker)

    # Set mesh accuracy
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.optimize("Netgen")
    
    # Generate mesh
    return gmsh_model_to_mesh(gmsh.model, cell_data=False, facet_data=True, gdim=gdim)