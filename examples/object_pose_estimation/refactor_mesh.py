from trimesh import load_mesh
import pyrender
import os

mesh_path = 'Repositories/solar_panels/solar_panel_02/meshes/obj/base_link.obj'
mesh_path = os.path.join(os.path.expanduser('~'), mesh_path)
mesh = load_mesh(mesh_path)
for key in mesh.geometry:
    trimesh = mesh.geometry[key]
    if len(trimesh.vertices) == 1849:
        new_mesh = trimesh

pymesh = pyrender.Mesh.from_trimesh(new_mesh)
scene = pyrender.Scene()
scene.add(pymesh)
scene.ambient_light = [0.5, 0.5, 0.5, 2.0]
pyrender.Viewer(scene, use_raymond_lighting=True)
