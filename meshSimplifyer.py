import open3d as o3d

# Cargar la malla
mesh = o3d.io.read_triangle_mesh("full_brain_binary.stl")

# Verificar si la malla se cargó correctamente
if not mesh.has_triangles():
    raise ValueError("Error: La malla no se cargó correctamente.")

print("Malla original:", mesh)

# Reducir la cantidad de triángulos (50% de reducción)
num_triangulos = len(mesh.triangles)
target_triangles = int(num_triangulos * 0.5)

if target_triangles <= 0:
    raise ValueError("Error: El número de triángulos objetivo es inválido.")

mesh_simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

# 📌 **Nuevo: Calcular normales antes de guardar**
mesh_simplified.compute_vertex_normals()

# Guardar la nueva malla
o3d.io.write_triangle_mesh("modelo_simplificado.stl", mesh_simplified)

# Visualizar la malla simplificada
o3d.visualization.draw_geometries([mesh_simplified])