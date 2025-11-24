import bpy
from mathutils import Vector, Euler

# Changes the camera properties
cam = bpy.data.cameras["Camera"]
cam.lens = 67
cam.sensor_width = 10
cam.display_size = 0.1

'''
#для генерации на харизме
grain_sizes = [0.005, 0.003, 0.001]
'''

grain_sizes = [0.005]  # для генерации локально
file_number = 0

for cubeRadius in grain_sizes:
    file_number += 1

    with open("list_" + str(file_number) + ".txt", "r") as file:
        str1 = file.readline()
        str1 = str1.replace('\n', "")
        str1 = str1.replace('[', "")
        str1 = str1.replace(']', "")
        str1 = str1.replace(',', "")
        cubes = str1.split()

        str2 = file.readline()
        str2 = file.readline()
        str2 = str2.replace('[', "")
        str2 = str2.replace(']', "")
        str2 = str2.replace(',', "")
        rotations = str2.split()

    x = float(cubes[0])
    y = float(cubes[1])
    z = float(cubes[2])
    loc = Vector([x, y, z])
    x = float(cubes[0])
    y = float(cubes[1])
    z = float(cubes[2])
    rot = Euler((x, y, z), 'XYZ')

    # Add the first cube (others will be duplicated from it)
    # Note: in blender 2.8 size arg is used instead of radius
    bpy.ops.mesh.primitive_cube_add(size=cubeRadius, location=loc, rotation=rot)

    cube = bpy.context.scene.objects['Cube']
    mat = bpy.data.materials.get("Material")
    cube.data.materials.append(mat)

    for i in range(3, len(cubes), 3):
        x = float(cubes[i])
        y = float(cubes[i + 1])
        z = float(cubes[i + 2])
        loc = Vector([x, y, z])
        x = float(cubes[i])
        y = float(cubes[i + 1])
        z = float(cubes[i + 2])
        rot = Euler((x, y, z), 'XYZ')

        dupliCube = cube.copy()
        dupliCube.location = loc
        dupliCube.rotation_euler = rot

        # bpy.context.scene.objects.link( dupliCube )
        # in blender 2.8 an api change requires to use the collection instead of the scene
        bpy.context.collection.objects.link(dupliCube)

    # Creates a list of camera locations
    # для генерации на харизме
    camera_locations = []
    y = 95
    while y > 0:
        x = 0
        while x < 95:
            camera_locations.append((x / 100.0, y / 100.0, 0.95))
            x += 10
        y -= 10

    '''camera_locations = [(0, 0.95 , 0.95), (0.1, 0.95, 0.95), (0.2, 0.95, 0.95)]'''  # для генерации локально

    # makes 100 images
    photo_number = 0
    for cam_location in camera_locations:
        photo_number += 1

        # change the location of camera
        camobj = bpy.data.objects["_cam"]
        camobj.location = cam_location

        # render and save the image
        scene = bpy.context.scene
        scene.camera = camobj
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = 'C:/Users/HOME/Desktop/Pobedit/GrainGenerator/blender' + str(file_number) + '_' + str(
            photo_number) + '.png'
        bpy.ops.render.render(write_still=1)

    '''
    #delete all the grains
    bpy.ops.object.select_all(action='SELECT')
    bpy.data.objects['_cam'].select_set(False)
    bpy.ops.object.delete()
    '''
