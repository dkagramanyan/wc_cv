import bpy
from random import random
from mathutils import Vector, Euler

#единицы измерения - метры

# min and max values for each axis for the random numbers
ranges = {
    'x' : { 'min' : -0.5, 'max' : 0.5 },
    'y' : { 'min' : -0.5, 'max' : 0.5 },
    'z' : { 'min' : -0.025, 'max' : 0.025 }
}

# Generates a random number within the axis minmax range
randLocInRange = lambda axis: ranges[axis]['min'] + random() * ( ranges[axis]['max'] - ranges[axis]['min'] )

#Changes the camera properties
cam = bpy.data.cameras["Camera"]
cam.lens =  67
cam.sensor_width = 10
cam.display_size = 0.1

#Creates a list of camera locations
'''
#для генерации на харизме
camera_locations = []
y = 45
while y > -45:
    x = -45
    while x < 45:
        camera_locations.append((x/100.0, y/100.0, 0.7))
        x+=10
    y-=10
'''
camera_locations = [(-0.45, -0.45, 0.7), (-0.35, -0.45, 0.7), (-0.25, -0.45, 0.7)]#для генерации локально

'''
#для генерации на харизме
grain_sizes = [0.005, 0.004, 0.003, 0.002, 0.001]
'''
grain_sizes = [0.05, 0.04, 0.03] #для генерации локально
sample_number = 0

for cubeRadius in grain_sizes:
    # Generates a random number within the axis minmax range
    randLocInRange = lambda axis: ranges[axis]['min'] + random() * ( ranges[axis]['max'] - ranges[axis]['min'] )

    size  = (0.05/cubeRadius**3)*(6/7) # Number of cubes
    
    cubes = []  # Cube coordinates list
    
    sample_number += 1
    
    loopIterations = 0
    while len( cubes ) < size:
        # Generate a random 3D coordinate
        loc = Vector([ randLocInRange( axis ) for axis in ranges.keys() ])    
        # Add coordinate to cube list
        cubes.append( loc )

    # Add the first cube (others will be duplicated from it)
    # Note: in blender 2.8 size arg is used instead of radius
    bpy.ops.mesh.primitive_cube_add( size = cubeRadius, location = cubes[0] )

    cube = bpy.context.scene.objects['Cube']
    mat = bpy.data.materials.get("Material")
    cube.data.materials.append(mat)

    # Add all other cubes
    for c in cubes[1:]:
        dupliCube = cube.copy()
        dupliCube.location = c
        dupliCube.rotation_euler = Euler((random(), random(), random()),'XYZ')
        dupliCube.scale = (1, 1, 1)
        # bpy.context.scene.objects.link( dupliCube )
        # in blender 2.8 an api change requires to use the collection instead of the scene
        bpy.context.collection.objects.link(dupliCube)
     
    #make 100 images    
    photo_number = 0
    for cam_location in camera_locations:
        photo_number+=1
        
        #change the location of camera
        camobj = bpy.data.objects["_cam"]
        camobj.location = cam_location
        
        #render and save the image
        scene = bpy.context.scene
        scene.camera = camobj
        scene.render.image_settings.file_format='PNG'
        scene.render.filepath='C:/Users/HOME/Desktop/Pobedit/GrainGenerator/blender'+str(sample_number)+'_'+str(photo_number)+'.png'
        bpy.ops.render.render(write_still=1)
        
    #delete all the grains
    bpy.ops.object.select_all(action='SELECT')
    bpy.data.objects['_cam'].select_set(False)
    bpy.ops.object.delete()