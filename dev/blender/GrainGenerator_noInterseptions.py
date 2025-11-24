import bpy
from random import random
from mathutils import Vector, Euler
import math

#единицы измерения - метры

#checks if 2 cubes are intersepting using their coordinates and rotations
def ifIntersept(loc1, loc2, rot1, rot2, size):

    #creates a list of coordinates of every vertice
    coord_list_1 = []
    coord_list_2 = []
    
    #calculates the coordinates and adds to list
    coord_list_1.append([loc1[0]+size/2, loc1[1]+size/2, loc1[2]+size/2])
    coord_list_1.append([loc1[0]+size/2, loc1[1]+size/2, loc1[2]-size/2])
    coord_list_1.append([loc1[0]+size/2, loc1[1]-size/2, loc1[2]+size/2])
    coord_list_1.append([loc1[0]+size/2, loc1[1]-size/2, loc1[2]-size/2])
    coord_list_1.append([loc1[0]-size/2, loc1[1]+size/2, loc1[2]+size/2])
    coord_list_1.append([loc1[0]-size/2, loc1[1]+size/2, loc1[2]-size/2])
    coord_list_1.append([loc1[0]-size/2, loc1[1]-size/2, loc1[2]+size/2])
    coord_list_1.append([loc1[0]-size/2, loc1[1]-size/2, loc1[2]-size/2])
    
    coord_list_2.append([loc2[0]+size/2, loc2[1]+size/2, loc2[2]+size/2])
    coord_list_2.append([loc2[0]+size/2, loc2[1]+size/2, loc2[2]-size/2])
    coord_list_2.append([loc2[0]+size/2, loc2[1]-size/2, loc2[2]+size/2])
    coord_list_2.append([loc2[0]+size/2, loc2[1]-size/2, loc2[2]-size/2])
    coord_list_2.append([loc2[0]-size/2, loc2[1]+size/2, loc2[2]+size/2])
    coord_list_2.append([loc2[0]-size/2, loc2[1]+size/2, loc2[2]-size/2])
    coord_list_2.append([loc2[0]-size/2, loc2[1]-size/2, loc2[2]+size/2])
    coord_list_2.append([loc2[0]-size/2, loc2[1]-size/2, loc2[2]-size/2])
    
    for coord in coord_list_1:
        coord[1]=coord[1]*math.cos(rot1[0]) + coord[2]*math.sin(rot[0])
        coord[2]=-coord[1]*math.sin(rot1[0]) + coord[2]*math.cos(rot[0])
        
        coord[0]=coord[0]*math.cos(rot1[1]) + coord[2]*math.sin(rot[1])
        coord[2]=-coord[0]*math.sin(rot1[1]) + coord[2]*math.cos(rot[1])
        
        coord[0]=coord[0]*math.cos(rot1[2]) - coord[1]*math.sin(rot[2])
        coord[1]=-coord[0]*math.sin(rot1[2]) + coord[1]*math.cos(rot[2])
    
    print(coord_list_1)    
    for coord in coord_list_2:
        coord[1]=coord[1]*math.cos(rot1[0]) + coord[2]*math.sin(rot[0])
        coord[2]=-coord[1]*math.sin(rot1[0]) + coord[2]*math.cos(rot[0])
        
        coord[0]=coord[0]*math.cos(rot1[1]) + coord[2]*math.sin(rot[1])
        coord[2]=-coord[0]*math.sin(rot1[1]) + coord[2]*math.cos(rot[1])
        
        coord[0]=coord[0]*math.cos(rot1[2]) - coord[1]*math.sin(rot[2])
        coord[1]=-coord[0]*math.sin(rot1[2]) + coord[1]*math.cos(rot[2])
    
    #Create a list of x, y, z coordinates of each cube
    x1_list = []
    y1_list = []
    z1_list = []
     
    for c in coord_list_1:
        x1_list.append(c[0])
        y1_list.append(c[1])
        z1_list.append(c[2])
            
    x2_list = []
    y2_list = []
    z2_list = []
     
    for c in coord_list_2:
        x2_list.append(c[0])
        y2_list.append(c[1])
        z2_list.append(c[2])
    
    #checks if cubes untersept        
    if ( (max(x1_list) >= min(x2_list)) and (min(x1_list) <= max(x2_list))
        and (max(y1_list) >= min(y2_list)) and (min(y1_list) <= max(y2_list))
        and (max(z1_list) >= min(z2_list)) and (min(z1_list)) <= max(z2_list) ):
            return False
    else:
        return True
        
        

# min and max values for each axis for the random numbers
ranges = {
    'x' : { 'min' : -0.5, 'max' : 0.5 },
    'y' : { 'min' : -0.5, 'max' : 0.5 },
    'z' : { 'min' : -0.025, 'max' : 0.025 }
}


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
grain_sizes = [0.005, 0.003, 0.001]
'''

grain_sizes = [0.05] #для генерации локально
sample_number = 0

for cubeRadius in grain_sizes:
    # Generates a random number within the axis minmax range
    randLocInRange = lambda axis: ranges[axis]['min'] + random() * ( ranges[axis]['max'] - ranges[axis]['min'] )

    size  = (0.05/cubeRadius**3)*(1/7) # Number of cubes
    
    cubes = []  # Cube coordinates list
    rotations = [] #cube rotations list
    
    sample_number += 1
    
    loopIterations = 0
    
    # Generate a random 3D coordinate
    loc = Vector([ randLocInRange( axis ) for axis in ranges.keys() ])    
    # Add coordinate to cube list
    cubes.append( loc )

    #Generates a random rotation
    rot = Euler((random(), random(), random()),'XYZ')
    #Add rotation to rotations list
    rotations.append (rot)
    
    # Add the first cube (others will be duplicated from it)
    # Note: in blender 2.8 size arg is used instead of radius
    bpy.ops.mesh.primitive_cube_add( size = cubeRadius, location = cubes[0], rotation = rotations[0] )

    cube = bpy.context.scene.objects['Cube']
    mat = bpy.data.materials.get("Material")
    cube.data.materials.append(mat)
    

    # Add all other cubes
    c = 0
    while c <= size:  
        while True:
            flag = True
            # Generate a random 3D coordinate
            loc = Vector([ randLocInRange( axis ) for axis in ranges.keys() ])
            # Generate a random rotation    
            rot = Euler((random(), random(), random()),'XYZ')
            for l in range (len(cubes)):
                if ifIntersept(loc, cubes[l], rot, rotations[l], cubeRadius) is False:
                    flag = False
            if flag is True:
                break
                      
        # Add coordinate to cube list
        cubes.append( loc )
        rotations.append(rot)
        dupliCube = cube.copy()
        dupliCube.location = loc
        dupliCube.rotation_euler = rot
        dupliCube.scale = (1, 1, 1)
        
        # bpy.context.scene.objects.link( dupliCube )
        # in blender 2.8 an api change requires to use the collection instead of the scene
        bpy.context.collection.objects.link(dupliCube)
        c+=1
     
    #makes 100 images    
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
    '''
    #delete all the grains
    bpy.ops.object.select_all(action='SELECT')
    bpy.data.objects['_cam'].select_set(False)
    bpy.ops.object.delete()
    '''   
        