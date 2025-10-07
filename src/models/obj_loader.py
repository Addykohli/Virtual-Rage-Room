import os
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import numpy as np

class Material:
    def __init__(self, **kwargs):
        self.name = ""
        self.texture = None
        self.ambient = [0.2, 0.2, 0.2]
        self.diffuse = [0.8, 0.8, 0.8]
        self.specular = [1.0, 1.0, 1.0]
        self.shininess = 100.0
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def bind(self):
        if self.texture:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture)
        else:
            glDisable(GL_TEXTURE_2D)
        
        glMaterialfv(GL_FRONT, GL_AMBIENT, self.ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, self.diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, self.specular)
        glMaterialf(GL_FRONT, GL_SHININESS, self.shininess)

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file."""
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.materials = {}
        self.current_material = 'default'
        self.display_list = None
        
        # Bounding box information
        self.min_vertex = [float('inf'), float('inf'), float('inf')]
        self.max_vertex = [float('-inf'), float('-inf'), float('-inf')]
        self.size = [0, 0, 0]
        self.center = [0, 0, 0]
        self.scale_factor = 1.0
        
        # Create a default material
        default_mtl = Material(name='default')
        default_mtl.diffuse = [0.8, 0.1, 0.1]  # Red color
        self.materials['default'] = default_mtl
        
        # Load MTL file if it exists
        mtl_path = os.path.splitext(filename)[0] + '.mtl'
        if os.path.exists(mtl_path):
            self.load_mtl(mtl_path)
        
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
                
            values = line.split()
            if not values:
                continue
            
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = [v[0], v[2], v[1]]
                self.vertices.append(v)
                # Update bounding box
                for i in range(3):
                    self.min_vertex[i] = min(self.min_vertex[i], v[i])
                    self.max_vertex[i] = max(self.max_vertex[i], v[i])
            
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            
            elif values[0] == 'usemtl':
                if values[1] in self.materials:
                    self.current_material = values[1]
            
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    # Handle different OBJ face formats (v, v/vt, v//vn, v/vt/vn)
                    if len(w) == 1:  # v
                        face.append(int(w[0]))
                        texcoords.append(0)
                        norms.append(0)
                    elif len(w) == 2:  # v/vt
                        face.append(int(w[0]))
                        texcoords.append(int(w[1]) if w[1] else 0)
                        norms.append(0)
                    elif len(w) == 3:  # v//vn or v/vt/vn
                        face.append(int(w[0]))
                        if w[1]:  # Has texture coordinate
                            texcoords.append(int(w[1]))
                        else:
                            texcoords.append(0)
                        if w[2]:  # Has normal
                            norms.append(int(w[2]))
                        else:
                            norms.append(0)
                
                # Use the current material or default if none specified
                material = self.current_material if self.current_material in self.materials else 'default'
                self.faces.append((face, norms, texcoords, material))
        
        # Calculate model dimensions and center
        if self.vertices:
            self.size = [self.max_vertex[i] - self.min_vertex[i] for i in range(3)]
            self.center = [(self.min_vertex[i] + self.max_vertex[i]) / 2 for i in range(3)]
            
            # Calculate scale factor to fit in a 2x2x2 cube by default
            max_dim = max(self.size)
            if max_dim > 0:
                self.scale_factor = 2.0 / max_dim
            
            # Generate display list
            self.display_list = glGenLists(1)
            if self.display_list:
                glNewList(self.display_list, GL_COMPILE)
                self._render_immediate()
                glEndList()
    
    def load_texture(self, image_path):
        try:
            img = Image.open(image_path)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = np.array(list(img.getdata()), np.uint8)
            
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            if img.mode == 'RGB':
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
            elif img.mode == 'RGBA':
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            
            return texture
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
    
    def load_mtl(self, filename):
        current_mtl = None
        mtl_dir = os.path.dirname(filename)
        
        for line in open(filename, 'r'):
            if line.startswith('#'):
                continue
                
            values = line.split()
            if not values:
                continue
                
            if values[0] == 'newmtl':
                current_mtl = values[1]
                self.materials[current_mtl] = Material(name=current_mtl)
            
            elif current_mtl is not None:
                mtl = self.materials[current_mtl]
                
                if values[0] == 'map_Kd':
                    # Handle texture paths with spaces
                    tex_path = ' '.join(values[1:])
                    tex_path = os.path.join(mtl_dir, tex_path)
                    mtl.texture = self.load_texture(tex_path)
                
                elif values[0] == 'Ka':
                    mtl.ambient = list(map(float, values[1:4]))
                
                elif values[0] == 'Kd':
                    mtl.diffuse = list(map(float, values[1:4]))
                
                elif values[0] == 'Ks':
                    mtl.specular = list(map(float, values[1:4]))
                
                elif values[0] == 'Ns':
                    mtl.shininess = float(values[1])
    
    def get_scale_factor(self, target_size=2.0):
        """Get the scale factor to fit the model within target_size units in its largest dimension."""
        if not self.vertices:
            return 1.0
        max_dim = max(self.size)
        return target_size / max_dim if max_dim > 0 else 1.0
    
    def _render_immediate(self):
        """Render the model using immediate mode."""
        glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glEnable(GL_COLOR_MATERIAL)
        
        # Apply scaling and centering
        glPushMatrix()
        glScalef(self.scale_factor, self.scale_factor, self.scale_factor)
        glTranslatef(-self.center[0], -self.center[1], -self.center[2])
        
        try:
            current_mtl = None
            for face in self.faces:
                vertices, normals, texture_coords, material_name = face
                
                # Only change material when necessary
                if material_name != current_mtl and material_name in self.materials:
                    self.materials[material_name].bind()
                    current_mtl = material_name
                
                glBegin(GL_POLYGON)
                for i in range(len(vertices)):
                    if normals and i < len(normals) and 0 <= normals[i] - 1 < len(self.normals):
                        glNormal3fv(self.normals[normals[i] - 1])
                    if texture_coords and i < len(texture_coords) and 0 <= texture_coords[i] - 1 < len(self.texcoords):
                        glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                    if 0 <= vertices[i] - 1 < len(self.vertices):
                        glVertex3fv(self.vertices[vertices[i] - 1])
                glEnd()
                
        except Exception as e:
            # Fallback to a simple shape if rendering fails
            import traceback
            traceback.print_exc()
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_LIGHTING)
            glColor3f(1.0, 0.0, 0.0)  # Red color
            glBegin(GL_TRIANGLES)
            glVertex3f(0, 0, 0)
            glVertex3f(1, 0, 0)
            glVertex3f(0, 1, 0)
            glEnd()
        
        glPopMatrix()  # Pop the model transformation
        glPopAttrib()
    
    def render(self):
        """Render the model using display list if available, otherwise use immediate mode."""
        if self.display_list is not None:
            try:
                glCallList(self.display_list)
                return
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.display_list = None
        
        # Fall back to immediate mode rendering
        self._render_immediate()
    
    def generate(self):
        # This is a no-op in this implementation
        # The actual generation happens in __init__
        pass

    def get_bullet_mesh(self):
        """
        Returns mesh data as (vertices, indices) suitable for PyBullet's GEOM_MESH.
        Vertices: flat list of floats [x0, y0, z0, x1, y1, z1, ...]
        Indices: flat list of ints [i0, i1, i2, ...] (triangles)
        """
        # OBJ indices are 1-based, PyBullet expects 0-based
        vertices = [coord for v in self.vertices for coord in v]
        indices = []
        for face, _, _, _ in self.faces:
            # Triangulate faces (OBJ can have quads or ngons)
            if len(face) < 3:
                continue
            # Fan triangulation
            for i in range(1, len(face) - 1):
                indices.extend([
                    face[0] - 1,
                    face[i] - 1,
                    face[i + 1] - 1
                ])
        return vertices, indices
