import os
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import numpy as np

class Material:
    def __init__(self, **kwargs):
        # Default material properties (RGBA format)
        self.name = "default_material"
        self.texture = None
        self.ambient = [0.2, 0.2, 0.2, 1.0]  # Ka
        self.diffuse = [0.8, 0.8, 0.8, 1.0]  # Kd
        self.specular = [0.5, 0.5, 0.5, 1.0]  # Ks
        self.emission = [0.0, 0.0, 0.0, 1.0]  # Ke
        # Clamp shininess to valid OpenGL range (0.0 to 128.0)
        self.shininess = min(max(0.0, 100.0), 128.0)  # Ns (0-128)
        self.opacity = 1.0  # d or 1-Tr
        self.has_texture = False
        
        # Apply any keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def bind(self):
        """Bind this material's properties to the current OpenGL state."""
        try:
            # Set material properties
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, self.ambient)
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, self.diffuse)
            
            # Set specular and shininess with validation
            specular = list(self.specular) if len(self.specular) >= 4 else [0.5, 0.5, 0.5, 1.0]
            shininess = max(0.0, min(float(self.shininess), 128.0))
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular)
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, self.emission)
            
            # Handle texturing
            if self.texture and self.has_texture:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, self.texture)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            else:
                glDisable(GL_TEXTURE_2D)
            
            # Handle transparency
            if self.opacity < 1.0:
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glDepthMask(GL_FALSE)
            else:
                glDisable(GL_BLEND)
                glDepthMask(GL_TRUE)
                
            # Set color for immediate mode rendering
            glColor4f(self.diffuse[0], self.diffuse[1], self.diffuse[2], self.opacity)
            
        except Exception as e:
            print(f"[ERROR] Error in Material.bind(): {e}")
            # Fallback to default material on error
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, [0.0, 0.0, 0.0, 1.0])
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_BLEND)
            glDepthMask(GL_TRUE)

class OBJ:
    def __init__(self, filename, swapyz=False):
        """Loads a Wavefront OBJ file."""
        self.filename = os.path.abspath(filename)  # Store absolute path for reference
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.materials = {}
        self.current_material = 'default'
        self.display_list = None
        self.textures = set()  # Track loaded textures for cleanup
        
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
    
    def __del__(self):
        # Clean up OpenGL resources
        try:
            # Check if OpenGL is still available
            if hasattr(self, 'display_list') and self.display_list is not None and glDeleteLists:
                glDeleteLists(self.display_list, 1)
                self.display_list = None
            
            # Clean up any loaded textures
            if hasattr(self, 'textures'):
                for tex_id in self.textures:
                    if glIsTexture(tex_id):
                        glDeleteTextures([tex_id])
                self.textures.clear()
        except Exception as e:
            # Ignore errors during cleanup, especially during interpreter shutdown
            import sys
            if sys is not None and hasattr(sys, 'meta_path') and sys.meta_path is not None:
                # Only print the error if we're not shutting down
                print(f"Warning: Error during OBJ cleanup: {e}")
    def load_texture(self, image_path):
        """Load a texture from an image file and return the OpenGL texture ID."""
        if not image_path:
            print("Error: No image path provided")
            return None
            
        try:            
            # Check if file exists and is readable
            if not os.path.exists(image_path):
                print(f"Texture file not found: {image_path}")
                return None
                
            if not os.access(image_path, os.R_OK):
                print(f"Cannot read texture file (permission denied): {image_path}")
                return None
            
            # Load the image file
            try:
                img = Image.open(image_path)
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Flip the image (OpenGL expects 0,0 at bottom-left, PIL at top-left)
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                
                img_data = np.array(img, np.uint8)
                width, height = img.size
                
                if width == 0 or height == 0:
                    print(f"Invalid image dimensions: {width}x{height}")
                    return None
                    
            except Exception as img_error:
                print(f"Error loading image {image_path}: {str(img_error)}")
                return None
            
            # Generate and bind texture
            texture = glGenTextures(1)
            if not texture:
                print("Failed to generate texture ID")
                return None
                
            # Save current texture binding state
            current_texture = glGetIntegerv(GL_TEXTURE_BINDING_2D)
            
            # Generate and bind texture
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            # Upload texture data
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 
                        0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            
            # Generate mipmaps
            glGenerateMipmap(GL_TEXTURE_2D)
            
            # Restore previous texture binding
            glBindTexture(GL_TEXTURE_2D, current_texture)
            
            return texture
            
        except Exception as e:
            import traceback
            print(f"Error loading texture {image_path}: {e}")
            traceback.print_exc()
            return None
    
    def load_mtl(self, filename):
        if not os.path.exists(filename):
            print(f"Material file not found: {filename}")
            return
            
        current_mtl = None
        mtl_dir = os.path.dirname(filename)
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading material file {filename}: {e}")
            return
            
        # Store the MTL directory for texture loading
        self.mtl_dir = mtl_dir
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            values = line.split()
            if not values:
                continue
                
            if values[0] == 'newmtl':
                current_mtl = values[1]
                self.materials[current_mtl] = Material(name=current_mtl)
            
            elif current_mtl is not None:
                mtl = self.materials.get(current_mtl)
                if not mtl:
                    continue
                
                try:
                    if values[0] == 'map_Kd':
                        # Handle texture paths with spaces and different path formats
                        tex_name = ' '.join(values[1:]).strip('"')
                        
                        # Get the directory of the OBJ file (which might be different from mtl_dir)
                        obj_dir = os.path.dirname(self.filename) if hasattr(self, 'filename') else mtl_dir
                        
                        # Try multiple possible paths in order of likelihood
                        possible_paths = [
                            os.path.join(mtl_dir, tex_name),  # Original path relative to MTL
                            os.path.join(mtl_dir, os.path.basename(tex_name)),  # Just filename in MTL dir
                            os.path.join(obj_dir, tex_name),  # Relative to OBJ file
                            os.path.join(obj_dir, os.path.basename(tex_name)),  # Just filename in OBJ dir
                            os.path.join('models', 'model_objs', tex_name),  # Common models directory
                            os.path.join('models', 'model_objs', os.path.basename(tex_name)),  # Just filename in model_objs
                            os.path.join('src', 'models', 'model_objs', tex_name),  # Full path from project root
                            os.path.join('src', 'models', 'model_objs', os.path.basename(tex_name)),  # Just filename in full path
                        ]
                        
                        # Add absolute paths for all possible locations
                        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                        possible_paths.extend([
                            os.path.join(project_root, 'models', 'model_objs', tex_name),
                            os.path.join(project_root, 'models', 'model_objs', os.path.basename(tex_name)),
                            os.path.join(project_root, 'src', 'models', 'model_objs', tex_name),
                            os.path.join(project_root, 'src', 'models', 'model_objs', os.path.basename(tex_name)),
                        ])
                        
                        # Try each possible path
                        texture_loaded = False
                        for path in possible_paths:
                            try:
                                if os.path.exists(path):
                                    texture = self.load_texture(path)
                                    if texture is not None:
                                        mtl.texture = texture
                                        mtl.has_texture = True
                                        self.textures.add(texture)  # Track the texture for cleanup
                                        texture_loaded = True
                                        break
                                    else:
                                        print(f"Failed to load texture (returned None): {path}")
                                else:
                                    print(f"Texture file not found: {path}")
                            except Exception as e:
                                print(f"Error loading texture {path}: {str(e)}")
                        
                        if not texture_loaded:
                            print(f"Warning: Could not load any texture for {current_mtl}. Tried the following paths:" + "\n  " + "\n  ".join(possible_paths))
                        
                        if not mtl.has_texture:
                            print(f"Warning: Could not load texture '{tex_name}' for material {current_mtl}. Using default material color.")
                            # Set a default color based on material name for better visual distinction
                            if 'body' in current_mtl.lower():
                                mtl.diffuse = [0.8, 0.1, 0.1]  # Red for body
                            elif 'wheel' in current_mtl.lower():
                                mtl.diffuse = [0.2, 0.2, 0.2]  # Dark gray for wheels
                            elif 'glass' in current_mtl.lower():
                                mtl.diffuse = [0.1, 0.3, 0.8, 0.5]  # Semi-transparent blue for glass
                            else:
                                # Generate a consistent color based on material name hash
                                import hashlib
                                hash_val = int(hashlib.md5(current_mtl.encode()).hexdigest(), 16)
                                mtl.diffuse = [
                                    ((hash_val & 0xFF) / 255.0) * 0.5 + 0.3,  # R: 0.3-0.8
                                    ((hash_val >> 8 & 0xFF) / 255.0) * 0.5 + 0.3,  # G: 0.3-0.8
                                    ((hash_val >> 16 & 0xFF) / 255.0) * 0.5 + 0.3  # B: 0.3-0.8
                                ]
                    
                    elif values[0] == 'Ka':
                        # Ambient color (RGB)
                        mtl.ambient = list(map(float, values[1:4]))
                        if len(mtl.ambient) == 3:
                            mtl.ambient.append(1.0)  # Add alpha if not present
                    
                    elif values[0] == 'Kd':
                        # Diffuse color (RGB)
                        mtl.diffuse = list(map(float, values[1:4]))
                        if len(mtl.diffuse) == 3:
                            mtl.diffuse.append(1.0)  # Add alpha if not present
                    
                    elif values[0] == 'Ks':
                        # Specular color (RGB)
                        mtl.specular = list(map(float, values[1:4]))
                        if len(mtl.specular) == 3:
                            mtl.specular.append(1.0)  # Add alpha if not present
                            
                    elif values[0] == 'Ke':
                        # Emissive color (RGB)
                        mtl.emission = list(map(float, values[1:4]))
                        if len(mtl.emission) == 3:
                            mtl.emission.append(1.0)  # Add alpha if not present
                    
                    elif values[0] == 'Ns':
                        # Shininess (0-128) - clamp to valid OpenGL range
                        mtl.shininess = max(0.0, min(float(values[1]), 128.0))
                        
                    elif values[0] == 'd' or values[0] == 'Tr':
                        # Transparency/opacity
                        mtl.opacity = float(values[1])
                        if values[0] == 'Tr':  # Tr is 1.0 - d
                            mtl.opacity = 1.0 - mtl.opacity
                        
                    elif values[0] in ['map_Kd', 'map_Ks', 'map_bump', 'bump', 'map_d']:
                        # Handle texture maps
                        tex_name = ' '.join(values[1:]).strip('"')
                        
                        # Try multiple possible paths to find the texture
                        possible_paths = [
                            os.path.join(self.mtl_dir, tex_name),
                            os.path.join(self.mtl_dir, os.path.basename(tex_name)),
                            os.path.join(os.path.dirname(self.filename), tex_name),
                            os.path.join(os.path.dirname(self.filename), os.path.basename(tex_name)),
                            os.path.join('models', 'model_objs', tex_name),
                            os.path.join('models', 'model_objs', os.path.basename(tex_name)),
                            os.path.join('src', 'models', 'model_objs', tex_name),
                            os.path.join('src', 'models', 'model_objs', os.path.basename(tex_name)),
                        ]
                        
                        # Try each possible path
                        texture_loaded = False
                        for path in possible_paths:
                            try:
                                if os.path.exists(path):
                                    texture = self.load_texture(path)
                                    if texture is not None:
                                        if values[0] == 'map_Kd':  # Diffuse texture
                                            mtl.texture = texture
                                            mtl.has_texture = True
                                        # Add other texture types here if needed
                                        texture_loaded = True
                                        break
                                    else:
                                        print(f"Failed to load texture (returned None): {path}")
                            except Exception as e:
                                print(f"Error loading texture {path}: {str(e)}")
                        
                        if not texture_loaded:
                            print(f"Warning: Could not load {values[0]} texture for {current_mtl}")
                        
                except Exception as e:
                    print(f"Error processing material property {values[0] if values else 'unknown'} in {filename}: {e}")
    
    def get_scale_factor(self, target_size=2.0):
        """Get the scale factor to fit the model within target_size units in its largest dimension."""
        if not self.vertices:
            return 1.0
        max_dim = max(self.size)
        return target_size / max_dim if max_dim > 0 else 1.0
    
    def _render_immediate(self):
        """Render the model using immediate mode."""
        # Save OpenGL state
        glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_TEXTURE_BIT | GL_LIGHTING_BIT | GL_DEPTH_BUFFER_BIT)
        try:
            # Set up OpenGL state
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glEnable(GL_NORMALIZE)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            
            # Apply model transformations
            glPushMatrix()
            glScalef(self.scale_factor, self.scale_factor, self.scale_factor)
            glTranslatef(-self.center[0], -self.center[1], -self.center[2])
            
            try:
                current_mtl = None
                for face in self.faces:
                    vertices, normals, texture_coords, material_name = face
                    
                    # Only change material when necessary
                    if material_name != current_mtl:
                        if material_name in self.materials:
                            self.materials[material_name].bind()
                        else:
                            # Use default material if material not found
                            default_mtl = Material()
                            default_mtl.diffuse = [0.8, 0.1, 0.1, 1.0]  # Red color
                            default_mtl.bind()
                        current_mtl = material_name
                    
                    # Draw the face
                    glBegin(GL_POLYGON)
                    for i in range(len(vertices)):
                        # Set normal if available
                        if normals and i < len(normals) and 0 <= normals[i] - 1 < len(self.normals):
                            glNormal3fv(self.normals[normals[i] - 1])
                        
                        # Set texture coordinate if available and material has a texture
                        current_mat = self.materials.get(material_name, None)
                        if current_mat and current_mat.texture and current_mat.has_texture:
                            if texture_coords and i < len(texture_coords) and 0 <= texture_coords[i] - 1 < len(self.texcoords):
                                glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                        
                        # Set vertex position
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
            
        finally:
            # Always restore the OpenGL state
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
            self.textures = set()  # Track loaded textures for cleanup
        
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
