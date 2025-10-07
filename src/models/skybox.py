import os
import math
import sys
import numpy as np
import pygame
from OpenGL.GL import *

class Skybox:
    def __init__(self, size=1000.0):
        self.size = size
        self.texture_id = None
        self.texture_loaded = False
        
        # Set the path to the skybox texture
        self.texture_path = os.path.join(os.path.dirname(__file__), '..', 'textures', 'BlueSkySkybox.png')
        
        # Load the texture
        self.load_texture()
        
        self.initialized = True
    
    def load_texture(self):
    
        # Reset texture loaded flag
        self.texture_loaded = False
        
        try:
            # Check if file exists and is accessible
            if not os.path.exists(self.texture_path):
                raise FileNotFoundError(f"Texture file not found: {self.texture_path}")
                
            # Generate a texture ID
            self.texture_id = glGenTextures(1)
            if not self.texture_id:
                raise RuntimeError("Failed to generate OpenGL texture ID")
                
            # Bind the texture
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            # Load the skybox texture using Pygame
            try:
                texture_surface = pygame.image.load(self.texture_path)
                texture_surface = pygame.transform.flip(texture_surface, False, True)  # Flip vertically
                width, height = texture_surface.get_size()
                
                # Convert to RGBA if not already
                if texture_surface.get_bytesize() != 4:  # Not RGBA
                    texture_surface = texture_surface.convert_alpha()
                
                # Get the raw pixel data
                texture_data = pygame.image.tostring(texture_surface, 'RGBA', 1)
                
                # Load the texture data into OpenGL
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
                            GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
                
                # Generate mipmaps
                glGenerateMipmap(GL_TEXTURE_2D)
                
                self.texture_loaded = True
                
            except pygame.error as e:
                raise
                
        except Exception as e:
            print(f"ERROR: Failed to load skybox texture: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to a simple blue gradient
            try:
                width, height = 2, 2
                texture_data = [
                    [135, 206, 250, 255], [135, 206, 250, 255],
                    [25, 25, 112, 255], [25, 25, 112, 255]
                ]
                texture_data = np.array(texture_data, dtype=np.uint8)
                
                if self.texture_id is None:
                    self.texture_id = glGenTextures(1)
                    
                glBindTexture(GL_TEXTURE_2D, self.texture_id)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
                            GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
                
                self.texture_loaded = True
                
            except Exception as fallback_error:
                self.texture_loaded = False
        
        finally:
            # Always unbind the texture when done
            glBindTexture(GL_TEXTURE_2D, 0)

    def draw(self):
        if not hasattr(self, 'initialized') or not self.initialized:
            return
            
        try:
            # Save current matrix and attributes
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            
            # Set up for skybox rendering
            glDisable(GL_DEPTH_TEST)  # Disable depth test so skybox is always behind everything
            glDisable(GL_LIGHTING)    # Disable lighting for the skybox
            glDisable(GL_BLEND)       # Disable blending
            
            # Enable texturing if we have a texture
            if self.texture_loaded and self.texture_id is not None:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, self.texture_id)
                glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            
            # Draw the skybox as a sphere with texture coordinates
            slices = 32
            stacks = 16
            radius = self.size / 2.0
            
            # Draw the sphere using immediate mode with texture coordinates
            for i in range(stacks):
                # Calculate the current and next latitude
                lat0 = math.pi * (-0.5 + (i) / stacks)
                z0 = math.sin(lat0)  # Calculate z0 for the current latitude
                zr0 = math.cos(lat0)
                
                lat1 = math.pi * (-0.5 + (i + 1) / stacks)
                z1 = math.sin(lat1)
                zr1 = math.cos(lat1)
                
                # Calculate texture coordinates - adjust v0 and v1 to fix rotation
                v0 = 1.0 - (i / stacks)  # Invert v-coordinate to fix orientation
                v1 = 1.0 - ((i + 1) / stacks)
                
                # Start a new quad strip for this stack
                glBegin(GL_QUAD_STRIP)
                for j in range(slices + 1):
                    # Calculate longitude and texture coordinates
                    lng = 2 * math.pi * (j) / slices
                    x = math.cos(lng)
                    y = math.sin(lng)
                    u = 1.0 - (j / slices)  # Invert u-coordinate to fix rotation
                    
                    # Calculate normals (pointing inward for skybox)
                    nx = -x * zr0
                    ny = -z0
                    nz = -y * zr0
                    # Set texture coordinate, normal and vertex for first point
                    glTexCoord2f(u, v0)
                    glNormal3f(nx, ny, nz)
                    glVertex3f(x * zr0 * radius, z0 * radius, y * zr0 * radius)
                    
                    # Calculate normals for second point
                    nx = -x * zr1
                    ny = -z1
                    nz = -y * zr1
                    
                    # Set texture coordinate, normal and vertex for second point
                    glTexCoord2f(u, v1)
                    glNormal3f(nx, ny, nz)
                    glVertex3f(x * zr1 * radius, z1 * radius, y * zr1 * radius)
                glEnd()
            
            # Reset color to white
            glColor3f(1.0, 1.0, 1.0)
            
        except Exception as e:
            print(f"ERROR in skybox.draw(): {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            # Unbind texture if it was bound
            if self.texture_loaded and self.texture_id is not None:
                glBindTexture(GL_TEXTURE_2D, 0)
                
            # Restore OpenGL state
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glDisable(GL_TEXTURE_2D)
            
            # Restore matrix
            try:
                glPopMatrix()
            except:
                # If pop fails, reset the matrix stack
                glLoadIdentity()
    
    
    def __del__(self):
        # Clean up the texture if it was loaded
        if hasattr(self, 'texture_id') and self.texture_id is not None:
            try:
                # Check if Python is shutting down
                import sys
                if sys is not None and hasattr(sys, 'modules') and hasattr(sys, 'meta_path') is not None:
                    glDeleteTextures([self.texture_id])
            except Exception as e:
                # Ignore any errors during cleanup, especially during interpreter shutdown
                if 'sys' in locals() and sys is not None and hasattr(sys, 'meta_path') and sys.meta_path is not None:
                    # Only print the error if we're not shutting down
                    print(f"Warning during skybox cleanup: {e}")
