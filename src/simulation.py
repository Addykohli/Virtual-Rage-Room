import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import glGenTextures, glBindTexture, glTexParameteri, glTexImage2D, glEnable, glGenerateMipmap
from OpenGL.GL import GL_TEXTURE_2D, GL_RGBA, GL_UNSIGNED_BYTE, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR
from OpenGL.GL import GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_REPEAT
import os
import sys
import math
import time
import random
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.player import Player
from models.obj_loader import OBJ
from physics_world import PhysicsWorld

# Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FOV = 45.0
NEAR_PLANE = 0.1
FAR_PLANE = 1000.0
GRID_SIZE = 20
TILE_SIZE = 1.0

class Simulation:
    def __init__(self):
        pygame.init()
        try:
            pygame.mixer.init()
            self.audio_enabled = True
        except pygame.error:
            self.audio_enabled = False
        
        self.display = (WINDOW_WIDTH, WINDOW_HEIGHT)
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("Simulation")
        
        # Mouse control settings
        self.mouse_grabbed = True
        pygame.event.set_grab(True)  # Grab the cursor for relative motion
        pygame.mouse.set_visible(False)  # Hide cursor
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.VIDEORESIZE, pygame.MOUSEBUTTONUP])
        self.relative_mouse_mode = True
        self.last_mouse_pos = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)  # Initialize last mouse position
        
        # Shooting/targeting
        self.target_point = [0, 0, 0]  # Where the crosshair is pointing
        self.bullets = []  # List to store active bullets
        self.bullet_speed = 50.0  # Speed of bullets
        self.bullet_radius = 0.05  # Size of bullets
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # Enable lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        
        # Set up light
        glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 10.0, 5.0, 1.0))  # Position the light higher up
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1.0))   # Slightly brighter ambient
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))   # Full white diffuse
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))  # Add specular highlights
        
        # Set up material properties
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set material properties
        glMaterialfv(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)
        
        # Initialize model library and load default model
        from model_lib import ModelLibrary
        from models.skybox import Skybox
        
        self.model_lib = ModelLibrary()
        self.currPlayerModel = self.model_lib.load_model('Punisher')
        
        # Initialize skybox (larger than the scene)
        try:
            self.skybox = Skybox(size=2000.0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.skybox = None
        
        # Audio resources
        self.shoot_sound = None
        self.hit_sound = None
        self.shoot_channel = None
        self.hit_channel = None
        self.hit_sound_playing = False
        self.load_audio_from_model()
        
        # Structure presets
        self.structure_presets = self.model_lib.get_environment_models()
        self.current_structure_index = 0
        self.active_structure = None
        
        # Set up the perspective
        self.setup_projection()
        
        # Initialize physics world with ground at y=0
        self.physics = PhysicsWorld(gravity=(0, -9.81*2, 0), grid_height=0.0)
        
        # Enable depth testing and other OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Create player positioned at ground level in physics coordinates
        self.player = Player(x=0, y=0, z=0)  # Initialize at origin
        self.player.ground_level = 0.0  # Set ground level to match physics world
        self.player.y = 0  # Position will be adjusted in draw_player
        
        # Override player's check_collision method to use physics world
        def check_collision(x, y, z):
            # Check if the player would collide with any structure at the given position
            # Exclude the player's own structure from collision checks
            y -= 1
            player_structure_id = getattr(self, 'player_structure_id', None)
            if player_structure_id is None:
                return False
                
            # Get player collision size
            player_size = getattr(self, 'player_collision_size', [1.0, 1.8, 1.0])
            player_half_size = [s / 2.0 for s in player_size]
            
            # Player collision box spans from y to y + height
            player_min_y = y
            player_max_y = y + player_size[1]
            
            for structure_id, structure in self.physics.structures.items():
                if structure_id == player_structure_id:
                    continue  # Skip self-collision
                pos = self.physics.get_structure_position(structure_id)
                if pos is None:
                    continue
                    
                # Get structure size and calculate bounds
                size = structure.get('size', [1, 1, 1])
                half_size = [s / 2.0 for s in size]
                struct_min_y = pos[1] - half_size[1]
                struct_max_y = pos[1] + half_size[1]
                
                # Check for AABB collision
                if (abs(x - pos[0]) < (player_half_size[0] + half_size[0]) and
                    abs(z - pos[2]) < (player_half_size[2] + half_size[2]) and
                    player_max_y > struct_min_y and player_min_y < struct_max_y):
                    return True
            return False
        self.player.check_collision = check_collision
        
        # Add player as a kinematic mesh structure to the physics world
        player_mesh_path = self.currPlayerModel.metadata.get('mesh_path', None)
        player_mesh_obj = None
        if player_mesh_path:
            from models.obj_loader import OBJ
            player_mesh_obj = OBJ(player_mesh_path)
        
        # Define proper player collision size (width, height, depth)
        player_collision_size = [1.0, 1.8, 1.0]  # 1m wide, 1.8m tall, 1m deep
        self.player_collision_size = player_collision_size
        
        self.player_structure_id = self.physics.add_structure(
            position=[self.player.x, self.player.y, self.player.z],  # Bottom of collision box at ground level
            size=player_collision_size,
            mass=0.0,  # Mass 0 makes it kinematic
            color=(0.0, 0.0, 0.0, 0.0),  # Invisible - only used for collision
            rotation=[0, 0, 0],
            fill='player',
            metadata={'stiff': True, 'kinematic': True},  # Mark as kinematic
            mesh_obj=player_mesh_obj
        )
        
        # Camera settings
        self.camera_distance = 9.0
        self.camera_height = self.currPlayerModel.metadata['camera_height']
        self.min_zoom = 2.0  # Minimum zoom distance
        self.max_zoom = 20.0  # Maximum zoom distance
        self.zoom_speed = 0.9  # Zoom speed multiplier
        
        # Mouse control
        self.mouse_x, self.mouse_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        pygame.mouse.set_pos(self.mouse_x, self.mouse_y)
        
        # Simulation variables
        self.running = True
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.delta_time = 0.0
        self.show_side_panel = False  # Initially hidden
        
        # Firing rate control
        self.fire_rate = 10.0  # Bullets per second
        self.fire_delay = 1.0 / self.fire_rate  # Time between shots in seconds
        self.last_shot_time = 0.0  # When the last shot was fired
        self.is_firing = False  # Whether the fire button is being held down
        self.bullet_impulse_active = False  # Whether bullet impulse mode is active
        
        # Colors
        self.colors = {
            'grid': (0.3, 0.3, 0.3),
            'player': (0.0, 1.0, 0.0),
            'background': (0.1, 0.1, 0.15),
            'menu_button': (0.2, 0.2, 0.25),
            'menu_button_hover': (0.3, 0.3, 0.35),
            'lock_button': (0.2, 0.2, 0.25),  # Default dark color
            'lock_button_active': (0.2, 0.6, 0.2),  # Green when active
            'lock_button_hover': (0.3, 0.3, 0.35),  # Lighter on hover
            'cube_button': (0.1, 0.3, 0.8),
            'cube_button_hover': (0.2, 0.4, 0.9),
            'cube_preview': (0.0, 0.5, 1.0, 0.4)
        }

        # UI elements
        structure_button_size = 40
        margin = 10
        button_bg_width = 60
        
        # Menu button (top right)
        self.menu_button_rect = pygame.Rect(
            self.display[0] - structure_button_size - margin,
            margin,
            structure_button_size,
            structure_button_size
        )
        
        # Second panel button (below menu button)
        self.panel2_button_rect = pygame.Rect(
            self.menu_button_rect.x,
            self.menu_button_rect.bottom + margin,
            structure_button_size,
            structure_button_size
        )
        self.show_side_panel2 = False  # Track second panel visibility
        
        # Lock button (bottom right)
        self.lock_button_rect = pygame.Rect(
            self.display[0] - structure_button_size - margin,
            self.display[1] - structure_button_size - margin,
            structure_button_size,  
            structure_button_size
        )

        # Reset button (above the lock button)
        self.reset_button_rect = pygame.Rect(
            self.lock_button_rect.x,
            self.lock_button_rect.y - structure_button_size - margin,
            structure_button_size,
            structure_button_size
        )

        # Structure placement state
        self.placing_cube = False
        self.cube_preview_pos = [0.0, 0.0, 0.0]
        self.cube_size = 1.0
        self.active_cube_size = [1.0, 1.0, 1.0]
        self.active_total_size = [1.0, 1.0, 1.0]
        self.structure_buttons = []
        self.prev_left_mouse = False
        self.prev_right_mouse = False
        self.awaiting_cube_release = False
        self.ground_texture = None
        self.set_active_structure()
        
        # Load ground texture
        self.load_ground_texture()
        
        # Textures will be loaded on-demand from the structure configuration
        self.texture_cache = {}
        
        # Pre-load small brick textures
        self.small_bricks_textures = []
        self.load_small_bricks_textures()
    
    def setup_projection(self):
        glViewport(0, 0, self.display[0], self.display[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV, (self.display[0] / self.display[1]), NEAR_PLANE, FAR_PLANE)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def load_small_bricks_textures(self):
        """Load all small brick textures from the textures/small_bricks directory"""
        small_bricks_dir = os.path.join(os.path.dirname(__file__), 'textures', 'small_bricks')
        if os.path.exists(small_bricks_dir):
            for filename in os.listdir(small_bricks_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    texture_path = os.path.join(small_bricks_dir, filename)
                    try:
                        texture = self.load_texture(texture_path)
                        if texture is not None:
                            self.small_bricks_textures.append(texture)
                    except Exception as e:
                        pass
        
    def load_texture(self, image_path):
        """Load a texture from file and return the texture ID"""
        try:
            # Load the image
            texture_surface = pygame.image.load(image_path)
            texture_data = pygame.image.tostring(texture_surface, 'RGBA', 1)
            
            # Generate a texture ID
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            # Upload texture data
            width, height = texture_surface.get_size()
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
                        GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            
            return texture_id
            
        except Exception as e:
            return None
            
    def load_ground_texture(self):
        """Load the ground texture from file, supporting both regular images and EXR"""
        try:
            # Try loading the EXR texture first
            exr_path = os.path.join(os.path.dirname(__file__), 'textures', 'gravelly_sand_rough_2k.exr')
            if os.path.exists(exr_path):
                import OpenEXR
                import Imath
                import numpy as np
                
                # Open the EXR file
                exr_file = OpenEXR.InputFile(exr_path)
                header = exr_file.header()
                
                # Get the data window and calculate dimensions
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                
                # List all available channels
                channels = list(header['channels'].keys())
                
                # Try to find the first available channel
                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                
                # Try common channel names in order of likelihood
                channel_to_use = None
                for possible_channel in ['R', 'Y', 'Luma', 'Luminance', 'Roughness', 'roughness']:
                    if possible_channel in channels:
                        channel_to_use = possible_channel
                        break
                
                if not channel_to_use and channels:  # If no common channel found, use the first available
                    channel_to_use = channels[0]
                
                if not channel_to_use:
                    raise ValueError("No valid channels found in EXR file")
                
                channel_data = exr_file.channel(channel_to_use, FLOAT)
                
                # Convert to numpy array
                img_data = np.frombuffer(channel_data, dtype=np.float32)
                img_data = img_data.reshape((height, width))
                
                # Load the color texture
                color_path = os.path.join(os.path.dirname(__file__), 'textures', 'gravelly_sand_diff_2k.jpg')
                if os.path.exists(color_path):
                    # Load the color texture as a surface
                    color_surface = pygame.image.load(color_path).convert()
                    
                    # Resize the color texture to match EXR dimensions if needed
                    if color_surface.get_size() != (width, height):
                        color_surface = pygame.transform.scale(color_surface, (width, height))
                    
                    # Convert color surface to numpy array
                    color_data = pygame.surfarray.array3d(color_surface).astype(np.float32) / 255.0
                    
                    # Combine color with the normal/roughness data
                    # For normal maps, we can use the Y channel for intensity
                    if 'nor_gl' in exr_path.lower() and len(img_data.shape) == 2:
                        # For normal maps, use the Y component for intensity
                        intensity = (img_data + 1.0) * 0.5  # Convert from [-1,1] to [0,1]
                        img_rgb = color_data * intensity[..., np.newaxis]
                    else:
                        # For other maps (like roughness), use as grayscale multiplier
                        img_rgb = color_data * img_data[..., np.newaxis]
                else:
                    # Fallback to grayscale if color texture not found
                    img_rgb = np.stack([img_data] * 3, axis=-1)
                
                # Generate texture ID and bind it
                self.ground_texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, self.ground_texture)
                
                # Set texture parameters
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                
                # Upload texture data
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, 
                            GL_RGB, GL_FLOAT, img_rgb)
                glGenerateMipmap(GL_TEXTURE_2D)
                
                return
                
            # Fall back to regular image if EXR not found
            texture_path = os.path.join(os.path.dirname(__file__), 'textures', 'gravelly_sand_diff_2k.jpg')
            texture_surface = pygame.image.load(texture_path).convert_alpha()
            
            # Get the raw pixel data
            texture_data = pygame.image.tostring(texture_surface, 'RGBA', 1)
            width, height = texture_surface.get_size()
            
            # Generate a texture ID
            self.ground_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.ground_texture)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            # Upload the texture data and generate mipmaps
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
                        GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.ground_texture = None
    
    def draw_ground(self):
        # Draw only the textured ground plane without grid lines
        ground_height = self.physics.grid_height
        grid_size = 50
          # Size of the ground plane
        
        if self.ground_texture:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.ground_texture)
            
            # Enable blending for transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Set material properties
            glColor4f(1.0, 1.0, 1.0, 1.0)
            glBegin(GL_QUADS)
            
            # Draw a large quad with the texture
            repeat = 10.0  # How many times to repeat the texture
            size = grid_size
            
            glTexCoord2f(0, 0)
            glVertex3f(-size, ground_height + 0.01, -size)  # Slightly above y=0
            
            glTexCoord2f(repeat, 0)
            glVertex3f(size, ground_height + 0.01, -size)
            
            glTexCoord2f(repeat, repeat)
            glVertex3f(size, ground_height + 0.01, size)
            
            glTexCoord2f(0, repeat)
            glVertex3f(-size, ground_height + 0.01, size)
            
            glEnd()
            
            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_2D)
    

    def draw_physics_objects(self):
        """Draw all physics objects in the scene"""
        # Draw all structures from the physics world
        for structure_id, structure in self.physics.structures.items():
            # Get position and orientation from physics
            pos = self.physics.get_structure_position(structure_id)
            orn = self.physics.get_structure_orientation(structure_id)
            
            if pos is None or orn is None:
                continue
                
            size = structure['size']
            color = structure['color']
            fill_type = structure.get('fill', 'solid')
            
            # Skip rendering for player structures (they're rendered separately)
            if fill_type == 'player':
                continue
                
            # Draw shadow first (under the object)
            self.draw_rectangular_shadow(pos[0], pos[2], size[0], size[2], alpha=0.25)
            
            # Save current matrix
            glPushMatrix()
            
            # Apply position and rotation
            glTranslatef(pos[0], pos[1], pos[2])
            
            # Convert quaternion to axis-angle for glRotatef
            angle = 2 * math.acos(orn[3])
            if angle > 0.0001:  # Avoid division by zero
                s = math.sqrt(1 - orn[3] * orn[3])
                if s < 0.001:  # Handle singularity
                    axis = [1, 0, 0]
                else:
                    axis = [orn[0]/s, orn[1]/s, orn[2]/s]
                
                # Convert to degrees for glRotatef
                angle_deg = math.degrees(angle)
                glRotatef(angle_deg, axis[0], axis[1], axis[2])
            
            # Set material color
            glColor4f(*color) if len(color) > 3 else glColor3f(*color)
            
            # Draw based on texture ID or fill type
            if 'texture_id' in structure and structure['texture_id'] is not None:
                # Use the stored texture ID
                self.draw_textured_cube(size, structure['texture_id'])
            elif fill_type in ['small_bricks', 'rust_metal', 'concrete']:
                # Fallback to random texture if texture_id is not set
                textures = getattr(self, f'{fill_type}_textures', [])
                if textures:
                    texture = random.choice(textures)
                    self.draw_textured_cube(size, texture)
            else:
                # Default solid color cube
                half = [s / 2.0 for s in size]
                glBegin(GL_QUADS)
                # Front face
                glVertex3f(-half[0], -half[1], -half[2])
                glVertex3f(half[0], -half[1], -half[2])
                glVertex3f(half[0], half[1], -half[2])
                glVertex3f(-half[0], half[1], -half[2])
                # Back face
                glVertex3f(-half[0], -half[1], half[2])
                glVertex3f(-half[0], half[1], half[2])
                glVertex3f(half[0], half[1], half[2])
                glVertex3f(half[0], -half[1], half[2])
                # Top face
                glVertex3f(-half[0], half[1], -half[2])
                glVertex3f(half[0], half[1], -half[2])
                glVertex3f(half[0], half[1], half[2])
                glVertex3f(-half[0], half[1], half[2])
                # Bottom face
                glVertex3f(-half[0], -half[1], -half[2])
                glVertex3f(-half[0], -half[1], half[2])
                glVertex3f(half[0], -half[1], half[2])
                glVertex3f(half[0], -half[1], -half[2])
                # Left face
                glVertex3f(-half[0], -half[1], -half[2])
                glVertex3f(-half[0], -half[1], half[2])
                glVertex3f(-half[0], half[1], half[2])
                glVertex3f(-half[0], half[1], -half[2])
                # Right face
                glVertex3f(half[0], -half[1], -half[2])
                glVertex3f(half[0], half[1], -half[2])
                glVertex3f(half[0], half[1], half[2])
                glVertex3f(half[0], -half[1], half[2])
                glEnd()
            
            # Restore matrix
            glPopMatrix()
    
    def update_target_point(self):
        """Update the point the crosshair is pointing at using raycasting"""
        # Get the center of the screen (crosshair position)
        mouse_x, mouse_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        
        # Get the view and projection matrices
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        
        # Convert screen coordinates to 3D world coordinates
        # First, unproject the near plane point
        win_x = mouse_x
        win_y = viewport[3] - mouse_y  # Convert to OpenGL coordinates
        
        # Get the near and far points in world space
        near_point = list(gluUnProject(win_x, win_y, 0.0, modelview, projection, viewport))
        far_point = list(gluUnProject(win_x, win_y, 1.0, modelview, projection, viewport))
        
        # Calculate ray direction
        ray_direction = [f - n for f, n in zip(far_point, near_point)]
        ray_length = math.sqrt(sum(d*d for d in ray_direction))
        ray_direction = [d/ray_length for d in ray_direction]
        
        # Calculate the end point of the ray
        ray_end = [p + d * 100.0 for p, d in zip(near_point, ray_direction)]
        
        # Cast the ray to find where it hits the ground or objects
        hit_object, hit_point = self.physics.ray_test(near_point, ray_end)
        
        if hit_point is not None:
            self.target_point = hit_point
        else:
            # If no hit, use a point far in the distance
            self.target_point = ray_end
    
    def update_bullets(self):
        """Update bullet positions and remove expired bullets"""
        current_time = pygame.time.get_ticks() / 1000.0  # Current time in seconds
        
        # Update bullet lifetimes and remove expired ones
        for bullet in self.bullets[:]:
            bullet['lifetime'] -= self.delta_time
            if bullet['lifetime'] <= 0:
                self.bullets.remove(bullet)
    
    def shoot(self):
        """Handle shooting mechanics with hitscan and high-impact force"""
        # Get bullet origin offsets from model metadata
        bullet_origins = [[0, -1.0, 0.0]]  # Default offset if not specified
        if hasattr(self.currPlayerModel, 'metadata') and 'bullet_origin' in self.currPlayerModel.metadata:
            # Convert single offset to list if it's not already
            origins = self.currPlayerModel.metadata['bullet_origin']
            bullet_origins = [origins] if isinstance(origins[0], (int, float)) else origins
        
        # Convert player rotation to radians and adjust for model's forward direction
        rad_yaw = math.radians(self.player.rotation_y + 90)  # +90 to align with model's forward
        cos_yaw = math.cos(rad_yaw)
        sin_yaw = math.sin(rad_yaw)
        
        # Fire a bullet for each origin point
        for bullet_offset in bullet_origins:
            # Get the bullet offset
            offset_x, offset_y, offset_z = bullet_offset
            
            # Rotate the offset around the Y-axis (yaw) to match player's facing direction
            # Negative sin terms make it rotate counter-clockwise
            rotated_x = offset_x * cos_yaw + offset_z * sin_yaw
            rotated_z = -offset_x * sin_yaw + offset_z * cos_yaw
            
            # Calculate the final bullet start position
            start_pos = [
                self.player.x + rotated_x,  # X position with rotation
                self.player.y + offset_y,   # Y position (height) - not affected by yaw
                self.player.z + rotated_z   # Z position with rotation
            ]
            
            # Calculate direction to target point
            direction = [
                self.target_point[0] - start_pos[0],
                self.target_point[1] - start_pos[1],
                self.target_point[2] - start_pos[2]
            ]
            
            # Normalize direction
            length = math.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
            if length > 0:
                direction = [d/length for d in direction]
                
                # Calculate end position (far away in the shooting direction)
                end_pos = [
                    start_pos[0] + direction[0] * 1000,  # 1000 units in the shooting direction
                    start_pos[1] + direction[1] * 1000,
                    start_pos[2] + direction[2] * 1000
                ]
            
            # Perform raycast to detect hits
            hit_object_id, hit_position = self.physics.ray_test(start_pos, end_pos)
            
            if hit_object_id is not None:
                # Visual feedback for the hitscan with hit position
                bullet_data = {
                    'type': 'hitscan',
                    'start_pos': start_pos.copy(),
                    'end_pos': hit_position,
                    'hit_position': hit_position,
                    'lifetime': 0.1  # Very short lifetime for the visual effect
                }
                
                # Only apply impulse to non-ground objects (ground has ID -1)
                if hit_object_id != -1:
                    # Get the bullet impulse value from the input field, default to 150 if not set
                    impulse_strength = getattr(self, 'bullet_impulse', 150.0)
                    # Apply the impulse at the hit point
                    impulse = [d * impulse_strength for d in direction]
                    self.physics.apply_impulse(hit_object_id, impulse, hit_position)
                    
                    # Play hit sound, passing the structure ID to use specific sounds if available
                    self.play_hit_sound(hit_object_id)
                    
                    # Add impact effect for objects
                    bullet_data['impact_effect'] = True
                    bullet_data['impulse_strength'] = impulse_strength  # Store for visual feedback if needed
                
                self.bullets.append(bullet_data)
            else:
                # If nothing was hit, just show the full ray
                self.bullets.append({
                    'type': 'hitscan',
                    'start_pos': start_pos.copy(),
                    'end_pos': end_pos,
                    'lifetime': 0.05  # Even shorter lifetime for misses
                })
    
    def draw_bullet_trail(self, bullet):
        """Draw a complete bullet trail with smooth glowing effect"""
        if bullet['type'] != 'hitscan':
            return
            
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive blending for bright effect
        glDepthMask(False)
        
        # Calculate points along the trail
        start = bullet['start_pos']
        end = bullet['end_pos']
        
        # Get the bullet's lifetime progress (0.0 to 1.0)
        if 'spawn_time' not in bullet:
            bullet['spawn_time'] = time.time()
            
        current_time = time.time()
        lifetime = current_time - bullet['spawn_time']
        max_lifetime = 0.1  # Match the lifetime from the shoot() function
        progress = min(lifetime / max_lifetime, 1.0)
        
        # Calculate fade out based on progress
        fade_alpha = 1.0 - progress  # Fade out over time
        
        # Draw multiple lines with varying widths and opacities for a glowing effect
        glLineWidth(1.5)  # Thinner core line (reduced from 4.0)
        glBegin(GL_LINES)
        # Core bright white line
        glColor4f(1.0, 1.0, 1.0, 0.8 * fade_alpha)
        glVertex3f(*start)
        glVertex3f(*end)
        glEnd()
        
        # Outer glow with color transition
        glLineWidth(0.8)  # Thinner outer glow (reduced from 2.0)
        glBegin(GL_LINES)
        # Transition from white to yellow to red
        glColor4f(1.0, 1.0, 0.2, 0.6 * fade_alpha)  # Yellow
        glVertex3f(*start)
        glColor4f(1.0, 0.5, 0.0, 0.4 * fade_alpha)  # Orange-red
        glVertex3f(*end)
        glEnd()
        
        # Faint outer glow - removed the outermost glow for a cleaner look
        glBegin(GL_LINES)
        glColor4f(1.0, 0.8, 0.5, 0.3 * fade_alpha)  # Soft orange
        glVertex3f(*start)
        glColor4f(1.0, 0.4, 0.2, 0.2 * fade_alpha)  # Soft red
        glVertex3f(*end)
        glEnd()
        
        # Draw small circles at both ends for impact effect
        def draw_circle(center, radius, color):
            glBegin(GL_TRIANGLE_FAN)
            glColor4f(*color, 0.5 * fade_alpha)
            glVertex3f(*center)
            for i in range(17):  # 16 segments
                angle = 2.0 * math.pi * i / 16.0
                dx = math.cos(angle) * radius
                dz = math.sin(angle) * radius
                glColor4f(*color, 0.0)  # Fade to transparent at edges
                glVertex3f(center[0] + dx, center[1], center[2] + dz)
            glEnd()
        
        # Draw impact circles at both ends
        impact_radius = 0.1 * (1.0 - progress)  # Shrink over time
        draw_circle(start, impact_radius, (1.0, 1.0, 1.0))  # White at start
        draw_circle(end, impact_radius, (1.0, 0.5, 0.0))    # Orange at end
        
        # Draw impact effect if this is a hit and we're at the end of the trail
        if progress >= 0.95 and 'hit_position' in bullet and 'impact_effect' in bullet and bullet['impact_effect']:
            glPointSize(8.0)
            glBegin(GL_POINTS)
            glColor4f(1.0, 0.5, 0.2, 0.8)  # Orange impact point for objects
            glVertex3f(*bullet['hit_position'])
            glEnd()
        
        glDepthMask(True)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
    
    def draw_circular_shadow(self, center_x, center_z, radius, alpha=0.3, segments=32):
        """Draw a circular shadow on the ground plane with light angle consideration.
        
        Args:
            center_x, center_z: X and Z coordinates of the shadow center
            radius: Base radius of the shadow circle
            alpha: Opacity of the shadow (0.0 to 1.0)
            segments: Number of segments to use for the circle
        """
        # Save current state including depth buffer
        glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Disable lighting and texturing for the shadow
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Disable depth testing to prevent z-fighting
        # glDisable(GL_DEPTH_TEST)
        
        # Fixed light direction (x, y, z) - coming from top-right
        light_dir = [1.0, -1.0, 0.5]
        length = math.sqrt(sum(x*x for x in light_dir))
        light_dir = [x/length for x in light_dir]  # Normalize
        
        # Calculate shadow offset based on light direction
        shadow_offset_x = -light_dir[0] * 0.1  # Scale factor for offset amount
        shadow_offset_z = -light_dir[2] * 0.1
        
        # Calculate stretched radius based on light angle
        angle_factor = max(0.2, 1.0 - abs(light_dir[1]) * 0.8)  # How much to stretch the shadow
        radius_x = radius * (1.0 + (1.0 - angle_factor) * 0.5)
        radius_z = radius * (1.0 + (1.0 - angle_factor) * 0.5)
        
        # Offset the shadow center based on light direction
        shadow_center_x = center_x + shadow_offset_x
        shadow_center_z = center_z + shadow_offset_z
        
        # Draw a smooth circle with gradient
        glBegin(GL_TRIANGLE_FAN)
        
        # Center point (slightly more transparent)
        glColor4f(0.0, 0.0, 0.0, alpha * 0.3)
        glVertex3f(shadow_center_x, self.physics.grid_height + 0.05, shadow_center_z)
        
        # Edge points (fade out)
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            x = shadow_center_x + radius_x * math.cos(angle)
            z = shadow_center_z + radius_z * math.sin(angle)
            # Fade out towards the edges
            edge_alpha = alpha * 0.7
            glColor4f(0.0, 0.0, 0.0, edge_alpha)
            glVertex3f(x, self.physics.grid_height + 0.05, z)
        
        glEnd()
        
        # Draw a darker core shadow
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(0.0, 0.0, 0.0, alpha * 0.4)
        glVertex3f(shadow_center_x, self.physics.grid_height + 0.06, shadow_center_z)
        
        inner_radius = radius * 0.3
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            x = shadow_center_x + inner_radius * math.cos(angle)
            z = shadow_center_z + inner_radius * math.sin(angle)
            glVertex3f(x, self.physics.grid_height + 0.06, z)
        
        glEnd()

        # Restore state
        glEnable(GL_DEPTH_TEST)
        glPopAttrib()
    
    def draw_rectangular_shadow(self, center_x, center_z, width, depth, alpha=0.25):
        """Draw a rectangular shadow on the ground plane with light angle consideration.
        
        Args:
            center_x, center_z: X and Z coordinates of the shadow center
            width: Width of the shadow (x-axis)
            depth: Depth of the shadow (z-axis)
            alpha: Opacity of the shadow (0.0 to 1.0)
        """
        # Save current state including depth buffer
        glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Disable lighting and texturing for the shadow
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable depth testing but don't write to depth buffer
        glEnable(GL_DEPTH_TEST)
        glDepthMask(False)  # Don't write to depth buffer
        glDepthFunc(GL_LEQUAL)  # Draw shadow if depth is less than or equal to existing depth
        
        # Fixed light direction (x, y, z) - coming from top-right
        light_dir = [1.0, -1.0, 0.5]
        length = math.sqrt(sum(x*x for x in light_dir))
        light_dir = [x/length for x in light_dir]  # Normalize
        
        # Calculate shadow offset based on light direction
        shadow_offset_x = -light_dir[0] * 0.1  # Scale factor for offset amount
        shadow_offset_z = -light_dir[2] * 0.1
        
        # Offset the shadow center based on light direction
        shadow_center_x = center_x + shadow_offset_x
        shadow_center_z = center_z + shadow_offset_z
        
        # Calculate half dimensions
        half_width = width * 0.5
        half_depth = depth * 0.5
        
        # Draw the main shadow quad with gradient
        glBegin(GL_QUADS)
        
        # Calculate corner points with slight offset for shadow softness
        softness = 0.15  # How much the shadow extends beyond the object
        x1 = shadow_center_x - half_width * (1 + softness)
        x2 = shadow_center_x + half_width * (1 + softness)
        z1 = shadow_center_z - half_depth * (1 + softness)
        z2 = shadow_center_z + half_depth * (1 + softness)
        
        # Inner rectangle (darker)
        ix1 = shadow_center_x - half_width * 0.9
        ix2 = shadow_center_x + half_width * 0.9
        iz1 = shadow_center_z - half_depth * 0.9
        iz2 = shadow_center_z + half_depth * 0.9
        
        # Position shadow slightly above ground to avoid z-fighting
        shadow_height = self.physics.grid_height + 0.01
        
        # Draw outer shadow (softer)
        glColor4f(0.0, 0.0, 0.0, 0.0)
        glVertex3f(x1, shadow_height, z1)
        glVertex3f(x2, shadow_height, z1)
        glColor4f(0.0, 0.0, 0.0, alpha * 0.3)
        glVertex3f(ix2, shadow_height, iz1)
        glVertex3f(ix1, shadow_height, iz1)
        
        glColor4f(0.0, 0.0, 0.0, 0.0)
        glVertex3f(x2, shadow_height, z1)
        glVertex3f(x2, shadow_height, z2)
        glColor4f(0.0, 0.0, 0.0, alpha * 0.3)
        glVertex3f(ix2, shadow_height, iz2)
        glVertex3f(ix2, shadow_height, iz1)
        
        glColor4f(0.0, 0.0, 0.0, 0.0)
        glVertex3f(x2, shadow_height, z2)
        glVertex3f(x1, shadow_height, z2)
        glColor4f(0.0, 0.0, 0.0, alpha * 0.3)
        glVertex3f(ix1, shadow_height, iz2)
        glVertex3f(ix2, shadow_height, iz2)
        
        glColor4f(0.0, 0.0, 0.0, 0.0)
        glVertex3f(x1, shadow_height, z2)
        glVertex3f(x1, shadow_height, z1)
        glColor4f(0.0, 0.0, 0.0, alpha * 0.3)
        glVertex3f(ix1, shadow_height, iz1)
        glVertex3f(ix1, shadow_height, iz2)
        
        # Draw inner shadow (darker)
        glColor4f(0.0, 0.0, 0.0, alpha * 0.5)
        glVertex3f(ix1, shadow_height + 0.01, iz1)
        glVertex3f(ix2, shadow_height + 0.01, iz1)
        glVertex3f(ix2, shadow_height + 0.01, iz2)
        glVertex3f(ix1, shadow_height + 0.01, iz2)
        
        glEnd()
        
        # Restore state
        glDepthMask(True)  # Re-enable writing to depth buffer
        glEnable(GL_LIGHTING)
        glPopAttrib()

    def draw_oval_shadow(self, center_x, center_z, width, height, alpha=0.4, segments=32):
        """Draw an oval shadow on the ground plane.
        
        Args:
            center_x, center_z: Center position of the shadow
            width: Width of the oval (x-axis)
            height: Height of the oval (z-axis)
            alpha: Opacity of the shadow (0.0 to 1.0)
            segments: Number of segments to use for the oval
        """
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(False)
        
        # Increased y-offset to prevent z-fighting with ground texture
        shadow_y = self.physics.grid_height + 0.05  # Increased from 0.01 to 0.05
        
        # Set shadow color (dark with alpha)
        glColor4f(0.1, 0.1, 0.1, alpha)
        
        # Draw filled oval using triangle fan
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(center_x, shadow_y, center_z)  # Center point with increased y
        
        # Add points around the oval with increased size (1.2x)
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            x = center_x + math.cos(angle) * width * 0.6  # Increased from 0.5 to 0.6 (20% larger)
            z = center_z + math.sin(angle) * height * 0.6  # Increased from 0.5 to 0.6 (20% larger)
            glVertex3f(x, shadow_y, z)
        
        glEnd()
        
        # Draw outline for better definition
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glColor4f(0.05, 0.05, 0.05, alpha * 0.8)
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            x = center_x + math.cos(angle) * width * 0.6  # Match the increased size
            z = center_z + math.sin(angle) * height * 0.6  # Match the increased size
            glVertex3f(x, shadow_y + 0.01, z)  # Slightly higher than the fill to prevent z-fighting
        glEnd()
        
        # Restore OpenGL state
        glDepthMask(True)
        glEnable(GL_LIGHTING)
        glDisable(GL_BLEND)
    
    def draw_player(self):
        # Draw player shadow first (under the player)
        # Using an oval shadow that's slightly elongated in the direction the player is facing
        angle = math.radians(self.player.rotation_y)
        # Increased size from 0.7,0.5 to 0.9,0.7 for better coverage
        self.draw_oval_shadow(self.player.x, self.player.z, 0.9, 0.7, alpha=0.25)

        
        # Draw the player model
        glPushMatrix()
        # Position the model at the player's position
        # The player's y position is at the base of the model (feet position)
        glTranslatef(self.player.x, self.player.y, self.player.z)
        
        # Apply yaw rotation to make model follow left/right camera movement
        # with an additional 45-degree left rotation
        glRotatef(self.player.rotation_y + 60+ self.currPlayerModel.metadata['Yaw_offset'], 0, 1, 0)  # Yaw with 45-degree left offset
        
        # Apply model's own rotation (roll)
        glRotatef(self.player.rotation_z, 0, 0, 1)  # Roll (tilt) around Z axis
        
        # Get model's metadata
        model_scale = 1.0
        model_offset = (0, 0, 0)
        
        if hasattr(self.currPlayerModel, 'metadata'):
            # Apply model scale from metadata if available
            if 'scale' in self.currPlayerModel.metadata:
                model_scale = self.currPlayerModel.metadata['scale']
            
            # Get model offset from metadata if available
            if 'offset' in self.currPlayerModel.metadata:
                model_offset = self.currPlayerModel.metadata['offset']
        
        # Apply model scale
        glScalef(model_scale, model_scale, model_scale)
        
        # Apply model offset (after scaling, before rotation)
        glTranslatef(model_offset[0], model_offset[1], model_offset[2])

        
        # Fixed model orientation (Y forward, X right, Z up)
        glRotatef(-90, 1, 0, 0)  # Rotate -90 degrees around X to make Y point forward
        glRotatef(90, 1, 0, 0)   # Additional 90 degrees around X (red axis)
        
        # Apply an additional offset to account for model's center
        # This will position the model 
        glTranslatef(0, -1.5, 0)  # Base of model 
        
        # Draw the model with its own scaling and centering
        glColor3f(0.3, 0.3, 0.3)  # Dark gray color for the model
        self.currPlayerModel.render()
        
        glPopMatrix()
    
    
    def draw_lock_button(self):
        # Define button sizes
        button_size = 40
        margin = 10
        
        # Position lock button at bottom of the right margin
        self.lock_button_rect.y = self.display[1] - button_size - margin
        
        # Switch to orthographic projection for 2D UI elements
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth testing for 2D elements
        glDisable(GL_DEPTH_TEST)
        
        # Check if mouse is hovering over the lock button
        mouse_x, mouse_y = pygame.mouse.get_pos()
        is_hovered = self.lock_button_rect.collidepoint(mouse_x, mouse_y)
        
        # Draw lock button background (always use the same color)
        if is_hovered:
            button_color = self.colors['lock_button_hover']
        else:
            button_color = self.colors['lock_button']
            
        glColor3f(*button_color)
        glBegin(GL_QUADS)
        glVertex2f(self.lock_button_rect.left, self.lock_button_rect.top)
        glVertex2f(self.lock_button_rect.right, self.lock_button_rect.top)
        glVertex2f(self.lock_button_rect.right, self.lock_button_rect.bottom)
        glVertex2f(self.lock_button_rect.left, self.lock_button_rect.bottom)
        glEnd()
        
        # Draw lock icon - change color based on lock state
        if self.mouse_grabbed:  # Locked state - dark icon
            icon_color = (0.3, 0.3, 0.3)  # Dark gray
        else:  # Unlocked state - light icon
            icon_color = (0.9, 0.9, 0.9)  # Very light gray
            
        glColor3f(*icon_color)
        glLineWidth(2.0)
        
        # Draw lock body
        lock_width = button_size // 2
        lock_height = button_size // 3
        lock_x = self.lock_button_rect.centerx - lock_width // 2
        lock_y = self.lock_button_rect.centery - lock_height // 3
        
        glBegin(GL_LINE_LOOP)
        glVertex2f(lock_x, lock_y)
        glVertex2f(lock_x + lock_width, lock_y)
        glVertex2f(lock_x + lock_width, lock_y + lock_height)
        glVertex2f(lock_x, lock_y + lock_height)
        glEnd()
        
        # Draw lock top
        glBegin(GL_LINE_STRIP)
        glVertex2f(lock_x + lock_width // 4, lock_y)
        glVertex2f(lock_x + lock_width // 4, lock_y - lock_height // 2)
        glVertex2f(lock_x + 3 * lock_width // 4, lock_y - lock_height // 2)
        glVertex2f(lock_x + 3 * lock_width // 4, lock_y)
        glEnd()
        
        # Re-enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Restore the projection and modelview matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def draw_reset_button(self):
        button_size = 40
        margin = 10

        # Position reset button above the lock button
        self.reset_button_rect.x = self.display[0] - button_size - margin
        self.reset_button_rect.y = self.display[1] - 2 * button_size - 2 * margin

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)

        mouse_x, mouse_y = pygame.mouse.get_pos()
        is_hovered = self.reset_button_rect.collidepoint(mouse_x, mouse_y)

        button_color = self.colors['lock_button_hover'] if is_hovered else self.colors['lock_button']

        glColor3f(*button_color)
        glBegin(GL_QUADS)
        glVertex2f(self.reset_button_rect.left, self.reset_button_rect.top)
        glVertex2f(self.reset_button_rect.right, self.reset_button_rect.top)
        glVertex2f(self.reset_button_rect.right, self.reset_button_rect.bottom)
        glVertex2f(self.reset_button_rect.left, self.reset_button_rect.bottom)
        glEnd()

        icon_color = (0.9, 0.9, 0.9) if is_hovered else (0.3, 0.3, 0.3)
        glColor3f(*icon_color)
        glLineWidth(2.0)

        center_x = self.reset_button_rect.centerx
        center_y = self.reset_button_rect.centery
        radius = button_size * 0.28

        glBegin(GL_LINE_STRIP)
        segments = 24
        start_angle = math.radians(60)
        end_angle = math.radians(330)
        angle_step = (end_angle - start_angle) / segments
        for i in range(segments + 1):
            angle = start_angle + i * angle_step
            glVertex2f(center_x + math.cos(angle) * radius, center_y + math.sin(angle) * radius)
        glEnd()

        arrow_length = button_size * 0.18
        arrow_angle = math.radians(320)
        arrow_base_x = center_x + math.cos(end_angle) * radius
        arrow_base_y = center_y + math.sin(end_angle) * radius

        left_angle = arrow_angle + math.radians(150)
        right_angle = arrow_angle - math.radians(150)

        glBegin(GL_LINES)
        glVertex2f(arrow_base_x, arrow_base_y)
        glVertex2f(arrow_base_x + math.cos(left_angle) * arrow_length,
                   arrow_base_y + math.sin(left_angle) * arrow_length)
        glVertex2f(arrow_base_x, arrow_base_y)
        glVertex2f(arrow_base_x + math.cos(right_angle) * arrow_length,
                   arrow_base_y + math.sin(right_angle) * arrow_length)
        glEnd()

        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def draw_menu_button(self):
        # Define button and panel sizes
        button_size = 40
        margin = 10
        button_bg_width = 60  # Width of the vertical strip for the button
        panel_width = 250  # Panel will be this wide
        
        # Position buttons on right side with margin
        self.menu_button_rect.x = self.display[0] - button_size - margin
        self.panel2_button_rect.x = self.menu_button_rect.x
        self.panel2_button_rect.y = self.menu_button_rect.bottom + margin
        self.panel2_button_rect.width = button_size
        self.panel2_button_rect.height = button_size
        
        # Switch to orthographic projection for 2D UI elements
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth testing for 2D elements
        glDisable(GL_DEPTH_TEST)
        
        # Get mouse position for hover detection
        mouse_x, mouse_y = pygame.mouse.get_pos()
        is_menu_hovered = self.menu_button_rect.collidepoint(mouse_x, mouse_y)
        is_panel2_hovered = self.panel2_button_rect.collidepoint(mouse_x, mouse_y)
        
        # Draw vertical background for buttons (narrow strip on the right)
        glColor3f(0.1, 0.1, 0.15)  # Panel color
        glBegin(GL_QUADS)
        glVertex2f(self.display[0] - button_bg_width, 0)
        glVertex2f(self.display[0], 0)
        glVertex2f(self.display[0], self.display[1])
        glVertex2f(self.display[0] - button_bg_width, self.display[1])
        glEnd()
        
        # Draw menu button background (plus button)
        menu_button_color = self.colors['menu_button_hover'] if is_menu_hovered else self.colors['menu_button']
        glColor3f(*menu_button_color)
        glBegin(GL_QUADS)
        glVertex2f(self.menu_button_rect.left, self.menu_button_rect.top)
        glVertex2f(self.menu_button_rect.right, self.menu_button_rect.top)
        glVertex2f(self.menu_button_rect.right, self.menu_button_rect.bottom)
        glVertex2f(self.menu_button_rect.left, self.menu_button_rect.bottom)
        
        # Draw second panel button background (X button) - different color
        panel2_button_color = (0.3, 0.2, 0.25) if is_panel2_hovered else (0.2, 0.15, 0.2)
        glColor3f(*panel2_button_color)
        glVertex2f(self.panel2_button_rect.left, self.panel2_button_rect.top)
        glVertex2f(self.panel2_button_rect.right, self.panel2_button_rect.top)
        glVertex2f(self.panel2_button_rect.right, self.panel2_button_rect.bottom)
        glVertex2f(self.panel2_button_rect.left, self.panel2_button_rect.bottom)
        glEnd()
        
        # Draw menu icon (plus)
        glColor3f(1.0, 1.0, 1.0)
        bar_thickness = 4
        bar_length = 18

        # Draw plus sign for menu button
        horizontal_top = self.menu_button_rect.centery - bar_thickness // 2
        vertical_left = self.menu_button_rect.centerx - bar_thickness // 2

        # Draw plus sign (always show both lines for plus)
        glBegin(GL_QUADS)
        # Horizontal line (always visible)
        glVertex2f(self.menu_button_rect.centerx - bar_length // 2, horizontal_top)
        glVertex2f(self.menu_button_rect.centerx + bar_length // 2, horizontal_top)
        glVertex2f(self.menu_button_rect.centerx + bar_length // 2, horizontal_top + bar_thickness)
        glVertex2f(self.menu_button_rect.centerx - bar_length // 2, horizontal_top + bar_thickness)
        
        # Vertical line (always visible)
        glVertex2f(vertical_left, self.menu_button_rect.centery - bar_length // 2)
        glVertex2f(vertical_left + bar_thickness, self.menu_button_rect.centery - bar_length // 2)
        glVertex2f(vertical_left + bar_thickness, self.menu_button_rect.centery + bar_length // 2)
        glVertex2f(vertical_left, self.menu_button_rect.centery + bar_length // 2)
        glEnd()
        
        # Draw cursive X for second panel button
        x_size = 16
        x_thickness = 3
        x_center = self.panel2_button_rect.centerx
        y_center = self.panel2_button_rect.centery
        
        # First stroke of X (top-left to bottom-right, with curve)
        points1 = [
            (x_center - x_size//2, y_center - x_size//4),
            (x_center - x_size//4, y_center - x_size//2),
            (x_center + x_size//4, y_center + x_size//2),
            (x_center + x_size//2, y_center + x_size//4)
        ]
        
        # Second stroke of X (top-right to bottom-left, with curve)
        points2 = [
            (x_center + x_size//2, y_center - x_size//4),
            (x_center + x_size//4, y_center - x_size//2),
            (x_center - x_size//4, y_center + x_size//2),
            (x_center - x_size//2, y_center + x_size//4)
        ]
        
        # Draw the cursive X using thick lines
        glLineWidth(x_thickness)
        glBegin(GL_LINE_STRIP)
        for x, y in points1:
            glVertex2f(x, y)
        glEnd()
        
        glBegin(GL_LINE_STRIP)
        for x, y in points2:
            glVertex2f(x, y)
        glEnd()
        
        # Reset line width for other drawing
        glLineWidth(1.0)
        
        # Re-enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Restore the projection and modelview matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def draw_side_panel(self):
        if not self.show_side_panel:
            return
            
        # Switch to orthographic projection for 2D UI elements
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth testing for 2D elements
        glDisable(GL_DEPTH_TEST)
        
        # Draw side panel background
        button_bg_width = 60  # Should match the width in draw_menu_button
        panel_width = 250
        panel_x = self.display[0] - panel_width - button_bg_width  # Position panel to the left of button background
        
        glColor3f(0.1, 0.1, 0.15)  # Panel color
        glBegin(GL_QUADS)
        glVertex2f(panel_x, 0)
        glVertex2f(panel_x + panel_width, 0)
        glVertex2f(panel_x + panel_width, self.display[1])
        glVertex2f(panel_x, self.display[1])
        glEnd()

        # Draw structure buttons when mouse is free
        rendered_buttons = []
        if not self.mouse_grabbed:
            rendered_buttons = self.draw_structure_buttons(panel_x)
        self.structure_buttons = rendered_buttons
        
        # Re-enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Restore the projection and modelview matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def draw_text(self, text, x, y, size, color):
        """Draw text on the screen using Pygame's font renderer"""
        font = pygame.font.Font(None, size)
        text_surface = font.render(text, True, (int(color[0]*255), int(color[1]*255), int(color[2]*255)))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        
        # Save current OpenGL state
        glPushAttrib(GL_ENABLE_BIT)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Set up orthographic projection for 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth testing for text
        glDisable(GL_DEPTH_TEST)
        
        # Draw the text
        glRasterPos2i(x, y + size)  # Adjust y position to align with Pygame's text rendering
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), 
                    GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        # Restore OpenGL state
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()
    
    def draw_side_panel2(self):
        if not hasattr(self, 'show_side_panel2') or not self.show_side_panel2:
            return
            
        # Switch to orthographic projection for 2D UI elements
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth testing for 2D elements
        glDisable(GL_DEPTH_TEST)
        
        # Draw side panel background (slightly different color to distinguish from first panel)
        button_bg_width = 60
        panel_width = 250
        panel_x = self.display[0] - panel_width - button_bg_width
        
        # Panel background
        glColor3f(0.1, 0.1, 0.18)  # Slightly more blue tint
        glBegin(GL_QUADS)
        glVertex2f(panel_x, 0)
        glVertex2f(panel_x + panel_width, 0)
        glVertex2f(panel_x + panel_width, self.display[1])
        glVertex2f(panel_x, self.display[1])
        glEnd()
        
        # Panel title
        title_x = panel_x + 15
        title_y = 20
        self.draw_text("variable Settings", title_x, title_y, 20, (0.9, 0.9, 0.9))
        
        # Bullet Impulse section
        section_y = title_y + 40
        self.draw_text("Bullet Impulse", panel_x + 15, section_y, 16, (0.9, 0.9, 0.9))
        
        # Input field background
        input_y = section_y + 25
        input_rect = pygame.Rect(panel_x + 15, input_y, panel_width - 30, 30)
        
        # Check if input is active
        mouse_x, mouse_y = pygame.mouse.get_pos()
        is_hovered = input_rect.collidepoint(mouse_x, mouse_y)
        
        # Set input background color
        if not hasattr(self, 'bullet_impulse_active'):
            self.bullet_impulse_active = False
        if not hasattr(self, 'bullet_impulse'):
            self.bullet_impulse = 50.0  # Default value
            
        if self.bullet_impulse_active:
            glColor3f(0.15, 0.15, 0.25)  # Active state
        elif is_hovered:
            glColor3f(0.18, 0.18, 0.28)  # Hover state
        else:
            glColor3f(0.12, 0.12, 0.22)  # Default state
            
        glBegin(GL_QUADS)
        glVertex2f(input_rect.left, input_rect.top)
        glVertex2f(input_rect.right, input_rect.top)
        glVertex2f(input_rect.right, input_rect.bottom)
        glVertex2f(input_rect.left, input_rect.bottom)
        glEnd()
        
        # Draw input border
        border_color = (0.3, 0.5, 0.8) if self.bullet_impulse_active else (0.2, 0.2, 0.3)
        glColor3f(*border_color)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(input_rect.left, input_rect.top)
        glVertex2f(input_rect.right, input_rect.top)
        glVertex2f(input_rect.right, input_rect.bottom)
        glVertex2f(input_rect.left, input_rect.bottom)
        glEnd()
        
        # Draw input text with better cursor positioning
        text_x = input_rect.left + 8
        text_y = input_rect.centery - 8
        
        # Get the display text from bullet_impulse_text if it exists, otherwise format the float
        if hasattr(self, 'bullet_impulse_text'):
            display_text = self.bullet_impulse_text
            if not display_text:  # If empty, show 0
                display_text = '0'
        else:
            # Fallback to float formatting if text buffer doesn't exist
            if self.bullet_impulse == int(self.bullet_impulse):
                display_text = str(int(self.bullet_impulse))
            else:
                display_text = f"{self.bullet_impulse:.1f}"
        
        # Draw the text
        self.draw_text(display_text, text_x, text_y, 16, (1.0, 1.0, 1.0))
        
        # Draw cursor if active
        if self.bullet_impulse_active and int(time.time() * 2) % 2 == 0:
            # Calculate cursor position based on actual text width
            font = pygame.font.Font(None, 16)
            text_surface = font.render(display_text, True, (255, 255, 255))
            cursor_x = text_x + text_surface.get_width() + 1  # Add 1px padding after text
            
            # Draw a vertical cursor line at the end of the text
            glColor3f(1.0, 1.0, 1.0)
            glLineWidth(2.0)
            glBegin(GL_LINES)
            glVertex2f(cursor_x, text_y + 2)
            glVertex2f(cursor_x, text_y + 14)
            glEnd()
        
        # Store input rect for click detection
        self.bullet_impulse_rect = input_rect
        
        # Re-enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def draw_structure_buttons(self, panel_x):
        panel_width = 250
        margin = 15
        button_size = 60
        spacing = 30
        button_x = panel_x + margin
        button_y = 80

        mouse_x, mouse_y = pygame.mouse.get_pos()
        font = pygame.font.Font(None, 18)
        buttons = []

        for preset in self.structure_presets:
            rect = pygame.Rect(button_x, button_y, button_size, button_size)
            is_hovered = rect.collidepoint(mouse_x, mouse_y)

            is_active = self.active_structure and preset['name'] == self.active_structure['name']
            button_color = self.colors['cube_button_hover'] if (is_hovered or is_active) else self.colors['cube_button']

            glColor3f(*button_color)
            glBegin(GL_QUADS)
            glVertex2f(rect.left, rect.top)
            glVertex2f(rect.right, rect.top)
            glVertex2f(rect.right, rect.bottom)
            glVertex2f(rect.left, rect.bottom)
            glEnd()

            image_surface = self.get_structure_thumbnail(preset)
            scaled_surface = None
            if image_surface:
                cache = preset.setdefault('_thumbnail_cache', {})
                if rect.size not in cache:
                    cache[rect.size] = pygame.transform.smoothscale(image_surface, rect.size)
                scaled_surface = cache[rect.size]
                image_data = pygame.image.tostring(scaled_surface, 'RGBA', True)

                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glRasterPos2i(rect.left, rect.top + rect.height)
                glDrawPixels(rect.width, rect.height, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
                glDisable(GL_BLEND)

            if is_active and self.placing_cube:
                label_text = "Click to place"
            elif is_active:
                label_text = f"{preset['name']} (active)"
            else:
                label_text = preset['name']

            text_surface = font.render(label_text, True, (255, 255, 255))
            text_data = pygame.image.tostring(text_surface, 'RGBA', True)

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            text_x = rect.left
            text_y = rect.bottom + 8 + text_surface.get_height()
            glRasterPos2i(text_x, text_y)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            glDisable(GL_BLEND)

            text_height = text_surface.get_height()
            total_height = button_size + text_height + 16
            expanded_rect = pygame.Rect(rect.left, rect.top, rect.width, total_height)
            buttons.append({'preset': preset, 'rect': expanded_rect})

            button_y += total_height + spacing

        return buttons

    def draw_cube_preview(self):
        if self.mouse_grabbed or not self.placing_cube:
            return

        hit_point = self.get_cursor_world_hit()
        if hit_point is None:
            return

        x, y, z = hit_point
        y_offset = self.get_structure_half_height()
        y = max(y, self.physics.grid_height) + y_offset
        self.cube_preview_pos = [x, y, z]

        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glPushMatrix()
        glTranslatef(*self.cube_preview_pos)

        half_sizes = [dim / 2.0 for dim in self.active_total_size]
        hx, hy, hz = half_sizes
        glColor4f(*self.colors['cube_preview'])
        glBegin(GL_QUADS)
        # Front face
        glVertex3f(-hx, -hy, hz)
        glVertex3f(hx, -hy, hz)
        glVertex3f(hx, hy, hz)
        glVertex3f(-hx, hy, hz)
        # Back face
        glVertex3f(-hx, -hy, -hz)
        glVertex3f(-hx, hy, -hz)
        glVertex3f(hx, hy, -hz)
        glVertex3f(hx, -hy, -hz)
        # Top face
        glVertex3f(-hx, hy, -hz)
        glVertex3f(-hx, hy, hz)
        glVertex3f(hx, hy, hz)
        glVertex3f(hx, hy, -hz)
        # Bottom face
        glVertex3f(-hx, -hy, -hz)
        glVertex3f(hx, -hy, -hz)
        glVertex3f(hx, -hy, hz)
        glVertex3f(-hx, -hy, hz)
        # Right face
        glVertex3f(hx, -hy, -hz)
        glVertex3f(hx, hy, -hz)
        glVertex3f(hx, hy, hz)
        glVertex3f(hx, -hy, hz)
        # Left face
        glVertex3f(-hx, -hy, -hz)
        glVertex3f(-hx, -hy, hz)
        glVertex3f(-hx, hy, hz)
        glVertex3f(-hx, hy, -hz)
        glEnd()

        glPopMatrix()
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def get_structure_thumbnail(self, preset):
        if 'thumbnail_surface' in preset:
            return preset['thumbnail_surface']

        thumb_path = preset.get('thumbnail')
        if not thumb_path:
            preset['thumbnail_surface'] = None
            return None

        if not os.path.isabs(thumb_path):
            thumb_path = os.path.join(os.path.dirname(__file__), thumb_path)

        if os.path.exists(thumb_path):
            preset['thumbnail_surface'] = pygame.image.load(thumb_path).convert_alpha()
        else:
            preset['thumbnail_surface'] = None

        return preset['thumbnail_surface']

    def load_audio_from_model(self):
        if not self.audio_enabled or not hasattr(self.currPlayerModel, 'metadata'):
            self.shoot_sound = None
            self.hit_sound = None
            return

        metadata = self.currPlayerModel.metadata
        
        # Load shoot sound (can be a single sound or a list)
        shoot_paths = metadata.get('shoot_sound')
        if shoot_paths:
            if isinstance(shoot_paths, str):
                shoot_paths = [shoot_paths]
            self.shoot_sound = []
            for path in shoot_paths:
                sound_file = self.resolve_audio_path(path)
                if sound_file:
                    sound = pygame.mixer.Sound(sound_file)
                    sound.set_volume(0.8)
                    self.shoot_sound.append(sound)
            if not self.shoot_sound:
                self.shoot_sound = None
            self.shoot_channel = pygame.mixer.Channel(1)
        else:
            self.shoot_sound = None

        # Load default hit sounds (can be a single sound or a list)
        hit_paths = metadata.get('hit_sound')
        if hit_paths:
            if isinstance(hit_paths, str):
                hit_paths = [hit_paths]
            self.hit_sound = hit_paths  # Store as list of paths for later resolution
            self.hit_channel = pygame.mixer.Channel(2)
        else:
            self.hit_sound = None

    def resolve_audio_path(self, path_value):
        if not path_value:
            return None
        
        candidates = []
        src_dir = os.path.dirname(__file__)

        # Try the path as-is first
        if os.path.isabs(path_value):
            candidates.append(path_value)
        else:
            # Try relative to source directory
            rel_path = os.path.join(src_dir, path_value)
            candidates.append(rel_path)
            
            # Try in audio subdirectory
            audio_path = os.path.join(src_dir, 'audio', path_value)
            candidates.append(audio_path)

        # Try with different filename variations
        basename = os.path.basename(path_value)
        if '-' in basename or '_' in basename:
            # Try with hyphens/underscores replaced
            alt_basename = basename.replace('-', '_') if '-' in basename else basename.replace('_', '-')
            alt_path = os.path.join(os.path.dirname(path_value), alt_basename)
            
            if os.path.isabs(path_value):
                candidates.append(alt_path)
            else:
                # Try relative to source directory
                rel_alt_path = os.path.join(src_dir, alt_path)
                candidates.append(rel_alt_path)
                
                # Try in audio subdirectory
                audio_alt_path = os.path.join(src_dir, 'audio', alt_basename)
                candidates.append(audio_alt_path)

        # Check each candidate
        for i, candidate in enumerate(candidates):
            if candidate and os.path.exists(candidate):
                return os.path.normpath(candidate)
            
            # Try with different case (for case-insensitive filesystems)
            if candidate and os.path.exists(candidate.lower()):
                return os.path.normpath(candidate.lower())
                
            # Try with .wav extension if not present
            base, ext = os.path.splitext(candidate)
            if ext.lower() != '.wav':
                wav_candidate = base + '.wav'
                if os.path.exists(wav_candidate):
                    return os.path.normpath(wav_candidate)
                
            # Try with .mp3 extension if not present
            if ext.lower() != '.mp3':
                mp3_candidate = base + '.mp3'
                if os.path.exists(mp3_candidate):
                    return os.path.normpath(mp3_candidate)

        return None

    def start_shoot_sound(self):
        if not self.audio_enabled or not self.shoot_sound:
            return
            
        # If shoot_sound is a list, play a random one, otherwise play the single sound
        if isinstance(self.shoot_sound, list):
            if not self.shoot_sound:  # Empty list
                return
            sound_to_play = random.choice(self.shoot_sound)
        else:
            sound_to_play = self.shoot_sound
            
        if self.shoot_channel is None:
            self.shoot_channel = pygame.mixer.Channel(1)
            
        if not self.shoot_channel.get_busy():
            self.shoot_channel.play(sound_to_play, loops=-1)

    def stop_shoot_sound(self):
        if self.shoot_channel and self.shoot_channel.get_busy():
            self.shoot_channel.stop()

    def play_random_sound(self, sound_paths, volume=0.8, channel_num=2):
        """Play a random sound from a list of sound files."""
        
        if not self.audio_enabled:
            return
            
        if not sound_paths:
            return
            
        # If a single sound path is provided, convert it to a list
        if isinstance(sound_paths, str):
            sound_paths = [sound_paths]
            
        if not sound_paths:
            return
            
        # Select a random sound
        sound_path = random.choice(sound_paths)
        
        # Resolve the full path to the sound file
        sound_file = self.resolve_audio_path(sound_path)
        if not sound_file:
            return
            
        try:
            # Try to load the sound
            sound = pygame.mixer.Sound(sound_file)
            
            # Set volume
            sound.set_volume(volume)
            
            # Get or create channel
            channel = pygame.mixer.Channel(channel_num)
            
            # Play the sound
            channel.play(sound)
            
            # Store the channel if needed for future reference
            if channel_num == 2:  # Default hit sound channel
                self.hit_channel = channel
                
            
        except Exception as e:
            import traceback
            traceback.print_exc()
                
    def play_hit_sound(self, structure_id=None):
        """Play a hit sound, optionally using a structure's specific sound."""
        
        if not self.audio_enabled:
            return
            
        # Try to get structure-specific sound if structure_id is provided
        if structure_id is not None:
            if structure_id in self.physics.structures:
                structure = self.physics.structures[structure_id]
                
                # Check for hit sound in metadata
                if 'metadata' in structure:
                    if 'hit_sound' in structure['metadata']:
                        sound_list = structure['metadata']['hit_sound']
                        if isinstance(sound_list, str):
                            sound_list = [sound_list]
                        if sound_list:  # Only play if we have sounds
                            self.play_random_sound(sound_list, volume=0.7)
                            return
                    else:
                        return
                else:
                    return
            else:
                return
        else:
            return
                
        # Fall back to default hit sound if no structure-specific sound was found
        if hasattr(self, 'hit_sound') and self.hit_sound:
            if isinstance(self.hit_sound, str):
                sound_list = [self.hit_sound]
            else:
                sound_list = self.hit_sound
            if sound_list:  # Only play if we have sounds
                self.play_random_sound(sound_list, volume=0.8)
        else:
            return

    def get_cursor_world_hit(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)

        near_point = gluUnProject(mouse_x, viewport[3] - mouse_y, 0.0, modelview, projection, viewport)
        far_point = gluUnProject(mouse_x, viewport[3] - mouse_y, 1.0, modelview, projection, viewport)

        if near_point is None or far_point is None:
            return None

        ray_direction = [f - n for f, n in zip(far_point, near_point)]
        length = math.sqrt(sum(d * d for d in ray_direction))
        if length == 0:
            return None
        ray_direction = [d / length for d in ray_direction]

        ray_end = [near_point[i] + ray_direction[i] * 100.0 for i in range(3)]
        hit_object, hit_point = self.physics.ray_test(list(near_point), ray_end)

        if hit_point is not None:
            return hit_point

        # Fallback to ground plane intersection
        if ray_direction[1] != 0:
            t = (self.physics.grid_height - near_point[1]) / ray_direction[1]
            if t >= 0:
                return [near_point[i] + ray_direction[i] * t for i in range(3)]

        return None

    def handle_cube_placement(self):
        if self.mouse_grabbed:
            self.placing_cube = False
            self.prev_left_mouse = False
            self.prev_right_mouse = False
            self.awaiting_cube_release = False
            return

        mouse_buttons = pygame.mouse.get_pressed()
        left_pressed = mouse_buttons[0]
        right_pressed = mouse_buttons[2]
        mouse_pos = pygame.mouse.get_pos()

        if self.show_side_panel and not self.mouse_grabbed:
            hovered_button = None
            for button in self.structure_buttons:
                if button['rect'].collidepoint(mouse_pos):
                    hovered_button = button
                    break

            if hovered_button and left_pressed and not self.prev_left_mouse:
                self.set_active_structure(hovered_button['preset']['name'])
                self.placing_cube = True
                self.awaiting_cube_release = True

        hit_point = None
        if self.placing_cube:
            hit_point = self.get_cursor_world_hit()
            if hit_point is not None:
                x, y, z = hit_point
                y_offset = self.get_structure_half_height()
                y = max(y, self.physics.grid_height) + y_offset
                self.cube_preview_pos = [x, y, z]

            if self.awaiting_cube_release:
                if not left_pressed:
                    self.awaiting_cube_release = False
            else:
                if left_pressed and not self.prev_left_mouse and hit_point is not None:
                    self.spawn_structure(self.cube_preview_pos)
                    self.placing_cube = False
                elif right_pressed and not self.prev_right_mouse:
                    self.placing_cube = False

        if not self.placing_cube:
            self.awaiting_cube_release = False

        self.prev_left_mouse = left_pressed
        self.prev_right_mouse = right_pressed

    def set_active_structure(self, name=None):
        if not self.structure_presets:
            self.active_structure = None
            return
        if name:
            for idx, preset in enumerate(self.structure_presets):
                if preset['name'] == name:
                    self.current_structure_index = idx
                    self.active_structure = preset
                    break
            else:
                self.active_structure = self.structure_presets[self.current_structure_index]
        else:
            self.active_structure = self.structure_presets[self.current_structure_index]

        if self.active_structure and self.active_structure.get('generator') == 'cluster':
            cluster = self.active_structure['cluster']
            self.active_cube_size = self._normalize_vector3(cluster.get('size', [1.0, 1.0, 1.0]), [1.0, 1.0, 1.0])
            grid_counts = self._normalize_grid_counts(cluster.get('grid_count', (1, 1, 1)))
            self.active_total_size = [self.active_cube_size[i] * grid_counts[i] for i in range(3)]
            self.cube_size = max(self.active_total_size)
        
    def get_structure_half_height(self):
        if not self.active_structure:
            return self.active_total_size[1] / 2.0
        if self.active_structure.get('generator') == 'cluster':
            cluster = self.active_structure['cluster']
            cube_size = self._normalize_vector3(cluster.get('size', self.active_cube_size), self.active_cube_size)
            grid_counts = self._normalize_grid_counts(cluster.get('grid_count', (1, 1, 1)))
            total_height = cube_size[1] * grid_counts[1]
            return total_height / 2.0
        return self.active_total_size[1] / 2.0

    def spawn_structure(self, position):
        if not self.active_structure:
            return

        # Check for nearby structures to prevent interlocking
        nearby_objects = self.physics.check_nearby_objects(position, radius=2.0)
        if nearby_objects:
            # If there are nearby objects, adjust position slightly
            position = list(position)
            position[1] += 0.1  # Move up slightly to prevent interlocking
            
        generator = self.active_structure.get('generator')
        if generator == 'cluster':
            self.spawn_cluster_structure(position, self.active_structure['cluster'])

    def draw_textured_cube(self, size, texture_id):
        """Draw a cuboid with texture applied to all faces"""
        # Half sizes for each dimension
        hx = size[0] / 2.0
        hy = size[1] / 2.0
        hz = size[2] / 2.0
        
        # Enable texturing and setup material properties
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        # Enable texture repeating
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        
        # Enable lighting for this object
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Begin drawing the cuboid
        glBegin(GL_QUADS)
        
        # Front face (z = hz)
        glNormal3f(0.0, 0.0, 1.0)  # Normal pointing out of front face
        glTexCoord2f(0.0, 0.0); glVertex3f(-hx, -hy,  hz)
        glTexCoord2f(1.0, 0.0); glVertex3f( hx, -hy,  hz)
        glTexCoord2f(1.0, 1.0); glVertex3f( hx,  hy,  hz)
        glTexCoord2f(0.0, 1.0); glVertex3f(-hx,  hy,  hz)
        
        # Back face (z = -hz)
        glNormal3f(0.0, 0.0, -1.0)  # Normal pointing out of back face
        glTexCoord2f(0.0, 0.0); glVertex3f(-hx, -hy, -hz)
        glTexCoord2f(0.0, 1.0); glVertex3f(-hx,  hy, -hz)
        glTexCoord2f(1.0, 1.0); glVertex3f( hx,  hy, -hz)
        glTexCoord2f(1.0, 0.0); glVertex3f( hx, -hy, -hz)
        
        # Top face (y = hy)
        glNormal3f(0.0, 1.0, 0.0)  # Normal pointing up
        glTexCoord2f(0.0, 0.0); glVertex3f(-hx,  hy, -hz)
        glTexCoord2f(0.0, 1.0); glVertex3f(-hx,  hy,  hz)
        glTexCoord2f(1.0, 1.0); glVertex3f( hx,  hy,  hz)
        glTexCoord2f(1.0, 0.0); glVertex3f( hx,  hy, -hz)
        
        # Bottom face (y = -hy)
        glTexCoord2f(0.0, 0.0); glVertex3f(-hx, -hy, -hz)
        glTexCoord2f(1.0, 0.0); glVertex3f( hx, -hy, -hz)
        glTexCoord2f(1.0, 1.0); glVertex3f( hx, -hy,  hz)
        glTexCoord2f(0.0, 1.0); glVertex3f(-hx, -hy,  hz)
        
        # Right face (x = hx)
        glTexCoord2f(0.0, 0.0); glVertex3f(hx, -hy, -hz)
        glTexCoord2f(0.0, 1.0); glVertex3f(hx,  hy, -hz)
        glTexCoord2f(1.0, 1.0); glVertex3f(hx,  hy,  hz)
        glTexCoord2f(1.0, 0.0); glVertex3f(hx, -hy,  hz)
        
        # Left face (x = -hx)
        glTexCoord2f(0.0, 0.0); glVertex3f(-hx, -hy, -hz)
        glTexCoord2f(1.0, 0.0); glVertex3f(-hx, -hy,  hz)
        glTexCoord2f(1.0, 1.0); glVertex3f(-hx,  hy,  hz)
        glTexCoord2f(0.0, 1.0); glVertex3f(-hx,  hy, -hz)
        
        glEnd()
        
        # Disable texturing
        glDisable(GL_TEXTURE_2D)
    
    def spawn_cluster_structure(self, position, config):        
        # Get cube size as a 3D vector, defaulting to [0.1, 0.1, 0.1] if not specified
        cube_size = self._normalize_vector3(config.get('size', 0.1), 0.1)
        
        # Get grid counts, defaulting to [4, 8, 2] if not specified
        grid_counts = self._normalize_grid_counts(config.get('grid_count', (4, 8, 2)))
        
        # Get mass, defaulting to 0.1 if not specified
        mass = config.get('mass', 0.1)
        
        # Get fill type and base color
        fill_type = config.get('fill', 'solid')
        base_color = config.get('base_color', (0.8, 0.8, 0.8, 1.0))
        
        # Calculate total structure size and half extents
        total_size = [cube_size[i] * grid_counts[i] for i in range(3)]
        half_total = [total_size[i] / 2.0 for i in range(3)]
        
        # Load textures if needed
        if fill_type == 'small_bricks' and not hasattr(self, 'small_bricks_textures'):
            self._load_textures('textures/small_bricks', 'small_bricks_textures')
        elif fill_type == 'rust_metal' and not hasattr(self, 'rust_metal_textures'):
            self._load_textures('textures/rust_metal', 'rust_metal_textures')
        elif fill_type == 'concrete' and not hasattr(self, 'concrete_textures'):
            self._load_textures('textures/concrete', 'concrete_textures')
        # Generate cubes in a 3D grid
        for ix in range(grid_counts[0]):
            for iy in range(grid_counts[1]):
                for iz in range(grid_counts[2]):
                    # Calculate position offset for this cube
                    offset = [
                        (ix * cube_size[0]) - half_total[0] + (cube_size[0] / 2.0),
                        (iy * cube_size[1]) - half_total[1] + (cube_size[1] / 2.0),
                        (iz * cube_size[2]) - half_total[2] + (cube_size[2] / 2.0)
                    ]
                    
                    # Calculate final position in world space
                    cube_position = [
                        position[0] + offset[0],
                        position[1] + offset[1],
                        position[2] + offset[2]
                    ]
                    
                    # Get fill type from config
                    fill_type = config.get('fill', 'solid')
                    
                    # Choose a random texture if needed
                    texture_id = None
                    if fill_type == 'small_bricks' and hasattr(self, 'small_bricks_textures') and self.small_bricks_textures:
                        texture_id = random.choice(self.small_bricks_textures)
                    elif fill_type == 'rust_metal' and hasattr(self, 'rust_metal_textures') and self.rust_metal_textures:
                        texture_id = random.choice(self.rust_metal_textures)
                    elif fill_type == 'concrete' and hasattr(self, 'concrete_textures') and self.concrete_textures:
                        texture_id = random.choice(self.concrete_textures)
                    # Prepare metadata for the structure
                    metadata = {}
                    if 'hit_sound' in config:
                        metadata['hit_sound'] = config['hit_sound']
                    if 'stiff' in config:
                        metadata['stiff'] = config['stiff']
                    
                    # Add to physics world with fill type, texture ID, and metadata
                    mesh_obj = None
                    if 'mesh_path' in config:
                        try:
                            from models.obj_loader import OBJ
                            mesh_obj = OBJ(config['mesh_path'])
                        except Exception as e:
                            mesh_obj = None
                    cube_id = self.physics.add_structure(
                        position=cube_position,
                        size=cube_size,
                        mass=0.0 if config.get('stiff', False) else mass,  # Set mass to 0 for stiff structures
                        color=base_color,
                        fill=fill_type,
                        metadata=metadata,
                        mesh_obj=mesh_obj
                    )
                    
                    # Store the texture ID in the structure data
                    if cube_id is not None and texture_id is not None:
                        self.physics.structures[cube_id]['texture_id'] = texture_id
                    
                    # Skip if a structure already exists at this position
                    if cube_id is None:
                        continue
                        
                    # Create display list for rendering
                    display_list = glGenLists(1)
                    glNewList(display_list, GL_COMPILE)
                    
                    # Apply tint variation if specified
                    if 'tint_variation' in config and 'tint_strength' in config:
                        tint_strength = config['tint_strength']
                        tint_variation = self._normalize_vector3(config['tint_variation'], 1.0)
                        r = min(1.0, base_color[0] * (1.0 + (random.random() - 0.5) * tint_strength * tint_variation[0]))
                        g = min(1.0, base_color[1] * (1.0 + (random.random() - 0.5) * tint_strength * tint_variation[1]))
                        b = min(1.0, base_color[2] * (1.0 + (random.random() - 0.5) * tint_strength * tint_variation[2]))
                        glColor4f(r, g, b, base_color[3] if len(base_color) > 3 else 1.0)
                    else:
                        glColor4f(*base_color) if len(base_color) > 3 else glColor3f(*base_color)
                    
                    # Choose rendering method based on fill type
                    if fill_type == 'small_bricks' and hasattr(self, 'small_bricks_textures') and self.small_bricks_textures:
                        # Use random small brick texture
                        texture = random.choice(self.small_bricks_textures)
                        self.draw_textured_cube(cube_size, texture)
                    elif fill_type == 'rust_metal' and hasattr(self, 'rust_metal_textures') and self.rust_metal_textures:
                        # Use random rust metal texture
                        texture = random.choice(self.rust_metal_textures)
                        self.draw_textured_cube(cube_size, texture)
                    elif fill_type == 'concrete' and hasattr(self, 'concrete_textures') and self.concrete_textures:
                        # Use random concrete texture
                        texture = random.choice(self.concrete_textures)
                        self.draw_textured_cube(cube_size, texture)
                    else:
                        # Default solid color cube
                        half = [s / 2.0 for s in cube_size]
                        glBegin(GL_QUADS)
                        # Front face
                        glVertex3f(-half[0], -half[1], half[2])
                        glVertex3f(half[0], -half[1], half[2])
                        glVertex3f(half[0], half[1], half[2])
                        glVertex3f(-half[0], half[1], half[2])
                        # Back face
                        glVertex3f(-half[0], -half[1], -half[2])
                        glVertex3f(-half[0], half[1], -half[2])
                        glVertex3f(half[0], half[1], -half[2])
                        glVertex3f(half[0], -half[1], -half[2])
                        # Top face
                        glVertex3f(-half[0], half[1], -half[2])
                        glVertex3f(-half[0], half[1], half[2])
                        glVertex3f(half[0], half[1], half[2])
                        glVertex3f(half[0], half[1], -half[2])
                        # Bottom face
                        glVertex3f(-half[0], -half[1], -half[2])
                        glVertex3f(half[0], -half[1], -half[2])
                        glVertex3f(half[0], -half[1], half[2])
                        glVertex3f(-half[0], -half[1], half[2])
                        # Right face
                        glVertex3f(half[0], -half[1], -half[2])
                        glVertex3f(half[0], half[1], -half[2])
                        glVertex3f(half[0], half[1], half[2])
                        glVertex3f(half[0], -half[1], half[2])
                        # Left face
                        glVertex3f(-half[0], -half[1], -half[2])
                        glVertex3f(-half[0], -half[1], half[2])
                        glVertex3f(-half[0], half[1], half[2])
                        glVertex3f(-half[0], half[1], -half[2])
                        glEnd()
                    
                    glEndList()


    def _load_textures(self, texture_dir, attribute_name):
        """Load all textures from the specified directory and store them in the given attribute."""
        import os
        import pygame
        
        # Initialize the textures list for this attribute
        setattr(self, attribute_name, [])
        textures = getattr(self, attribute_name)
        
        # Check if the texture directory exists
        if not os.path.isdir(texture_dir):
            return
        
        # Get all image files from the directory
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tga']
        for filename in os.listdir(texture_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                try:
                    # Load the image file
                    filepath = os.path.join(texture_dir, filename)
                    texture_surface = pygame.image.load(filepath).convert_alpha()
                    
                    # Convert to the correct format for OpenGL
                    texture_data = pygame.image.tostring(texture_surface, 'RGBA', 1)
                    width, height = texture_surface.get_rect().size
                    
                    # Generate a texture ID
                    texture_id = glGenTextures(1)
                    glBindTexture(GL_TEXTURE_2D, texture_id)
                    
                    # Set texture parameters
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                    
                    # Upload the texture data and generate mipmaps
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
                    glGenerateMipmap(GL_TEXTURE_2D)
                    
                    # Store the texture ID
                    textures.append(texture_id)
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
        
        if not textures:
            return
    
    @staticmethod
    def _normalize_vector3(value, default):
        if isinstance(value, (list, tuple)):
            vec = []
            for i in range(3):
                if i < len(value) and value[i] is not None:
                    vec.append(float(value[i]))
                else:
                    vec.append(float(default) if not isinstance(default, (list, tuple)) else float(default[i]))
        else:
            vec = [float(value if value is not None else default)] * 3
        return vec

    @staticmethod
    def _normalize_grid_counts(value):
        if isinstance(value, (list, tuple)):
            counts = []
            for i in range(3):
                if i < len(value) and value[i] is not None:
                    counts.append(max(1, int(round(value[i]))))
                else:
                    counts.append(1)
        else:
            count = max(1, int(round(value if value is not None else 1)))
            counts = [count, count, count]
        return counts

    def reset_structures(self):
        """Remove all structures from the physics world"""
        # Create a list of structure IDs to avoid modifying the dictionary during iteration
        structure_ids = list(self.physics.structures.keys())
        for structure_id in structure_ids:
            self.physics.remove_structure(structure_id)
            
        # Reset placement state
        self.placing_cube = False
        self.awaiting_cube_release = False
    
    def update_camera(self):
        glLoadIdentity()
        
        # Calculate camera position behind and above the player
        rad_yaw = math.radians(self.player.rotation_y)
        rad_pitch = math.radians(self.player.rotation_x)
        
        # Calculate camera offset from player
        horizontal_distance = self.camera_distance * math.cos(rad_pitch)
        vertical_offset = self.camera_distance * math.sin(rad_pitch)
        
        # Calculate camera position (2 units to the left of the player)
        camera_x = self.player.x - math.sin(rad_yaw) * horizontal_distance + math.cos(rad_yaw) * -2.0
        camera_z = self.player.z - math.cos(rad_yaw) * horizontal_distance - math.sin(rad_yaw) * -2.0
        camera_y = self.player.y + self.player.camera_height + vertical_offset
        
        # Calculate look-at point (2 units to the right of the player)
        look_x = self.player.x + math.sin(rad_yaw) * 2.0 - math.cos(rad_yaw) * 2.0
        look_z = self.player.z + math.cos(rad_yaw) * 2.0 + math.sin(rad_yaw) * 2.0
        look_y = self.player.y + 1.0  # Look slightly above the player
        
        # Calculate camera forward vector (from camera to look-at point)
        forward_x = look_x - camera_x
        forward_y = look_y - camera_y
        forward_z = look_z - camera_z
        
        # Normalize forward vector
        forward_length = math.sqrt(forward_x**2 + forward_y**2 + forward_z**2)
        if forward_length > 0:
            forward_x /= forward_length
            forward_y /= forward_length
            forward_z /= forward_length
            
            # If camera is below ground, move it along forward vector until it's above ground
            min_ground_height = self.player.ground_level + 0.1  # Small offset above ground
            if camera_y < min_ground_height:
                # Calculate how much we need to move the camera up to reach min_ground_height
                t = (min_ground_height - camera_y) / forward_y if forward_y != 0 else 0
                if t > 0:  # Only move forward if that will help (forward_y is positive)
                    camera_x += forward_x * t
                    camera_y += forward_y * t
                    camera_z += forward_z * t
        
        # Set up the camera with world-up vector to prevent roll
        up_x = 0.0
        up_y = 1.0  # World up is always in the positive Y direction
        up_z = 0.0
        
        # Set up the camera
        gluLookAt(
            camera_x, camera_y, camera_z,  # Camera position
            look_x, look_y, look_z,       # Look at point
            up_x, up_y, up_z              # Up vector (adjusted for pitch)
        )
    
    def draw_crosshair(self):
        # Switch to orthographic projection for 2D UI elements
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth testing for 2D elements
        glDisable(GL_DEPTH_TEST)
        
        # Draw crosshair
        size = 3
        center_x, center_y = self.display[0] // 2, self.display[1] // 2
        glColor3f(1.0, 1.0, 1.0)  # White color
        glLineWidth(1.0)
        glBegin(GL_LINES)
        # Horizontal line
        glVertex2f(center_x - size, center_y)
        glVertex2f(center_x + size, center_y)
        # Vertical line
        glVertex2f(center_x, center_y - size)
        glVertex2f(center_x, center_y + size)
        glEnd()
        
        # Restore OpenGL state for 3D rendering
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        # Target point visualization removed
        glEnable(GL_LIGHTING)
    
    def toggle_mouse_grab(self):
        # Toggle mouse grab state
        self.mouse_grabbed = not self.mouse_grabbed
        pygame.event.set_grab(self.mouse_grabbed)
        pygame.mouse.set_visible(not self.mouse_grabbed)
        
        if self.mouse_grabbed:
            # When regrabbing, center the mouse
            pygame.mouse.set_pos(self.display[0] // 2, self.display[1] // 2)
        else:
            # When releasing, store the last mouse position
            self.last_mouse_pos = pygame.mouse.get_pos()
    
    def handle_input(self):
        # Handle events
        current_time = pygame.time.get_ticks() / 1000.0  # Current time in seconds
        
        # Check for held mouse button for auto-fire
        if self.mouse_grabbed and pygame.mouse.get_pressed()[0]:  # Left mouse button
            if not self.is_firing:  # First frame of holding the button
                self.is_firing = True
                self.start_shoot_sound()
                self.last_shot_time = current_time - self.fire_delay  # Allow immediate first shot
            
            # Check if enough time has passed since the last shot
            if current_time - self.last_shot_time >= self.fire_delay:
                self.shoot()
                self.last_shot_time = current_time
        else:
            if self.is_firing:
                self.stop_shoot_sound()
            self.is_firing = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self.bullet_impulse_active:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        self.bullet_impulse_active = False
                        # Clean up text buffer on exit
                        if hasattr(self, 'bullet_impulse_text'):
                            del self.bullet_impulse_text
                        self.bullet_impulse_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        # Ensure bullet_impulse_text exists before using
                        if not hasattr(self, 'bullet_impulse_text'):
                            self.bullet_impulse_text = '0'
                        if self.bullet_impulse_text:
                            self.bullet_impulse_text = self.bullet_impulse_text[:-1]
                            if not self.bullet_impulse_text:  # If empty after backspace
                                self.bullet_impulse_text = '0'
                        # Update the actual value if valid
                        try:
                            if self.bullet_impulse_text:
                                self.bullet_impulse = int(self.bullet_impulse_text)
                            else:
                                self.bullet_impulse = 0
                        except ValueError:
                            self.bullet_impulse = 0
                            self.bullet_impulse_text = '0'
                    elif event.unicode.isdigit():
                        # Only allow up to 4 digits
                        if not hasattr(self, 'bullet_impulse_text'):
                            self.bullet_impulse_text = ''
                        if len(self.bullet_impulse_text) < 4:
                            # Handle leading zero
                            if self.bullet_impulse_text == '0':
                                self.bullet_impulse_text = event.unicode
                            else:
                                self.bullet_impulse_text += event.unicode
                        # Update the actual value if valid
                        try:
                            new_value = int(self.bullet_impulse_text)
                            if 0 <= new_value <= 9999:
                                self.bullet_impulse = new_value
                            else:
                                # Revert if out of bounds
                                self.bullet_impulse_text = str(self.bullet_impulse)
                        except ValueError:
                            # Revert to last valid value if invalid
                            self.bullet_impulse_text = str(self.bullet_impulse)
                else:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_TAB:
                        self.toggle_mouse_grab()
                    elif event.key == pygame.K_r:
                        self.reset_structures()
                    elif event.key == pygame.K_m:  # Add M key to toggle side panel
                        self.show_side_panel = not self.show_side_panel
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # Right mouse button
                    self.toggle_mouse_grab()
                if event.button == 1:  # Left mouse button
                    # Check if clicking on bullet impulse input
                    if hasattr(self, 'bullet_impulse_rect') and self.bullet_impulse_rect.collidepoint(event.pos):
                        self.bullet_impulse_active = True
                    else:
                        self.bullet_impulse_active = False
                        
                        # Check if menu button was clicked
                        if self.menu_button_rect.collidepoint(event.pos):
                            self.show_side_panel = not self.show_side_panel
                            if self.show_side_panel and self.show_side_panel2:
                                self.show_side_panel2 = False
                        # Check if second panel button was clicked
                        elif self.panel2_button_rect.collidepoint(event.pos):
                            self.show_side_panel2 = not self.show_side_panel2
                            if self.show_side_panel2 and self.show_side_panel:
                                self.show_side_panel = False
                        # Check if lock button was clicked
                        elif self.lock_button_rect.collidepoint(event.pos):
                            self.toggle_mouse_grab()
                        elif self.mouse_grabbed:  # Left click to shoot if mouse is grabbed
                            self.shoot()
                            self.last_shot_time = current_time
                            self.is_firing = True
                            self.start_shoot_sound()
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button released
                    if self.reset_button_rect.collidepoint(event.pos):
                        self.reset_structures()
                    if self.is_firing:
                        self.stop_shoot_sound()
                    self.is_firing = False
            elif event.type == pygame.MOUSEWHEEL:
                # Handle mouse wheel zoom
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # Zoom in/out based on scroll direction
                    zoom_amount = event.y * self.zoom_speed * 0.2  # Reduced sensitivity for scroll
                    self.camera_distance = max(self.min_zoom, min(self.max_zoom, self.camera_distance - zoom_amount))
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                self.display = (event.w, event.h)
                self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL | RESIZABLE)
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_grabbed:
                    rel_x, rel_y = event.rel
                    
                    # Update player rotation based on relative movement
                    if rel_x != 0 or rel_y != 0:
                        # Update yaw (left/right)
                        self.player.rotation_y = (self.player.rotation_y - rel_x * self.player.mouse_sensitivity) % 360
                        
                        # Handle vertical mouse movement for camera distance when right mouse button is down
                        if pygame.mouse.get_pressed()[2]:  # Right mouse button
                            # Adjust camera distance based on vertical movement (slower than scroll)
                            zoom_amount = rel_y * 0.01  # Very small multiplier for smooth adjustment
                            self.camera_distance = max(self.min_zoom, min(self.max_zoom, self.camera_distance - zoom_amount))
                        else:
                            # Normal pitch control when not right-click dragging
                            new_pitch = self.player.rotation_x + rel_y * self.player.mouse_sensitivity
                            self.player.rotation_x = max(self.player.min_pitch, min(self.player.max_pitch, new_pitch))
        
        # Get time since last frame
        self.delta_time = self.clock.tick(self.fps) / 1000.0
        
        # Handle continuous key states for movement
        keys = pygame.key.get_pressed()
        move_speed = self.player.move_speed * self.delta_time
        
        # Forward/backward movement (relative to box orientation)
        if keys[pygame.K_s]:  # S key moves forward
            self.player.move_forward(move_speed)
        if keys[pygame.K_w]:  # W key moves backward
            self.player.move_forward(-move_speed)
            
        # Strafing (left/right movement relative to box orientation)
        if keys[pygame.K_d]:  # D key strafes right
            self.player.move_right(move_speed)
        if keys[pygame.K_a]:  # A key strafes left
            self.player.move_right(-move_speed)
            
        # Q key still moves up 
        if keys[pygame.K_q]:
            self.player.jump()

        # Handle cube placement input when mouse grab is released
        self.handle_cube_placement()
    
    def update(self):
        """Update the simulation state."""
        # Step the physics simulation
        self.physics.step()
        
        # Update target point (where crosshair is pointing)
        self.update_target_point()
        
        # Update bullets
        self.update_bullets()
        
        # Update player position with physics
        self.player.update_position(self.delta_time)
        # Update camera to follow player
        self.update_camera()

    def render_scene(self):
        try:
            # Clear the screen and depth buffer with a dark color
            glClearColor(0.0, 0.0, 0.1, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # ===== SKYBOX RENDERING =====
            # Save projection matrix
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            
            # Set up projection with a fixed FOV
            aspect_ratio = self.display[0] / float(self.display[1])
            gluPerspective(45.0, aspect_ratio, 0.1, 1000.0)
            
            # Switch to modelview matrix
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            # Save current state
            glPushAttrib(GL_ENABLE_BIT | GL_VIEWPORT_BIT | GL_TRANSFORM_BIT | GL_TEXTURE_BIT | GL_COLOR_BUFFER_BIT | GL_LIGHTING_BIT)
            
            try:
                # Save current lighting state
                lighting_was_enabled = glIsEnabled(GL_LIGHTING)
                
                # Set up for skybox rendering
                glDisable(GL_DEPTH_TEST)   # Disable depth test so skybox is always behind everything
                glDisable(GL_LIGHTING)     # Disable lighting for the skybox
                glDisable(GL_BLEND)        # Disable blending
                glEnable(GL_TEXTURE_2D)    # Enable texturing for the skybox
                glDepthMask(False)         # Don't write to depth buffer
                
                # Set a fixed color to prevent lighting from affecting the skybox
                glColor4f(1.0, 1.0, 1.0, 1.0)
                
                # Set up the camera orientation (only rotation, no translation)
                # Keep the original pitch rotation but maintain the inverted yaw
                glRotatef(self.player.rotation_x, 1, 0, 0)  # Original pitch
                glRotatef(-self.player.rotation_y, 0, 1, 0)  # Inverted yaw
                
                # Draw the skybox if available
                if hasattr(self, 'skybox') and self.skybox is not None:
                    try:
                        self.skybox.draw()
                    except Exception as e:
                        print(f"ERROR in skybox.draw(): {e}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                print(f"Error in skybox rendering: {e}")
                import traceback
                traceback.print_exc()
                
            finally:
                # Restore OpenGL state
                glDepthMask(True)
                glEnable(GL_DEPTH_TEST)
                # Only re-enable lighting if it was enabled before
                if lighting_was_enabled:
                    glEnable(GL_LIGHTING)
                glPopAttrib()
                
                # Restore matrices
                glPopMatrix()  # Pop modelview
                glMatrixMode(GL_PROJECTION)
                glPopMatrix()  # Pop projection
                
            # ===== SCENE RENDERING =====
            # Set up projection for the rest of the scene
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45.0, aspect_ratio, 0.1, 1000.0)
            
            # Switch to modelview matrix and set up camera
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            self.update_camera()
            
            # Enable depth testing and lighting for the scene
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            
        except Exception as e:
            print(f"Error in render_scene: {e}")
            import traceback
            traceback.print_exc()
            # Try to recover by resetting the matrix mode
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
        
        # Draw the ground
        self.draw_ground()
        
        # Draw physics objects
        self.draw_physics_objects()
        
        # Draw the player
        self.draw_player()
        
        # Draw cube preview if placing
        if self.placing_cube:
            self.draw_cube_preview()
        
        # Draw bullet trails
        for bullet in self.bullets:
            self.draw_bullet_trail(bullet)
        
        # Draw UI elements
        if self.show_side_panel:
            self.draw_side_panel()
        elif self.show_side_panel2:
            self.draw_side_panel2()
        self.draw_crosshair()
        self.draw_menu_button()
        self.draw_reset_button()
        self.draw_lock_button()
    
    def run(self):
        # Initialize mouse settings
        pygame.mouse.set_visible(False)
        pygame.mouse.set_pos(self.display[0] // 2, self.display[1] // 2)  # Center mouse
        
        while self.running:
            # Handle input and update
            self.handle_input()
            self.update()  # Update simulation state
            self.update_camera()
            
            # Render the scene
            self.render_scene()
            
            # Update the display
            pygame.display.flip()
            
            # Cap the frame rate
            self.clock.tick(self.fps)
        
        # Clean up
        self.physics.cleanup()
        pygame.quit()

if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()
