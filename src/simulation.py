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
from pathlib import Path
import pybullet as p  # Add pybullet import
import threading
import time

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
            
        # Initialize structure input fields
        self.active_input_field = None
        self.input_text = ''
        
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
        self.last_click_time = 0  # For click debouncing
        
        # Player model selection
        self.show_player_panel = False
        self.player_buttons = []
        
        # Side panel scrolling
        self.side_panel_scroll_y = 0
        self.side_panel_max_scroll = 0
        self.scroll_bar_rect = None
        
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
        # Initialize loading state
        self.is_loading_model = False
        self.loading_text = ""
        self.loading_start_time = 0
        # Load initial model
        self.currPlayerModel = self.model_lib.load_model('Cannon')
        
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
        
        # Initialize input_values for all presets
        for preset in self.structure_presets:
            if 'input_values' not in preset:
                preset['input_values'] = {}
                
            # Initialize default values for cluster structures
            if preset.get('generator') == 'cluster' and 'cluster' in preset:
                cluster = preset['cluster']
                # Initialize size values
                size = cluster.get('size', [0.1, 0.1, 0.1])
                preset['input_values'].update({
                    'size_x': str(size[0]),
                    'size_y': str(size[1]),
                    'size_z': str(size[2])
                })
                
                # Initialize grid values
                grid = cluster.get('grid_count', (1, 1, 1))
                if isinstance(grid, (list, tuple)) and len(grid) >= 3:
                    preset['input_values'].update({
                        'grid_x': str(grid[0]),
                        'grid_y': str(grid[1]),
                        'grid_z': str(grid[2])
                    })
        
        self.current_structure_index = 0
        self.active_structure = None
        
        # Set up the perspective
        self.setup_projection()
        
        # Initialize physics world with ground at y=0
        self.physics = PhysicsWorld(gravity=(0, -9.81*3, 0), grid_height=0.0)
        
        # Initialize pybullet physics client
        self.physics_client = p.connect(p.DIRECT)  # Use p.GUI for visualization
        p.setGravity(0, -9.81*3, 0)
        
        # Load cannonball texture
        self.cannonball_texture = self.load_texture(os.path.join('textures', 'cannonball.png'))
        
        # Enable depth testing and other OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Create player positioned at ground level in physics coordinates
        self.player = Player(x=0, y=0, z=0)  # Initialize at origin
        self.player.ground_level = 0.0  # Set ground level to match physics world
        self.player.y = 0  # Position will be adjusted in draw_player
        
        # Override player's check_collision method to use physics world with OBB collision detection
        def check_collision(x, y, z):
            # Check if the player would collide with any structure at the given position
            # Exclude the player's own structure from collision checks
            player_structure_id = getattr(self, 'player_structure_id', None)
            if player_structure_id is None:
                print("No player_structure_id found!")
                return False
                
            # Get player collision size and position
            player_size = getattr(self, 'player_collision_size', [1.0, 1.8, 1.0])
            player_half_extents = [s / 2.0 for s in player_size]
            # Offset player position 1.1 units down on Y-axis for better grounding
            player_pos = [x, y - 0.99, z]
            player_bottom = y - 0.99
            player_top = y - 0.99 + player_size[1]
            

            # Player collision box is axis-aligned (no rotation)
            player_rotation = [0, 0, 0, 1]  # Identity quaternion
            

            collision_found = False
            
            for structure_id, structure in self.physics.structures.items():
                if structure_id == player_structure_id:
                    continue  # Skip self-collision
                    
                # Get structure position and orientation
                struct_pos = self.physics.get_structure_position(structure_id)
                if struct_pos is None:
                    print(f"  - Structure {structure_id}: No position")
                    continue
                    
                struct_orn = self.physics.get_structure_orientation(structure_id)
                if struct_orn is None:
                    struct_orn = [0, 0, 0, 1]  # Default to no rotation
                    
                # Get structure size and type
                struct_size = structure.get('size', [1, 1, 1])
                struct_type = structure.get('fill', 'unknown')
                struct_half_extents = [s / 2.0 for s in struct_size]
                
                # Calculate structure bounds
                struct_bottom = struct_pos[1] - struct_half_extents[1]
                struct_top = struct_pos[1] + struct_half_extents[1]
                
                # First, check if player is within the structure's height range
                if player_top < struct_bottom or player_bottom > struct_top:
                    continue
                
                # Convert quaternion to rotation matrix for SAT
                def quat_to_rot_matrix(q):
                    w, x, y, z = q
                    return [
                        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
                    ]
                
                # Get rotation matrices
                rot_a = quat_to_rot_matrix(player_rotation)
                rot_b = quat_to_rot_matrix(struct_orn)
                
                # Calculate relative position (player to structure)
                rel_pos = [player_pos[i] - struct_pos[i] for i in range(3)]
                
                # Convert to structure's local space
                rel_pos_local = [
                    sum(rot_b[0][i] * rel_pos[i] for i in range(3)),
                    sum(rot_b[1][i] * rel_pos[i] for i in range(3)),
                    sum(rot_b[2][i] * rel_pos[i] for i in range(3))
                ]
                
                # Calculate rotation matrix: C = A^T * B
                rot_matrix = [[0]*3 for _ in range(3)]
                for i in range(3):
                    for j in range(3):
                        rot_matrix[i][j] = sum(rot_a[k][i] * rot_b[k][j] for k in range(3))
                
                # Check for separating axis along each axis of box A (player)
                separating_axis_found = False
                for i in range(3):
                    ra = player_half_extents[i]
                    rb = sum(struct_half_extents[j] * abs(rot_matrix[i][j]) for j in range(3))
                    t = abs(sum(rel_pos[j] * rot_b[j][i] for j in range(3)))
                    if t > ra + rb + 0.1:  # Increased epsilon for better stability
                        separating_axis_found = True
                        break
                
                if separating_axis_found:
                    continue
                
                # Check for separating axis along each axis of box B (structure)
                for i in range(3):
                    ra = sum(player_half_extents[j] * abs(rot_matrix[j][i]) for j in range(3))
                    rb = struct_half_extents[i]
                    t = abs(sum(rel_pos[j] * rot_b[j][i] for j in range(3)))
                    if t > ra + rb + 0.1:  # Increased epsilon for better stability
                        separating_axis_found = True
                        break
                
                if separating_axis_found:
                    continue
                
                # Check cross products of axes (9 cases, but only 6 are unique)
                for i in range(3):
                    for j in range(3):
                        if i == j:
                            continue
                            
                        # Cross product of axis i of A and axis j of B
                        axis = [
                            rot_b[1][i] * rot_b[2][j] - rot_b[2][i] * rot_b[1][j],
                            rot_b[2][i] * rot_b[0][j] - rot_b[0][i] * rot_b[2][j],
                            rot_b[0][i] * rot_b[1][j] - rot_b[1][i] * rot_b[0][j]
                        ]
                        
                        # Skip if axis is zero (parallel axes)
                        length = math.sqrt(sum(x*x for x in axis))
                        if length < 1e-6:
                            continue
                            
                        # Normalize axis
                        axis = [x / length for x in axis]
                        
                        # Project both boxes onto the axis
                        ra = sum(player_half_extents[k] * abs(sum(rot_a[m][k] * axis[m] for m in range(3))) for k in range(3))
                        rb = sum(struct_half_extents[k] * abs(sum(rot_b[m][k] * axis[m] for m in range(3))) for k in range(3))
                        t = abs(sum(rel_pos[k] * axis[k] for k in range(3)))
                        
                        if t > ra + rb + 0.1:  # Increased epsilon for better stability
                            separating_axis_found = True
                            break
                    
                    if separating_axis_found:
                        break
                
                if not separating_axis_found:
                    # No separating axis found, boxes are colliding
                    collision_found = True
                    break  # No need to check other structures if we found a collision
            
            return collision_found
            
            print("No collisions detected with any structure")
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
        self.fire_rate = 10.0  # Bullets per second (for non-cannon weapons)
        self.fire_delay = 1.0 / self.fire_rate  # Time between shots in seconds
        self.last_shot_time = 0.0  # When the last shot was fired
        self.is_firing = False  # Whether the fire button is being held down
        self.bullet_impulse_active = False  # Whether bullet impulse mode is active
        
        # Cannon-specific settings
        self.cannon_fire_rate = 1.0  # Shots per second for cannon
        self.cannon_fire_delay = 1.0 / self.cannon_fire_rate
        self.cannon_ball_speed = 80.0  # Speed of cannonballs
        self.cannon_ball_radius = 0.15  # Size of cannonballs
        self.cannon_ball_mass = 19.0  # Mass of cannonballs
        self.last_cannon_shot = 0  # Last time a cannonball was fired
        
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
        self.cube_rotation = 0  # Rotation angle in degrees
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
            exr_path = os.path.join(os.path.dirname(__file__), 'textures', 'brown_mud_leaves_01_rough_4k.exr')
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
                color_path = os.path.join(os.path.dirname(__file__), 'textures', 'brown_mud_leaves_01_diff_4k.jpg')
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
            texture_path = os.path.join(os.path.dirname(__file__), 'textures', 'brown_mud_leaves_01_diff_4k.jpg')
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
                
            # Draw circular shadow first (under the object)
            # Use the maximum dimension for the radius but scale it down for better appearance
            shadow_radius = max(size[0], size[2]) * 0.3  # 30% of the largest dimension
            self.draw_circular_shadow(pos[0], pos[2], shadow_radius, alpha=0.4)
            
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
            # Update cannonball positions from physics
            if bullet['type'] == 'cannonball' and 'body_id' in bullet:
                try:
                    pos, _ = p.getBasePositionAndOrientation(bullet['body_id'])
                    bullet['position'] = pos
                    
                    # Remove if below ground or timed out
                    if pos[1] < self.physics.grid_height or current_time - bullet['spawn_time'] > 10.0:
                        p.removeBody(bullet['body_id'])
                        self.bullets.remove(bullet)
                        continue
                except:
                    self.bullets.remove(bullet)
                    continue
            
            # Update lifetime for hitscan bullets
            bullet['lifetime'] -= self.delta_time
            if bullet['lifetime'] <= 0:
                if bullet['type'] == 'cannonball' and 'body_id' in bullet:
                    try:
                        p.removeBody(bullet['body_id'])
                    except:
                        pass
                self.bullets.remove(bullet)
                
    def fire_cannon_ball(self, start_pos, direction):
        """Fire a physical cannonball from the given position in the given direction"""
        current_time = pygame.time.get_ticks() / 1000.0
        if current_time - self.last_cannon_shot < self.cannon_fire_delay:
            return  # Rate limiting
            
        self.last_cannon_shot = current_time
        
        try:
            # Create a sphere collision shape for the cannonball
            col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=self.cannon_ball_radius)
            
            # Create the rigid body for the cannonball
            body_id = p.createMultiBody(
                baseMass=self.cannon_ball_mass,
                baseCollisionShapeIndex=col_shape,
                basePosition=start_pos,
                baseOrientation=[0, 0, 0, 1]
            )
            
            # Adjust physics properties to reduce gravity effect
            # Use a lighter mass and adjust damping for a floatier feel
            p.changeDynamics(
                body_id, 
                -1,  # -1 for the base
                mass=self.cannon_ball_mass, 
                linearDamping=0.03,  # More damping to slow down faster
                angularDamping=0.5,
                lateralFriction=0.5,
                spinningFriction=0.3,
                rollingFriction=0.1,
                restitution=0.3,  # Bounciness
                localInertiaDiagonal=[0.1, 0.1, 0.1]  # Lower inertia for more responsive movement
            )
            
            # Set the initial velocity (scaled by speed)
            velocity = [d * self.cannon_ball_speed for d in direction]
            p.resetBaseVelocity(body_id, linearVelocity=velocity)
            
            # Add to bullets list for rendering
            self.bullets.append({
                'type': 'cannonball',
                'body_id': body_id,
                'position': start_pos.copy(),
                'lifetime': 10.0,  # Max lifetime in seconds
                'spawn_time': current_time
            })
            
            # Play shoot sound if available
            if hasattr(self, 'shoot_sound') and self.shoot_sound:
                if not self.shoot_channel or not self.shoot_channel.get_busy():
                    if isinstance(self.shoot_sound, list):
                        # Play a random sound from the list
                        sound_to_play = random.choice(self.shoot_sound)
                        self.shoot_channel = sound_to_play.play()
                    else:
                        # Play the single sound
                        self.shoot_channel = self.shoot_sound.play()
                    
            # Step the simulation to ensure the body is created
            p.stepSimulation()
                    
        except Exception as e:
            print(f"Error creating cannonball: {e}")
            import traceback
            traceback.print_exc()
    
    def shoot(self):
        """Handle shooting mechanics with hitscan and high-impact force"""
        # Check if we're using the Cannon model
        model_name = self.currPlayerModel.metadata.get('name', '')
        is_cannon = 'Cannon' in model_name
        
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
            
            # For cannon, fire a physical cannonball
            if is_cannon:
                self.fire_cannon_ball(start_pos, direction)
                return  # Only fire one cannonball per shot
                
            # For hitscan weapons, perform raycast to detect hits
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
    
    def load_cannonball_texture(self):
        """Load the cannonball texture if available, otherwise use a procedural texture"""
        try:
            # Initialize texture ID if it doesn't exist
            if not hasattr(self, 'cannonball_texture'):
                self.cannonball_texture = glGenTextures(1)
                
            texture_path = os.path.join('textures', 'cannonball.png')

            
            # Generate a procedural texture if file doesn't exist
            if not os.path.exists(texture_path):
                    # Create a simple procedural texture
                    size = 256
                    texture = np.zeros((size, size, 4), dtype=np.uint8)
                    center = size // 2
                    radius = size // 2 - 2
                    
                    # Create a radial gradient for the cannonball
                    for y in range(size):
                        for x in range(size):
                            dx = x - center
                            dy = y - center
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist <= radius:
                                # Metallic gray with some noise for texture
                                intensity = 0.3 + 0.7 * (1.0 - dist/radius)
                                noise = 0.1 * (random.random() - 0.5)
                                value = int(55 + 100 * (intensity + noise))
                                value = max(0, min(255, value))
                                texture[y, x] = (value, value, value, 255)  # Grayscale
                            else:
                                texture[y, x] = (0, 0, 0, 0)  # Transparent
                    
                    # Convert to Pygame surface and then to OpenGL texture
                    surf = pygame.surfarray.make_surface(texture)
                    texture_data = pygame.image.tostring(surf, 'RGBA', 1)
                    
                    # Generate and bind the texture
                    self.cannonball_texture = glGenTextures(1)
                    glBindTexture(GL_TEXTURE_2D, self.cannonball_texture)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                    
                    # Upload the texture data
                    width, height = size, size
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
                               GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
                    glGenerateMipmap(GL_TEXTURE_2D)
                    
                    return True
                
            # If we get here, the file exists, so load it
            try:
                # Load the texture using Pygame
                texture_surface = pygame.image.load(texture_path).convert_alpha()
                texture_data = pygame.image.tostring(texture_surface, 'RGBA', 1)
                
                # Generate and bind the texture
                self.cannonball_texture = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, self.cannonball_texture)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                
                # Upload the texture data
                width, height = texture_surface.get_size()
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
                           GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
                glGenerateMipmap(GL_TEXTURE_2D)
                
                return True
            except Exception as e:
                print(f"Error loading cannonball texture: {e}")
                return False
        except Exception as e:
            print(f"Error in load_cannonball_texture: {e}")
            return False

    def draw_cannonball(self, bullet):
        """Draw a cannonball with texture"""
        if 'position' not in bullet:
            return
            
        # Try to load the texture if not already loaded
        if not hasattr(self, 'cannonball_texture_loaded'):
            self.cannonball_texture_loaded = self.load_cannonball_texture()
        
        glPushMatrix()
        glDisable(GL_LIGHTING)
        
        # Position the cannonball
        glTranslatef(*bullet['position'])
        
        # Add a subtle rotation based on time for a more dynamic look
        glRotatef(time.time() * 30 % 360, 1, 1, 1)
        
        # Draw the cannonball with texture if available
        if hasattr(self, 'cannonball_texture') and self.cannonball_texture_loaded:
            # Save the current matrix and enable texturing
            glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT)
            glEnable(GL_TEXTURE_2D)
            
            # Bind the texture and set parameters
            glBindTexture(GL_TEXTURE_2D, self.cannonball_texture)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            
            # Disable blending for 100% opacity
            glDisable(GL_BLEND)
            
            # Set material properties (100% opaque white)
            glColor4f(1.0, 1.0, 1.0, 1.0)
            
            # Draw the sphere with texture
            quad = gluNewQuadric()
            gluQuadricTexture(quad, GL_TRUE)
            gluQuadricNormals(quad, GLU_SMOOTH)
            gluSphere(quad, self.cannon_ball_radius, 32, 32)
            gluDeleteQuadric(quad)
            
            # Restore the previous state
            glPopAttrib()
            glEnable(GL_LIGHTING)
        else:
            # Fallback to a simple metallic sphere without texture
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            
            # Metallic dark gray color
            glColor3f(0.2, 0.2, 0.2)
            

            
            glow_quad = gluNewQuadric()
            gluSphere(glow_quad, self.cannon_ball_radius * 1.15, 16, 16)
            gluDeleteQuadric(glow_quad)
            
            # Draw the main sphere
            glColor3f(0.2, 0.2, 0.2)
            
            # Draw the sphere with smooth shading
            quad = gluNewQuadric()
            gluQuadricNormals(quad, GLU_SMOOTH)
            gluSphere(quad, self.cannon_ball_radius, 24, 24)
            gluDeleteQuadric(quad)
            
            # Add some specular highlights
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glColor4f(0.2, 0.2, 0.2, 0.95)
            
            quad = gluNewQuadric()
            gluSphere(quad, self.cannon_ball_radius * 1.05, 16, 16)
            gluDeleteQuadric(quad)
            
            glDisable(GL_BLEND)
        
        
        quad = gluNewQuadric()
        gluSphere(quad, self.cannon_ball_radius * 1.2, 12, 12)
        gluDeleteQuadric(quad)
        
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glPopMatrix()
    
    def draw_bullet_trail(self, bullet):
        """Draw a complete bullet trail with smooth glowing effect"""
        if bullet['type'] == 'cannonball':
            self.draw_cannonball(bullet)
            return
        elif bullet['type'] != 'hitscan':
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
        
        # Initialize player button rect if it doesn't exist
        if not hasattr(self, 'player_button_rect'):
            self.player_button_rect = pygame.Rect(
                self.panel2_button_rect.x,
                self.panel2_button_rect.bottom + margin,
                button_size,
                button_size
            )
        else:
            self.player_button_rect.x = self.panel2_button_rect.x
            self.player_button_rect.y = self.panel2_button_rect.bottom + margin
            self.player_button_rect.width = button_size
            self.player_button_rect.height = button_size
        
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
        self.is_player_button_hovered = hasattr(self, 'player_button_rect') and self.player_button_rect.collidepoint(mouse_x, mouse_y)
        
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
        
        # Draw player model button background (stick figure) - below panel2 button
        player_button_color = (0.2, 0.25, 0.3) if hasattr(self, 'is_player_button_hovered') and self.is_player_button_hovered else (0.15, 0.2, 0.25)
        glColor3f(*player_button_color)
        glVertex2f(self.player_button_rect.left, self.player_button_rect.top)
        glVertex2f(self.player_button_rect.right, self.player_button_rect.top)
        glVertex2f(self.player_button_rect.right, self.player_button_rect.bottom)
        glVertex2f(self.player_button_rect.left, self.player_button_rect.bottom)
        glEnd()
        
        # Draw stick figure icon (simplified)
        icon_size = min(self.player_button_rect.width, self.player_button_rect.height) * 0.6
        icon_x = self.player_button_rect.centerx
        icon_y = self.player_button_rect.centery
        
        # Set icon color (white for visibility)
        glColor3f(1.0, 1.0, 1.0)
        glLineWidth(2.0)
        
        # Draw head (circle)
        glBegin(GL_LINE_LOOP)
        for i in range(32):
            angle = 2.0 * 3.1415926 * i / 32
            dx = math.cos(angle) * (icon_size * 0.15)
            dy = math.sin(angle) * (icon_size * 0.15)
            glVertex2f(icon_x + dx, icon_y - icon_size * 0.4 + dy)
        glEnd()
        
        # Draw body (line from head to mid-body)
        glBegin(GL_LINES)
        glVertex2f(icon_x, icon_y - icon_size * 0.25)  # Bottom of head
        glVertex2f(icon_x, icon_y + icon_size * 0.1)   # Mid-body
        glEnd()
        
        # Draw arms
        glBegin(GL_LINES)
        # Left arm
        glVertex2f(icon_x - icon_size * 0.2, icon_y - icon_size * 0.1)
        glVertex2f(icon_x + icon_size * 0.2, icon_y - icon_size * 0.1)
        # Legs
        glVertex2f(icon_x, icon_y + icon_size * 0.1)  # Mid-body
        glVertex2f(icon_x - icon_size * 0.15, icon_y + icon_size * 0.35)  # Left leg
        glVertex2f(icon_x, icon_y + icon_size * 0.1)  # Mid-body
        glVertex2f(icon_x + icon_size * 0.15, icon_y + icon_size * 0.35)  # Right leg
        glEnd()
        
        # Reset line width
        glLineWidth(1.0)
        
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
        panel_height = self.display[1]
        
        # Draw panel background
        glColor3f(0.1, 0.1, 0.15)  # Panel color
        glBegin(GL_QUADS)
        glVertex2f(panel_x, 0)
        glVertex2f(panel_x + panel_width, 0)
        glVertex2f(panel_x + panel_width, panel_height)
        glVertex2f(panel_x, panel_height)
        glEnd()
        
        header_height = 40
        
        # Enable scissor test to clip content to the panel
        glEnable(GL_SCISSOR_TEST)
        glScissor(panel_x, 0, panel_width, panel_height)
        
        # Draw structure buttons
        rendered_buttons = self.draw_structure_buttons(panel_x, self.side_panel_scroll_y)
        self.structure_buttons = rendered_buttons
        
        # Disable scissor test
        glDisable(GL_SCISSOR_TEST)
        
        # Draw panel header with label (drawn after content to ensure it's on top)
        glColor3f(0.15, 0.15, 0.2)  # Slightly lighter than panel background
        glBegin(GL_QUADS)
        glVertex2f(panel_x, 0)
        glVertex2f(panel_x + panel_width, 0)
        glVertex2f(panel_x + panel_width, header_height)
        glVertex2f(panel_x, header_height)
        glEnd()
        
        # Draw panel label - matching variable settings style
        self.draw_text("Add Structures", panel_x + 15, 20, 20, (0.9, 0.9, 0.9))
        
        # Calculate scroll bar dimensions
        total_content_height = len(self.structure_presets) * 180  # Approximate height of each button with spacing
        visible_height = panel_height - header_height
        
        # Only show scrollbar if content is taller than the panel
        if total_content_height > visible_height:
            # Calculate scroll bar dimensions
            scrollbar_width = 10
            scrollbar_x = panel_x + panel_width - scrollbar_width - 2
            
            # Calculate scroll bar height based on visible area
            scrollbar_height = max(30, int((visible_height / total_content_height) * visible_height))
            
            # Calculate scroll bar position
            scroll_ratio = self.side_panel_scroll_y / (total_content_height - visible_height)
            scrollbar_y = header_height + scroll_ratio * (visible_height - scrollbar_height)
            
            # Draw scrollbar track
            glColor3f(0.2, 0.2, 0.25)
            glBegin(GL_QUADS)
            glVertex2f(scrollbar_x, header_height)
            glVertex2f(scrollbar_x + scrollbar_width, header_height)
            glVertex2f(scrollbar_x + scrollbar_width, panel_height)
            glVertex2f(scrollbar_x, panel_height)
            glEnd()
            
            # Draw scrollbar thumb
            glColor3f(0.4, 0.4, 0.5)
            glBegin(GL_QUADS)
            glVertex2f(scrollbar_x, scrollbar_y)
            glVertex2f(scrollbar_x + scrollbar_width, scrollbar_y)
            glVertex2f(scrollbar_x + scrollbar_width, scrollbar_y + scrollbar_height)
            glVertex2f(scrollbar_x, scrollbar_y + scrollbar_height)
            glEnd()
            
            # Store scroll bar rect for click detection
            self.scroll_bar_rect = pygame.Rect(scrollbar_x, header_height, scrollbar_width, visible_height)
        else:
            self.side_panel_scroll_y = 0
            self.scroll_bar_rect = None
        
        # Re-enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Restore the projection and modelview matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def draw_text(self, text, x, y, size, color, center=False):
        """Draw text on the screen using Pygame's font renderer"""
        font = pygame.font.Font(None, size)
        text_surface = font.render(text, True, (int(color[0]*255), int(color[1]*255), int(color[2]*255)))
        
        if center:
            x -= text_surface.get_width() // 2
            
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
        
        # Add move speed input field below bullet impulse
        move_speed_y = input_rect.bottom + 30
        
        # Move Speed section
        self.draw_text("Player Move Speed", panel_x + 15, move_speed_y - 25, 16, (0.9, 0.9, 0.9))
        
        # Input field background
        move_speed_rect = pygame.Rect(panel_x + 15, move_speed_y, panel_width - 30, 30)
        
        # Check if input is active
        mouse_x, mouse_y = pygame.mouse.get_pos()
        is_move_speed_hovered = move_speed_rect.collidepoint(mouse_x, mouse_y)
        
        # Initialize move speed attributes if they don't exist
        if not hasattr(self, 'move_speed_active'):
            self.move_speed_active = False
        if not hasattr(self, 'player_move_speed'):
            self.player_move_speed = self.player.move_speed  # Get initial value from player
            
        # Set input background color
        if self.move_speed_active:
            glColor3f(0.15, 0.15, 0.25)  # Active state
        elif is_move_speed_hovered:
            glColor3f(0.18, 0.18, 0.28)  # Hover state
        else:
            glColor3f(0.12, 0.12, 0.22)  # Default state
            
        glBegin(GL_QUADS)
        glVertex2f(move_speed_rect.left, move_speed_rect.top)
        glVertex2f(move_speed_rect.right, move_speed_rect.top)
        glVertex2f(move_speed_rect.right, move_speed_rect.bottom)
        glVertex2f(move_speed_rect.left, move_speed_rect.bottom)
        glEnd()
        
        # Draw input border
        border_color = (0.3, 0.5, 0.8) if self.move_speed_active else (0.2, 0.2, 0.3)
        glColor3f(*border_color)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(move_speed_rect.left, move_speed_rect.top)
        glVertex2f(move_speed_rect.right, move_speed_rect.top)
        glVertex2f(move_speed_rect.right, move_speed_rect.bottom)
        glVertex2f(move_speed_rect.left, move_speed_rect.bottom)
        glEnd()
        
        # Draw input text
        text_x = move_speed_rect.left + 8
        text_y = move_speed_rect.centery - 8
        
        # Get the display text from move_speed_text if it exists, otherwise use player_move_speed
        if hasattr(self, 'move_speed_text'):
            display_text = self.move_speed_text
            # If empty but active, show empty (user is deleting)
            if not display_text and self.move_speed_active:
                display_text = ''
            # If empty and not active, show default
            elif not display_text:
                display_text = str(int(self.player_move_speed))
        else:
            # Fallback to current player move speed
            display_text = str(int(self.player_move_speed))
        
        # Draw the text
        self.draw_text(display_text, text_x, text_y, 16, (1.0, 1.0, 1.0))
        
        # Draw cursor if active
        if self.move_speed_active and int(time.time() * 2) % 2 == 0:
            # Calculate cursor position based on actual text width
            font = pygame.font.Font(None, 16)
            text_surface = font.render(display_text, True, (255, 255, 255))
            cursor_x = text_x + text_surface.get_width() + 1
            
            # Draw a vertical cursor line at the end of the text
            glColor3f(1.0, 1.0, 1.0)
            glLineWidth(2.0)
            glBegin(GL_LINES)
            glVertex2f(cursor_x, text_y + 2)
            glVertex2f(cursor_x, text_y + 14)
            glEnd()
        
        # Store input rect for click detection
        self.move_speed_rect = move_speed_rect
        
        # Re-enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def draw_input_field(self, x, y, width, height, label, value, field_id, is_active):
        # Draw label (centered vertically with the input field)
        label_y = y + (height // 2) - 6  # Half the height of the input field minus half the text height
        self.draw_text(label, x - 10, label_y, 12, (0.9, 0.9, 0.9))
        
        # Draw input background
        glColor3f(0.2, 0.2, 0.25) if not is_active else glColor3f(0.15, 0.15, 0.25)
        glBegin(GL_QUADS)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glVertex2f(x + width, y + height)
        glVertex2f(x, y + height)
        glEnd()
        
        # Draw border
        border_color = (0.3, 0.5, 0.8) if is_active else (0.2, 0.2, 0.3)
        glColor3f(*border_color)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glVertex2f(x + width, y + height)
        glVertex2f(x, y + height)
        glEnd()
        
        # Get the display text
        if is_active and self.active_input_field == field_id:
            display_text = getattr(self, f"{field_id}_text", str(value))
            if not display_text:  # If empty, show 0
                display_text = '0'
        else:
            # Format the value with decimal places only if needed
            if float(value) == int(float(value)):
                display_text = str(int(float(value)))
            else:
                display_text = f"{float(value):.1f}"
        
        # Draw the text
        self.draw_text(display_text, x + 5, y + 3, 12, (1.0, 1.0, 1.0))
        
        # Draw cursor if active
        if is_active and self.active_input_field == field_id and int(time.time() * 2) % 2 == 0:
            font = pygame.font.Font(None, 12)
            text_surface = font.render(display_text, True, (255, 255, 255))
            cursor_x = x + 5 + text_surface.get_width() + 1
            glColor3f(1.0, 1.0, 1.0)
            glLineWidth(1.5)
            glBegin(GL_LINES)
            glVertex2f(cursor_x, y + 3)
            glVertex2f(cursor_x, y + 15)
            glEnd()
        
        # Store the rect for click detection
        input_rect = pygame.Rect(x, y, width, height)
        setattr(self, f"{field_id}_rect", input_rect)
        
        return input_rect

    def draw_structure_buttons(self, panel_x, scroll_y=0):
        panel_width = 450  # Increased width to fit inputs on the right
        margin = 15
        button_size = 60
        spacing = 15  # Spacing between buttons
        text_spacing = 8  # Spacing between button and text
        button_x = panel_x + margin
        input_x = button_x + button_size + 20  # Position inputs to the right of buttons
        button_y = 60 - scroll_y  # Start below header, account for scroll
        buttons_per_row = 1  # One button per row
        button_with_text_height = max(button_size, 100)  # Ensure enough height for inputs
        
        # Store the current scissor test state
        scissor_enabled = glIsEnabled(GL_SCISSOR_TEST)
        if not scissor_enabled:
            glEnable(GL_SCISSOR_TEST)
            
        # Set scissor to panel area to clip content
        glScissor(panel_x, 0, panel_width, self.display[1])

        mouse_x, mouse_y = pygame.mouse.get_pos()
        # Adjust mouse y for scrolling when checking button collisions
        mouse_y_scrolled = mouse_y + scroll_y
        font = pygame.font.Font(None, 18)
        buttons = []

        for i, preset in enumerate(self.structure_presets):
            # Calculate position based on button index
            col = i % buttons_per_row
            row = i // buttons_per_row
            
            # Calculate button position
            x = button_x
            y = button_y + (button_with_text_height + spacing) * row
            
            # Button rectangle (just the clickable/image area)
            rect = pygame.Rect(x, y, button_size, button_size)
            
            # Input area rectangle (to the right of the button)
            input_rect = pygame.Rect(input_x, y, panel_width - input_x - margin, button_size)
            is_hovered = rect.collidepoint(mouse_x, mouse_y_scrolled)

            is_active = self.active_structure and preset['name'] == self.active_structure['name']
            button_color = self.colors['cube_button_hover'] if (is_hovered or is_active) else self.colors['cube_button']

            # Draw button background
            glColor3f(*button_color)
            glBegin(GL_QUADS)
            glVertex2f(rect.left, rect.top)
            glVertex2f(rect.right, rect.top)
            glVertex2f(rect.right, rect.bottom)
            glVertex2f(rect.left, rect.bottom)
            glEnd()

            # Draw button image/icon
            image_surface = self.get_structure_thumbnail(preset)
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

            # Prepare button label text
            if is_active and self.placing_cube:
                label_text = "Click to place"
            else:
                # Split long names to fit better
                name = preset['name']
                if len(name) > 8:  # Adjust this number based on your button width
                    # Try to split at spaces
                    words = name.split(' ')
                    if len(words) > 1:
                        # Try to split into two lines
                        mid = len(words) // 2
                        label_text = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                    else:
                        # For single long words, just truncate
                        label_text = name[:8] + '...'
                else:
                    label_text = name

            # Render and draw the text
            if '\n' in label_text:
                # Handle multi-line text
                lines = label_text.split('\n')
                for line_num, line in enumerate(lines):
                    text_surface = font.render(line, True, (255, 255, 255))
                    text_data = pygame.image.tostring(text_surface, 'RGBA', True)
                    text_width = text_surface.get_width()
                    
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    text_x = rect.left + (button_size - text_width) // 2  # Center text
                    text_y = rect.bottom + 8 + (text_surface.get_height() * line_num)
                    # Only draw if the text is within the visible area
                    if y + button_size + 8 + (text_surface.get_height() * (line_num + 1)) > 0 and y < self.display[1]:
                        glRasterPos2i(text_x, text_y + text_surface.get_height())
                        glDrawPixels(text_surface.get_width(), text_surface.get_height(), 
                                   GL_RGBA, GL_UNSIGNED_BYTE, text_data)
                    glDisable(GL_BLEND)
            else:
                # Single line of text
                text_surface = font.render(label_text, True, (255, 255, 255))
                text_data = pygame.image.tostring(text_surface, 'RGBA', True)
                text_width = text_surface.get_width()
                
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                text_x = rect.left + (button_size - text_width) // 2  # Center text
                text_y = rect.bottom + 8
                # Only draw if the text is within the visible area
                if y + button_size + 8 + text_surface.get_height() > 0 and y < self.display[1]:
                    glRasterPos2i(text_x, text_y + text_surface.get_height())
                    glDrawPixels(text_surface.get_width(), text_surface.get_height(), 
                               GL_RGBA, GL_UNSIGNED_BYTE, text_data)
                glDisable(GL_BLEND)

            # Input fields for cluster structures
            if preset.get('generator') == 'cluster':
                cluster_config = preset.get('cluster', {})
                
                # Initialize input fields if they don't exist
                if 'input_values' not in preset:
                    preset['input_values'] = {
                        'size_x': str(cluster_config.get('size', [0.1, 0.1, 0.1])[0]),
                        'size_y': str(cluster_config.get('size', [0.1, 0.1, 0.1])[1]),
                        'size_z': str(cluster_config.get('size', [0.1, 0.1, 0.1])[2]),
                        'grid_x': str(cluster_config.get('grid_count', (1, 1, 1))[0]),
                        'grid_y': str(cluster_config.get('grid_count', (1, 1, 1))[1]),
                        'grid_z': str(cluster_config.get('grid_count', (1, 1, 1))[2])
                    }
                
                # Calculate positions for input fields to the right of the button
                input_start_x = input_x
                input_y = rect.y + 5
                input_width = 60
                input_height = 22
                input_spacing = 5
                
                # Draw size inputs in a column
                self.draw_text("Size", input_start_x, input_y - 15, 14, (0.9, 0.9, 0.9))
                
                # Size X input
                size_x_rect = self.draw_input_field(
                    input_start_x, input_y, input_width, input_height,
                    "X", preset['input_values']['size_x'],
                    f"{preset['name']}_size_x", 
                    self.active_input_field == f"{preset['name']}_size_x"
                )
                
                # Size Y input
                size_y_rect = self.draw_input_field(
                    input_start_x, input_y + input_height + input_spacing, input_width, input_height,
                    "Y", preset['input_values']['size_y'],
                    f"{preset['name']}_size_y", 
                    self.active_input_field == f"{preset['name']}_size_y"
                )
                
                # Size Z input
                size_z_rect = self.draw_input_field(
                    input_start_x, input_y + (input_height + input_spacing) * 2, input_width, input_height,
                    "Z", preset['input_values']['size_z'],
                    f"{preset['name']}_size_z", 
                    self.active_input_field == f"{preset['name']}_size_z"
                )
                
                # Draw grid count inputs in a second column
                grid_start_x = input_start_x + input_width + 20
                self.draw_text("Grid", grid_start_x, input_y - 15, 14, (0.9, 0.9, 0.9))
                
                # Grid X input
                grid_x_rect = self.draw_input_field(
                    grid_start_x, input_y, input_width, input_height,
                    "X", preset['input_values']['grid_x'],
                    f"{preset['name']}_grid_x", 
                    self.active_input_field == f"{preset['name']}_grid_x"
                )
                
                # Grid Y input
                grid_y_rect = self.draw_input_field(
                    grid_start_x, input_y + input_height + input_spacing, input_width, input_height,
                    "Y", preset['input_values']['grid_y'],
                    f"{preset['name']}_grid_y", 
                    self.active_input_field == f"{preset['name']}_grid_y"
                )
                
                # Grid Z input
                grid_z_rect = self.draw_input_field(
                    grid_start_x, input_y + (input_height + input_spacing) * 2, input_width, input_height,
                    "Z", preset['input_values']['grid_z'],
                    f"{preset['name']}_grid_z", 
                    self.active_input_field == f"{preset['name']}_grid_z"
                )
                
                # Update the button height to include inputs
                button_height = grid_z_rect.bottom - rect.y + 10
                button_rect = pygame.Rect(rect.x, rect.y, rect.width, button_height)
            else:
                # For non-cluster structures, just use the normal button size
                button_rect = pygame.Rect(rect.x, rect.y, rect.width, button_size + 30)
            
            buttons.append((button_rect, preset))
            
            # Draw a border around the active button
            if is_active:
                glLineWidth(2.0)
                glColor3f(1.0, 1.0, 1.0)  # White border
                glBegin(GL_LINE_LOOP)
                glVertex2f(rect.left - 2, rect.top - 2)
                glVertex2f(rect.right + 2, rect.top - 2)
                glVertex2f(rect.right + 2, rect.bottom + 2)
                glVertex2f(rect.left - 2, rect.bottom + 2)
                glEnd()
                glLineWidth(1.0)

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
        
        # Apply rotation around Y axis
        glRotatef(self.cube_rotation, 0, 1, 0)

        # Always use the original dimensions, rotation is handled by the modelview matrix
        hx = self.active_total_size[0] / 2.0
        hy = self.active_total_size[1] / 2.0
        hz = self.active_total_size[2] / 2.0
        
        # Draw semi-transparent fill
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

    def draw_player_panel(self):
        """Draw the player model selection panel with thumbnails."""
        if not hasattr(self, 'show_player_panel') or not self.show_player_panel:
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
        button_bg_width = 60
        panel_width = 300  # Wider panel for thumbnails
        panel_x = self.display[0] - panel_width - button_bg_width
        
        # Panel background
        glColor3f(0.1, 0.1, 0.15)
        glBegin(GL_QUADS)
        glVertex2f(panel_x, 0)
        glVertex2f(panel_x + panel_width, 0)
        glVertex2f(panel_x + panel_width, self.display[1])
        glVertex2f(panel_x, self.display[1])
        glEnd()
        
        # Panel header
        header_height = 40
        glColor3f(0.15, 0.15, 0.2)
        glBegin(GL_QUADS)
        glVertex2f(panel_x, 0)
        glVertex2f(panel_x + panel_width, 0)
        glVertex2f(panel_x + panel_width, header_height)
        glVertex2f(panel_x, header_height)
        glEnd()
        
        # Panel title
        self.draw_text("Player Models", panel_x + 15, 20, 20, (0.9, 0.9, 0.9))
        
        # Get player models from the model library
        player_models = self.model_lib.get_player_models()
        
        # Calculate button layout - one button per row with larger buttons
        button_width = panel_width - 30  # Full width of panel minus margins
        button_height = 120  # Taller buttons for better visibility
        button_margin = 15  # Increased margin for better spacing
        thumbnail_size = 80  # Larger thumbnail size
        start_x = panel_x + 15
        start_y = header_height + 15
        
        # Clear and update player buttons
        self.player_buttons = []
        
        # Draw player model buttons - one per row
        for i, model_info in enumerate(player_models):
            y = start_y + (button_height + button_margin) * i
            
            # Store button rect for click detection
            button_rect = pygame.Rect(start_x, y, button_width, button_height)
            model_name = model_info['name']
            self.player_buttons.append((model_name, button_rect))
            
            # Button background (highlight if hovered or selected)
            mouse_x, mouse_y = pygame.mouse.get_pos()
            is_hovered = button_rect.collidepoint(mouse_x, mouse_y)
            is_selected = (self.currPlayerModel.metadata['name'] == model_name)
            
            if is_selected:
                glColor3f(0.2, 0.4, 0.6)  # Blue for selected
            elif is_hovered:
                glColor3f(0.3, 0.3, 0.4)  # Light gray for hover
            else:
                glColor3f(0.2, 0.2, 0.25)  # Dark gray for normal
            
            # Draw button background with rounded corners effect
            glBegin(GL_QUADS)
            glVertex2f(start_x, y)
            glVertex2f(start_x + button_width, y)
            glVertex2f(start_x + button_width, y + button_height)
            glVertex2f(start_x, y + button_height)
            glEnd()
            
            # Draw thumbnail if available
            if 'thumbnail' in model_info and model_info['thumbnail']:
                try:
                    # Load thumbnail if not already loaded
                    if 'thumbnail_surface' not in model_info:
                        thumb_path = model_info['thumbnail']
                        if os.path.exists(thumb_path):
                            # Load and scale the thumbnail
                            thumb_surface = pygame.image.load(thumb_path).convert_alpha()
                            # Flip the image vertically to fix upside-down issue
                            thumb_surface = pygame.transform.flip(thumb_surface, False, True)
                            # Scale to fit while maintaining aspect ratio
                            thumb_rect = thumb_surface.get_rect()
                            scale = min(thumbnail_size / thumb_rect.width, thumbnail_size / thumb_rect.height)
                            # Increase thumbnail size by 50%
                            scale *= 1.5
                            new_size = (int(thumb_rect.width * scale), int(thumb_rect.height * scale))
                            thumb_surface = pygame.transform.scale(thumb_surface, new_size)
                            model_info['thumbnail_surface'] = thumb_surface
                        else:
                            print(f"Thumbnail not found: {thumb_path}")
                            model_info['thumbnail_surface'] = None
                    
                    # Draw the thumbnail if available
                    if 'thumbnail_surface' in model_info and model_info['thumbnail_surface']:
                        thumb_surface = model_info['thumbnail_surface']
                        thumb_rect = thumb_surface.get_rect()
                        thumb_x = start_x + 10  # 10px from left
                        thumb_y = y + (button_height - thumb_rect.height) // 2  # Vertically centered
                        
                        # Draw the thumbnail using Pygame's blit
                        glPushAttrib(GL_ENABLE_BIT | GL_CURRENT_BIT)
                        glDisable(GL_LIGHTING)
                        glEnable(GL_TEXTURE_2D)
                        glEnable(GL_BLEND)
                        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                        glColor3f(1.0, 1.0, 1.0)  # Ensure full brightness
                        
                        # Create a texture from the surface
                        texture_data = pygame.image.tostring(thumb_surface, 'RGBA', 1)
                        texture_id = glGenTextures(1)
                        glBindTexture(GL_TEXTURE_2D, texture_id)
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, thumb_rect.width, thumb_rect.height, 
                                    0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
                        
                        # Draw textured quad
                        glBegin(GL_QUADS)
                        glTexCoord2f(0, 0); glVertex2f(thumb_x, thumb_y)
                        glTexCoord2f(1, 0); glVertex2f(thumb_x + thumb_rect.width, thumb_y)
                        glTexCoord2f(1, 1); glVertex2f(thumb_x + thumb_rect.width, thumb_y + thumb_rect.height)
                        glTexCoord2f(0, 1); glVertex2f(thumb_x, thumb_y + thumb_rect.height)
                        glEnd()
                        
                        # Clean up
                        glDeleteTextures([texture_id])
                        glPopAttrib()  # Restore previous GL state
                
                except Exception as e:
                    print(f"Error loading thumbnail for {model_name}: {e}")
            
            # Draw model name on the right side of the thumbnail
            name = model_name.replace('_', ' ').title()
            text_x = start_x + thumbnail_size + 50  # 20px right of thumbnail
            text_y = y + (button_height - 20) // 2  # Vertically centered
            self.draw_text(name, text_x, text_y, 16, (1, 1, 1))
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def _load_model_thread(self, model_name):
        """Thread function to load the model in the background."""
        try:
            return self.model_lib.load_model(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def _on_model_loaded(self, model, model_name):
        """Handle model loaded callback in the main thread."""
        try:
            if model is None:
                self.is_loading_model = False
                return
                
            # Unload current model if exists
            if self.currPlayerModel is not None:
                self.currPlayerModel.unload()
            
            # Set up the new model
            self.currPlayerModel = model
            
            # Update camera height based on new model
            if 'camera_height' in self.currPlayerModel.metadata:
                self.camera_height = self.currPlayerModel.metadata['camera_height']
            
            # Update player collision in physics world
            if hasattr(self, 'player_structure_id'):
                # Remove old player collision
                self.physics.remove_structure(self.player_structure_id)
                
                # Add new player collision
                player_mesh_path = self.currPlayerModel.metadata.get('mesh_path')
                player_mesh_obj = None
                if player_mesh_path:
                    from models.obj_loader import OBJ
                    player_mesh_obj = OBJ(player_mesh_path)
                
                self.player_structure_id = self.physics.add_structure(
                    position=[self.player.x, self.player.y, self.player.z],
                    size=self.player_collision_size,
                    mass=0.0,
                    color=(0.0, 0.0, 0.0, 0.0),
                    rotation=[0, 0, 0],
                    fill='player',
                    metadata={'stiff': True, 'kinematic': True},
                    mesh_obj=player_mesh_obj
                )
            
            # Reload audio for the new model
            self.load_audio_from_model()
            
        except Exception as e:
            print(f"Error initializing model {model_name}: {e}")
        finally:
            self.is_loading_model = False
            
    def _load_model_thread(self, model_name):
        """Thread function to load the model in the background."""
        try:
            return self.model_lib.load_model(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None

   def _on_model_loaded(self, model, model_name):
        """Handle model loaded callback in the main thread."""
        try:
            if model is None:
                self.is_loading_model = False
                return

            # Unload current model if exists
            if hasattr(self, 'currPlayerModel') and self.currPlayerModel is not None:
                self.currPlayerModel.unload()

            # Set up the new model
            self.currPlayerModel = model

            # Update camera height based on new model
            if hasattr(self.currPlayerModel, 'metadata') and 'camera_height' in self.currPlayerModel.metadata:
                self.camera_height = self.currPlayerModel.metadata['camera_height']

            # Update player collision in physics world
            if hasattr(self, 'player_structure_id'):
                # Remove old player collision
                self.physics.remove_structure(self.player_structure_id)

                # Add new player collision
                player_mesh_path = self.currPlayerModel.metadata.get('mesh_path') if hasattr(self.currPlayerModel, 'metadata') else None
                player_mesh_obj = None
                if player_mesh_path:
                    from models.obj_loader import OBJ
                    player_mesh_obj = OBJ(player_mesh_path)

                self.player_structure_id = self.physics.add_structure(
                    position=[self.player.x, self.player.y, self.player.z],
                    size=self.player_collision_size,
                    mass=0.0,
                    color=(0.0, 0.0, 0.0, 0.0),
                    rotation=[0, 0, 0],
                    fill='player',
                    metadata={'stiff': True, 'kinematic': True},
                    mesh_obj=player_mesh_obj
                )

            # Reload audio for the new model
            self.load_audio_from_model()

        except Exception as e:
            print(f"Error initializing model {model_name}: {e}")
        finally:
            self.is_loading_model = False

   def switch_player_model(self, model_name):
        """Switch to the specified player model with loading indicator."""
        if self.is_loading_model:
            return  # Don't allow multiple model switches at once
            
        self.is_loading_model = True
        self.loading_text = f"Loading {model_name}..."
        self.loading_start_time = time.time()
        
        # Start loading in a separate thread
        def _load_thread():
            model = self._load_model_thread(model_name)
            # Post an event to the main thread to handle the loaded model
            pygame.event.post(pygame.event.Event(
                pygame.USEREVENT, 
                {"type": "model_loaded", "model": model, "model_name": model_name}
            ))
            
        threading.Thread(target=_load_thread, daemon=True).start()
                    # Remove old player collision
                    self.physics.remove_structure(self.player_structure_id)
                    
                    # Add new player collision
                    player_mesh_path = self.currPlayerModel.metadata.get('mesh_path')
                    player_mesh_obj = None
                    if player_mesh_path:
                        from models.obj_loader import OBJ
                        player_mesh_obj = OBJ(player_mesh_path)
                    
                    self.player_structure_id = self.physics.add_structure(
                        position=[self.player.x, self.player.y, self.player.z],
                        size=self.player_collision_size,
                        mass=0.0,
                        color=(0.0, 0.0, 0.0, 0.0),
                        rotation=[0, 0, 0],
                        fill='player',
                        metadata={'stiff': True, 'kinematic': True},
                        mesh_obj=player_mesh_obj
                    )
                
                # Reload audio for the new model
                self.load_audio_from_model()
                
        except Exception as e:
            print(f"Error switching to model {model_name}: {e}")
    
    def render_loading_screen(self):
        """Render the loading screen with a loading message."""
        # Save current projection and modelview matrices
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable lighting for 2D rendering
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Draw semi-transparent overlay
        glColor4f(0.1, 0.1, 0.1, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(self.display[0], 0)
        glVertex2f(self.display[0], self.display[1])
        glVertex2f(0, self.display[1])
        glEnd()
        
        # Draw loading text with blinking dots
        loading_text = self.loading_text
        if time.time() - self.loading_start_time > 0.5:  # Start showing dots after 0.5 seconds
            dot_count = int((time.time() * 2) % 4)  # Cycle through 0-3 dots
            loading_text += '.' * dot_count
        
        # Center the text
        text_width = len(loading_text) * 10  # Approximate width
        text_x = (self.display[0] - text_width) // 2
        text_y = self.display[1] // 2
        
        # Draw text
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glRasterPos2f(text_x, text_y)
        for char in loading_text:
            pygame.font.init()
            font = pygame.font.SysFont('Arial', 24)
            text_surface = font.render(char, True, (255, 255, 255))
            text_data = pygame.image.tostring(text_surface, 'RGBA', True)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(), 
                        GL_RGBA, GL_UNSIGNED_BYTE, text_data)
            glRasterPos2f(glGetDoublev(GL_CURRENT_RASTER_POSITION)[0] + text_surface.get_width(), text_y)
        
        # Restore OpenGL state
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        # Restore matrices
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def draw_circular_shadow(self, x, z, radius, alpha=0.4):
        """Draw a circular shadow on the ground for an object at (x, z) with given radius."""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Use a dark gray color with transparency
        glColor4f(0.1, 0.1, 0.1, alpha)
        
        # Draw a circle for the shadow
        glPushMatrix()
        glTranslatef(x, 0.015, z)  # Slightly above ground to prevent z-fighting
        
        # Draw a circle using a polygon
        glBegin(GL_POLYGON)
        num_segments = 16
        for i in range(num_segments):
            theta = 2.0 * math.pi * float(i) / float(num_segments)
            dx = radius * math.cos(theta)
            dz = radius * math.sin(theta)
            glVertex3f(dx, 0, dz)
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
        
        # Load default shoot sounds (can be a single sound or a list)
        shoot_paths = metadata.get('shoot_sound')
        if shoot_paths:
            if isinstance(shoot_paths, str):
                shoot_paths = [shoot_paths]
            sounds = []
            for path in shoot_paths:
                sound_file = self.resolve_audio_path(path)
                if sound_file:
                    sound = pygame.mixer.Sound(sound_file)
                    sound.set_volume(0.8)
                    sounds.append(sound)
            
            if sounds:
                if len(sounds) == 1:
                    self.shoot_sound = sounds[0]  # Single sound
                else:
                    self.shoot_sound = sounds  # List of sounds
                self.shoot_channel = pygame.mixer.Channel(1)
            else:
                self.shoot_sound = None
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
            for button_rect, preset in self.structure_buttons:
                if button_rect.collidepoint(mouse_pos):
                    hovered_button = {'rect': button_rect, 'preset': preset}
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
                    self.spawn_structure(self.cube_preview_pos, self.cube_rotation)
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
            
        # Store the current input values if we have an active structure
        if hasattr(self, 'active_structure') and self.active_structure and 'input_values' in self.active_structure:
            current_inputs = {}
            for key in ['size_x', 'size_y', 'size_z', 'grid_x', 'grid_y', 'grid_z']:
                if key in self.active_structure['input_values']:
                    current_inputs[key] = self.active_structure['input_values'][key]
            
            # If we have a name and it's different from current, update the new structure with these values
            if name and name != self.active_structure.get('name'):
                for preset in self.structure_presets:
                    if preset['name'] == name and 'input_values' not in preset:
                        preset['input_values'] = current_inputs
                        break
        
        # Set the new active structure
        if name:
            for idx, preset in enumerate(self.structure_presets):
                if preset['name'] == name:
                    self.current_structure_index = idx
                    self.active_structure = preset
                    # Ensure input_values exists
                    if 'input_values' not in self.active_structure:
                        self.active_structure['input_values'] = {}
                    break
            else:
                self.active_structure = self.structure_presets[self.current_structure_index]
        else:
            self.active_structure = self.structure_presets[self.current_structure_index]
            # Ensure input_values exists
            if 'input_values' not in self.active_structure:
                self.active_structure['input_values'] = {}

        if self.active_structure and self.active_structure.get('generator') == 'cluster':
            cluster = self.active_structure['cluster']
            input_values = self.active_structure.get('input_values', {})
            
            # Get size from input values or fall back to defaults
            size_x = float(input_values.get('size_x', cluster.get('size', [0.1, 0.1, 0.1])[0]))
            size_y = float(input_values.get('size_y', cluster.get('size', [0.1, 0.1, 0.1])[1]))
            size_z = float(input_values.get('size_z', cluster.get('size', [0.1, 0.1, 0.1])[2]))
            self.active_cube_size = [size_x, size_y, size_z]
            
            # Get grid counts from input values or fall back to defaults
            default_grid = cluster.get('grid_count', (1, 1, 1))
            grid_x = int(float(input_values.get('grid_x', default_grid[0])))
            grid_y = int(float(input_values.get('grid_y', default_grid[1])))
            grid_z = int(float(input_values.get('grid_z', default_grid[2])))
            grid_counts = (grid_x, grid_y, grid_z)
            
            self.active_total_size = [self.active_cube_size[i] * grid_counts[i] for i in range(3)]
            self.cube_size = max(self.active_total_size)
            
            # Update the cluster config with the current values
            cluster['size'] = [size_x, size_y, size_z]
            cluster['grid_count'] = grid_counts
        
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

    def spawn_structure(self, position, rotation=0):
        if not self.active_structure:
            return
            
        # Reset rotation after placement
        self.cube_rotation = 0
        # Convert rotation to radians for physics
        rotation_rad = math.radians(rotation)
        # Create rotation quaternion for y-axis rotation
        rotation_quat = [0, math.sin(rotation_rad/2), 0, math.cos(rotation_rad/2)]

        # Check for nearby structures to prevent interlocking
        nearby_objects = self.physics.check_nearby_objects(position, radius=2.0)
        if nearby_objects:
            # If there are nearby objects, adjust position slightly
            position = list(position)
            position[1] += 0.1  # Move up slightly to prevent interlocking
            
        generator = self.active_structure.get('generator')
        if generator == 'cluster':
            # Update the structure's rotation in its metadata
            structure_config = self.active_structure['cluster'].copy()
            structure_config['rotation'] = rotation_quat
            
            # Add input values to the config if they exist
            if hasattr(self, 'input_values'):
                structure_config['input_values'] = self.input_values.copy()
                
            self.spawn_cluster_structure(position, structure_config)

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
        # Override size and grid_count with input values if available
        if 'input_values' in config:
            try:
                config = config.copy()
                config['cluster'] = config.get('cluster', {}).copy()
                
                # Get size from input fields
                size_x = float(config['input_values'].get('size_x', 0.1))
                size_y = float(config['input_values'].get('size_y', 0.1))
                size_z = float(config['input_values'].get('size_z', 0.1))
                config['cluster']['size'] = [size_x, size_y, size_z]
                
                # Get grid count from input fields
                grid_x = int(float(config['input_values'].get('grid_x', 1)))
                grid_y = int(float(config['input_values'].get('grid_y', 1)))
                grid_z = int(float(config['input_values'].get('grid_z', 1)))
                config['cluster']['grid_count'] = (grid_x, grid_y, grid_z)
                
            except (ValueError, TypeError) as e:
                print(f"Error parsing input values: {e}")
                # Fall back to default values if parsing fails
                config['cluster']['size'] = config['cluster'].get('size', [0.1, 0.1, 0.1])
                config['cluster']['grid_count'] = config['cluster'].get('grid_count', (1, 1, 1))
        
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
        half_size = [s / 2.0 for s in total_size]
        
        # Get rotation quaternion if it exists
        rotation_quat = config.get('rotation', [0, 0, 0, 1])
        
        # Initialize rotation angle and matrix
        rotation_angle = 0
        rotation_matrix = None
        
        # Process rotation if quaternion is provided
        if isinstance(rotation_quat, (list, tuple)) and len(rotation_quat) == 4:
            # Quaternion is [x, y, z, w] format
            x, y, z, w = rotation_quat
            rotation_angle = 2 * math.acos(w)  # Angle in radians
            if y < 0:
                rotation_angle = -rotation_angle  # Handle direction of rotation
                
            # Calculate rotation matrix from quaternion
            xx = x * x
            xy = x * y
            xz = x * z
            xw = x * w
            yy = y * y
            yz = y * z
            yw = y * w
            zz = z * z
            zw = z * w
            
            rotation_matrix = [
                1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw),
                2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw),
                2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)
            ]
        
        # Load textures if needed
        if fill_type == 'small_bricks' and not hasattr(self, 'small_bricks_textures'):
            self._load_textures('textures/small_bricks', 'small_bricks_textures')
        elif fill_type == 'rust_metal' and not hasattr(self, 'rust_metal_textures'):
            self._load_textures('textures/rust_metal', 'rust_metal_textures')
        elif fill_type == 'concrete' and not hasattr(self, 'concrete_textures'):
            self._load_textures('textures/concrete', 'concrete_textures')
        
        # Calculate center of the entire cluster
        center_x = position[0]
        center_z = position[2]
        
        # Generate cubes in a 3D grid
        for ix in range(grid_counts[0]):
            for iy in range(grid_counts[1]):
                for iz in range(grid_counts[2]):
                    # Calculate position in the grid (before rotation)
                    x = (ix * cube_size[0]) - half_size[0] + (cube_size[0] / 2.0)
                    y = (iy * cube_size[1]) - half_size[1] + (cube_size[1] / 2.0)
                    z = (iz * cube_size[2]) - half_size[2] + (cube_size[2] / 2.0)
                    
                    # Store the original position for rotation
                    original_pos = [x, y, z]
                    
                    # If we have a rotation matrix, apply it to the position
                    if rotation_matrix is not None:
                        # Apply rotation matrix to the position
                        x = (rotation_matrix[0] * original_pos[0] + 
                             rotation_matrix[1] * original_pos[1] + 
                             rotation_matrix[2] * original_pos[2])
                        y = (rotation_matrix[3] * original_pos[0] + 
                             rotation_matrix[4] * original_pos[1] + 
                             rotation_matrix[5] * original_pos[2])
                        z = (rotation_matrix[6] * original_pos[0] + 
                             rotation_matrix[7] * original_pos[1] + 
                             rotation_matrix[8] * original_pos[2])
                    
                    # Calculate final position in world space
                    cube_position = [
                        center_x + x,
                        position[1] + y,
                        center_z + z
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
                    if 'shard_config' in config:
                        metadata['shard_config'] = config['shard_config']
                    
                    # Add to physics world with fill type, texture ID, and metadata
                    mesh_obj = None
                    if 'mesh_path' in config:
                        try:
                            from models.obj_loader import OBJ
                            mesh_obj = OBJ(config['mesh_path'])
                        except Exception as e:
                            mesh_obj = None
                    # For physics, we'll use the same rotation for all cubes in the cluster
                    # This ensures they maintain their relative orientations
                    rotation = rotation_quat if hasattr(self.physics, 'supports_quaternions') and self.physics.supports_quaternions else [0, rotation_angle, 0]
                    
                    cube_id = self.physics.add_structure(
                        position=cube_position,
                        size=cube_size,
                        mass=0.0 if config.get('stiff', False) else mass,  # Set mass to 0 for stiff structures
                        color=base_color,
                        rotation=rotation,
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
        
        # Draw crosshair with equal length lines in all directions
        size = 5  # Length of each crosshair arm
        center_x = int(self.display[0] // 2) + 0.5  # Add 0.5 to align to pixel center
        center_y = int(self.display[1] // 2) + 0.5  # Add 0.5 to align to pixel center
        
        glColor3f(1.0, 1.0, 1.0)  # White color
        glLineWidth(1.0)
        glBegin(GL_LINES)
        # Horizontal line (left to right)
        glVertex2f(center_x - size, center_y)
        glVertex2f(center_x + size + 1, center_y)  # +1 to ensure equal length
        # Vertical line (top to bottom)
        glVertex2f(center_x, center_y - size)
        glVertex2f(center_x, center_y + size + 1)  # +1 to ensure equal length
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
        # Reset scroll if panel is closed
        if not self.show_side_panel:
            self.side_panel_scroll_y = 0
            
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
                # Handle input for active structure parameter fields
                if hasattr(self, 'active_input_field') and self.active_input_field:
                    field_id = self.active_input_field
                    text_attr = f"{field_id}_text"
                    
                    if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        # Save the input when Enter is pressed
                        if event.key == pygame.K_RETURN and hasattr(self, text_attr):
                            # Find which preset this input belongs to
                            for preset in self.structure_presets:
                                if field_id.startswith(preset['name']):
                                    parts = field_id.split('_')
                                    preset_name = parts[0]
                                    param_type = parts[1]  # 'size' or 'grid'
                                    dim = parts[2]  # 'x', 'y', or 'z'
                                    
                                    # Get the text value and validate it
                                    text_value = getattr(self, text_attr, '0')
                                    try:
                                        # Convert to float for size, int for grid
                                        if param_type == 'size':
                                            new_value = max(0.1, min(10.0, float(text_value or '0.1')))
                                            preset['input_values'][f"{param_type}_{dim}"] = new_value
                                        else:  # grid
                                            new_value = max(1, min(100, int(float(text_value or '1'))))
                                            preset['input_values'][f"{param_type}_{dim}"] = new_value
                                    except (ValueError, TypeError):
                                        pass  # Keep old value if conversion fails
                                    break
                        
                        # Clean up and deactivate
                        if hasattr(self, text_attr):
                            delattr(self, text_attr)
                        self.active_input_field = None
                    
                    elif event.key == pygame.K_BACKSPACE:
                        # Handle backspace
                        if not hasattr(self, text_attr):
                            # Initialize with current value if it doesn't exist
                            for preset in self.structure_presets:
                                if field_id.startswith(preset['name']):
                                    parts = field_id.split('_')
                                    param_type = parts[1]
                                    dim = parts[2]
                                    current_val = preset['input_values'].get(f"{param_type}_{dim}", "1")
                                    setattr(self, text_attr, str(int(float(current_val)) if float(current_val).is_integer() else current_val))
                                    break
                        
                        if hasattr(self, text_attr):
                            current_text = getattr(self, text_attr)
                            if len(current_text) > 0:
                                new_text = current_text[:-1]
                                setattr(self, text_attr, new_text if new_text else '0')
                    
                    elif event.unicode.isdigit() or (event.unicode == '.' and '.' not in getattr(self, text_attr, '')):
                        # Handle digit or decimal point input
                        if not hasattr(self, text_attr):
                            # Initialize with current value if it doesn't exist
                            for preset in self.structure_presets:
                                if field_id.startswith(preset['name']):
                                    parts = field_id.split('_')
                                    param_type = parts[1]
                                    dim = parts[2]
                                    current_val = preset['input_values'].get(f"{param_type}_{dim}", "1")
                                    setattr(self, text_attr, str(int(float(current_val)) if float(current_val).is_integer() else current_val))
                                    break
                        
                        current_text = getattr(self, text_attr, '')
                        # Limit to 6 characters total (including decimal point)
                        if len(current_text) < 6:
                            # Handle leading zero
                            if current_text == '0' and event.unicode != '.':
                                setattr(self, text_attr, event.unicode)
                            else:
                                setattr(self, text_attr, current_text + event.unicode)
                
                # Handle move speed input
                elif hasattr(self, 'move_speed_active') and self.move_speed_active:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        self.move_speed_active = False
                        # Clean up text buffer on exit
                        if hasattr(self, 'move_speed_text'):
                            try:
                                # If empty or invalid, use the previous valid value
                                if not self.move_speed_text or not self.move_speed_text.isdigit():
                                    self.move_speed_text = str(int(self.player_move_speed))
                                
                                new_speed = int(self.move_speed_text)
                                # Clamp the value between 10 and 50
                                new_speed = max(10, min(50, new_speed))
                                self.player_move_speed = new_speed
                                self.player.move_speed = float(new_speed)
                                self.move_speed_text = str(new_speed)  # Update with clamped value
                            except (ValueError, AttributeError):
                                # Fall back to current speed on any error
                                self.move_speed_text = str(int(self.player_move_speed))
                            del self.move_speed_text
                        self.move_speed_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        # Initialize text buffer if needed
                        if not hasattr(self, 'move_speed_text'):
                            self.move_speed_text = str(int(self.player_move_speed))
                        # Allow deleting all digits - just remove the last character
                        self.move_speed_text = self.move_speed_text[:-1]
                    elif event.unicode.isdigit():
                        # Initialize text buffer if needed
                        if not hasattr(self, 'move_speed_text'):
                            self.move_speed_text = ''
                        
                        # Allow up to 2 digits
                        if len(self.move_speed_text) < 2:
                            # If this is the first digit and it's 0, replace it with the new digit
                            if not self.move_speed_text and event.unicode == '0':
                                self.move_speed_text = '10'  # Set minimum valid value
                            else:
                                # Add the new digit
                                self.move_speed_text += event.unicode
                            
                            # Validate the new value
                            try:
                                new_speed = int(self.move_speed_text)
                                # If exceeds max, cap at 50
                                if new_speed > 50:
                                    self.move_speed_text = '50'
                                # If less than min, set to min (but allow typing first digit)
                                elif new_speed < 1 and len(self.move_speed_text) == 1:
                                    self.move_speed_text = '10'
                            except (ValueError, AttributeError):
                                # Fall back to current speed on any error
                                self.move_speed_text = str(int(self.player_move_speed))
                
                # Handle bullet impulse input (existing code)
                elif hasattr(self, 'bullet_impulse_active') and self.bullet_impulse_active:
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
                        if self.placing_cube:  # Rotate if in placement mode
                            self.cube_rotation = (self.cube_rotation + 45) % 360  # Rotate 45 degrees around Y-axis
                    elif event.key == pygame.K_m:  # Add M key to toggle side panel
                        self.show_side_panel = not self.show_side_panel
                    elif event.key == pygame.K_p:  # P key toggles player panel
                        self.show_player_panel = not self.show_player_panel
                        if self.show_player_panel:
                            self.show_side_panel = False
                            self.show_side_panel2 = False
                            pygame.mouse.set_visible(True)
                            pygame.event.set_grab(False)
                        else:
                            pygame.mouse.set_visible(False)
                            pygame.event.set_grab(True)
            elif event.type == pygame.MOUSEWHEEL:
                # Check for Ctrl+scroll for zooming
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # Handle zoom
                    zoom_speed = 0.2
                    if event.y > 0:  # Scrolling up - zoom in
                        self.camera_distance = max(self.min_zoom, self.camera_distance - zoom_speed)
                    else:  # Scrolling down - zoom out
                        self.camera_distance = min(self.max_zoom, self.camera_distance + zoom_speed)
                # Handle panel scrolling if panel is visible
                elif self.show_side_panel:
                    # Calculate the total content height and visible height
                    total_content_height = len(self.structure_presets) * 180  # Same as in draw_side_panel
                    visible_height = self.display[1] - 40  # Panel height minus header
                    
                    # Only process scrolling if content is taller than the visible area
                    if total_content_height > visible_height:
                        # Scroll up (negative y) or down (positive y)
                        scroll_amount = event.y * 30  # Adjust scroll speed as needed
                        
                        # Update scroll position with bounds checking
                        self.side_panel_scroll_y = max(0, min(
                            self.side_panel_scroll_y - scroll_amount, 
                            total_content_height - visible_height
                        ))
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # Right mouse button
                    self.toggle_mouse_grab()
                if event.button == 1:  # Left mouse button
                    mouse_pos = event.pos
                    input_clicked = False
                    
                    # Check if clicking on move speed input
                    if hasattr(self, 'move_speed_rect') and self.move_speed_rect.collidepoint(mouse_pos):
                        self.move_speed_active = True
                        input_clicked = True
                        # Initialize text buffer if needed
                        if not hasattr(self, 'move_speed_text'):
                            self.move_speed_text = str(int(self.player_move_speed))
                    elif hasattr(self, 'move_speed_active') and self.move_speed_active:
                        # Clicked outside the move speed input while it was active - save the value
                        self.move_speed_active = False
                        if hasattr(self, 'move_speed_text'):
                            try:
                                # If empty or invalid, use the previous valid value
                                if not self.move_speed_text or not self.move_speed_text.isdigit():
                                    self.move_speed_text = str(int(self.player_move_speed))
                                
                                new_speed = int(self.move_speed_text)
                                # Clamp the value between 10 and 50
                                new_speed = max(10, min(50, new_speed))
                                self.player_move_speed = new_speed
                                self.player.move_speed = float(new_speed)
                                self.move_speed_text = str(new_speed)  # Update with clamped value
                            except (ValueError, AttributeError):
                                # Fall back to current speed on any error
                                self.move_speed_text = str(int(self.player_move_speed))
                            del self.move_speed_text
                    
                    # Check if clicking on bullet impulse input
                    if hasattr(self, 'bullet_impulse_rect') and self.bullet_impulse_rect.collidepoint(mouse_pos):
                        self.bullet_impulse_active = True
                        input_clicked = True
                        # Initialize text buffer if needed
                        if not hasattr(self, 'bullet_impulse_text'):
                            self.bullet_impulse_text = str(int(self.bullet_impulse))
                    
                    # Check if clicking on any structure input field
                    if not input_clicked and hasattr(self, 'structure_buttons'):
                        for button_rect, preset in self.structure_buttons:
                            if preset.get('generator') == 'cluster' and 'input_values' in preset:
                                for dim in ['size_x', 'size_y', 'size_z', 'grid_x', 'grid_y', 'grid_z']:
                                    field_id = f"{preset['name']}_{dim}"
                                    if hasattr(self, f"{field_id}_rect"):
                                        field_rect = getattr(self, f"{field_id}_rect")
                                        if field_rect.collidepoint(mouse_pos):
                                            self.active_input_field = field_id
                                            # Initialize text buffer with current value
                                            current_val = preset['input_values'][dim]
                                            setattr(self, f"{field_id}_text", str(int(float(current_val)) if float(current_val).is_integer() else current_val))
                                            input_clicked = True
                                            break
                                if input_clicked:
                                    break
                    
                    if not input_clicked:
                        self.active_input_field = None
                        self.bullet_impulse_active = False
                        self.move_speed_active = False
                        
                        # Check if menu button was clicked
                        if self.menu_button_rect.collidepoint(mouse_pos):
                            self.show_side_panel = not self.show_side_panel
                            if self.show_side_panel:
                                self.show_side_panel2 = False
                                self.show_player_panel = False
                        # Check if second panel button was clicked
                        elif self.panel2_button_rect.collidepoint(event.pos):
                            self.show_side_panel2 = not self.show_side_panel2
                            if self.show_side_panel2:
                                self.show_side_panel = False
                                self.show_player_panel = False
                        # Check if player button was clicked
                        elif hasattr(self, 'player_button_rect') and self.player_button_rect.collidepoint(event.pos):
                            self.show_player_panel = not self.show_player_panel
                            if self.show_player_panel:
                                self.show_side_panel = False
                                self.show_side_panel2 = False
                                pygame.mouse.set_visible(True)
                                pygame.event.set_grab(False)
                            else:
                                pygame.mouse.set_visible(False)
                                pygame.event.set_grab(True)
                        # Check for clicks on player model buttons in the panel
                        elif hasattr(self, 'show_player_panel') and self.show_player_panel and hasattr(self, 'player_buttons'):
                            for model_name, button_rect in self.player_buttons:
                                if button_rect.collidepoint(event.pos):
                                    self.switch_player_model(model_name)
                                    break
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
        """Render the 3D scene."""
        # Clear the screen and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
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
        elif self.show_player_panel:
            self.draw_player_panel()
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
