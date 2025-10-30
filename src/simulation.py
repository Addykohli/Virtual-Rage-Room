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
import asyncio
import json

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.player import Player
from models.obj_loader import OBJ
from physics_world import PhysicsWorld
from ai_description import AIDescriptionHandler

# Constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FOV = 45.0
NEAR_PLANE = 0.1
FAR_PLANE = 1000.0
GRID_SIZE = 20
TILE_SIZE = 1.0

class Simulation:
    def _init_ai_panel_rects(self):
        """Initialize the rects for AI panel elements."""
        panel_width = 700  # Increased width to accommodate the close button
        panel_height = 700  # Increased height to match the panel height in draw_ai_panel
        panel_x = (self.display[0] - panel_width) // 2
        panel_y = (self.display[1] - panel_height) // 2
        
        # Input area rect
        input_x = panel_x + 20
        input_y = panel_y + 60
        input_width = panel_width - 40
        input_height = 200
        self.ai_input_rect = pygame.Rect(input_x, input_y, input_width, input_height)
        
        # Close button rect (top-right corner)
        close_btn_size = 30
        close_btn_margin = 20
        self.ai_close_rect = pygame.Rect(
            panel_x + panel_width - close_btn_size - close_btn_margin,
            panel_y + close_btn_margin,
            close_btn_size,
            close_btn_size
        )
        
        # Generate button rect
        btn_width = 150
        btn_height = 35
        btn_y = panel_y + panel_height - 60
        
        # Generate button
        self.ai_generate_rect = pygame.Rect(
            panel_x + (panel_width - btn_width) // 2,  # Center the button
            btn_y,
            btn_width,
            btn_height
        )
        
        # Buttons
        button_width = 150
        button_height = 35
        button_y = panel_y + panel_height - 60
        
        # Generate button
        generate_x = panel_x + 30
        self.ai_generate_rect = pygame.Rect(generate_x, button_y, button_width, button_height)
        
        # Close button
        close_x = panel_x + panel_width - 30 - button_width
        self.ai_close_rect = pygame.Rect(close_x, button_y, button_width, button_height)

    def __init__(self):
        # Initialize asyncio event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        pygame.init()
        # Loading state
        self.is_loading = False
        self.loading_message = ""
        self.loading_progress = 0
        self.loading_font = pygame.font.SysFont('Arial', 24)
        
        # Create saves directory if it doesn't exist
        self.saves_dir = os.path.join(os.path.dirname(__file__), 'saves')
        os.makedirs(self.saves_dir, exist_ok=True)
        
        # Initialize fonts
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 16)
        
        # Set the Gemini API key first
        self.gemini_api_key = "AIzaSyA0MVNgdYADmTORp9A5lrcORRDbC7DA4LE"
        
        # Initialize AI Description Handler with the API key
        self.ai_handler = AIDescriptionHandler(self.saves_dir, self.gemini_api_key)
        
        # Initialize AI handler
        self.ai_initialized = False
        self.ai_status_text = "Initializing AI..."
        self.init_ai_handler()
        
        self.show_ai_panel = False
        self.ai_input_text = ""
        self.ai_input_cursor_pos = 0  # Track cursor position in text
        self.ai_input_active = False
        self.ai_input_rect = None  # Will be set in draw_ai_panel
        self.ai_generate_rect = None  # Will be set in draw_ai_panel
        self.ai_close_rect = None  # Will be set in draw_ai_panel
        self.ai_status_timer = 180  # Show status for 3 seconds at 60 FPS
        self.ai_generation_future = None
        self.is_fetching_genai = False  # Track if we're currently fetching from GenAI
        self.thumbnails_dir = 'thumbnails'
    def init_ai_handler(self):
        """Initialize the AI handler and verify the API key synchronously"""
        print(f"[Simulation] Initializing AI with API key (length: {len(self.gemini_api_key) if self.gemini_api_key else 0})")
        
        # Make sure we have an API key
        if not self.gemini_api_key:
            error_msg = "No API key provided"
            print(f"[Simulation] {error_msg}")
            self.ai_status_text = error_msg
            self.ai_initialized = False
            return
            
        # Set and verify the API key in one step
        if self.ai_handler.set_api_key(self.gemini_api_key):
            print("[Simulation] AI handler initialized successfully")
            self.ai_status_text = "AI ready!"
            self.ai_initialized = True
        else:
            error_msg = getattr(self.ai_handler, 'last_error', 'Unknown error')
            error_msg = f"Failed to initialize AI: {error_msg}"
            print(f"[Simulation] {error_msg}")
            self.ai_status_text = error_msg
            self.ai_initialized = False
            print(f"[Simulation] {error_msg}")
            self.ai_status_text = f"AI Error: {error_msg}"
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
        
        # Enable lighting and normalize normals for proper lighting calculations
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        
        # Enable multiple light sources for more realistic lighting
        glEnable(GL_LIGHT0)  # Main directional light (sun)
        glEnable(GL_LIGHT1)  # Fill light (ambient fill)
        
        # Set up main directional light (sun-like, from top-right)
        light0_position = [10.0, 15.0, 10.0, 0.0]  # Directional light (w=0)
        light0_diffuse = [0.9, 0.85, 0.8, 1.0]     # Slightly warm white
        light0_specular = [1.0, 0.98, 0.95, 1.0]   # Slightly warm highlights
        light0_ambient = [0.1, 0.1, 0.15, 1.0]     # Dark blue ambient
        
        glLightfv(GL_LIGHT0, GL_POSITION, light0_position)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient)
        
        # Set up fill light (soft ambient from opposite side)
        light1_position = [-5.0, 5.0, -5.0, 0.0]   # Directional fill light
        light1_diffuse = [0.2, 0.2, 0.25, 1.0]     # Very soft blue-grey
        light1_ambient = [0.1, 0.1, 0.1, 1.0]      # Dark ambient
        
        glLightfv(GL_LIGHT1, GL_POSITION, light1_position)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse)
        glLightfv(GL_LIGHT1, GL_AMBIENT, light1_ambient)
        
        # Set up material properties for all objects
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set default material properties (can be overridden per-object)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.3, 0.3, 0.3, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 32.0)
        
        # Enable smooth shading for better lighting interpolation
        glShadeModel(GL_SMOOTH)
        
        # Initialize model library and load all models
        from model_lib import ModelLibrary
        from models.skybox import Skybox
        from models.obj_loader import OBJ
        
        self.model_lib = ModelLibrary()
        self.currPlayerModel = self.model_lib.load_model('Cannon')
        
        # Initialize models dictionary to store loaded models
        self.models = {}
        
        # Load all structure models
        for model_info in self.model_lib.get_environment_models():
            if 'path' in model_info and os.path.exists(model_info['path']):
                try:
                    self.models[model_info['path']] = OBJ(model_info['path'])
                except Exception as e:
                    print(f"[ERROR] Failed to load model {model_info['path']}: {e}")
        
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
        # Store a reference to this simulation instance in physics for sound handling
        self.physics.simulation = self
        
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
        self.max_zoom = 2000.0  # Maximum zoom distance
        self.zoom_speed = 250  # Zoom speed multiplier
        
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
        self.cannon_ball_speed = 60.0  # Speed of cannonballs
        self.cannon_ball_radius = 0.15  # Size of cannonballs
        self.cannon_ball_mass = 49.0  # Mass of cannonballs
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
        
        self.player_button_rect = pygame.Rect(
                self.panel2_button_rect.x,
                self.panel2_button_rect.bottom + margin,
                structure_button_size,
                structure_button_size
            )
        self.show_player_panel = False  # Track player panel visibility


        self.save_panel_button_rect = pygame.Rect(
            self.player_button_rect.x,
            self.player_button_rect.bottom + margin,
            structure_button_size,
            structure_button_size
        )
        self.show_save_panel = False  # Track save panel visibility

        self.ai_panel_button_rect = pygame.Rect(
            self.save_panel_button_rect.x,
            self.save_panel_button_rect.bottom + margin,
            structure_button_size,
            structure_button_size
        )
        self.show_ai_panel = False  # Track AI panel visibility


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
        
        # AI handler is already initialized in __init__
        self.show_ai_panel = False
        
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
            # Load the image with alpha channel
            texture_surface = pygame.image.load(image_path).convert_alpha()
            
            # Convert to RGBA format if not already
            if texture_surface.get_bytesize() != 4:
                texture_surface = texture_surface.convert_alpha()
                
            # Flip the image vertically to match OpenGL's coordinate system
            texture_surface = pygame.transform.flip(texture_surface, False, True)
            
            # Get the raw pixel data
            texture_data = pygame.image.tostring(texture_surface, 'RGBA', 1)
            
            # Generate a texture ID
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            # Set texture parameters for better quality
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            
            # Upload texture data and generate mipmaps
            width, height = texture_surface.get_size()
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, 
                        GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
            
            # Generate mipmaps for better quality when scaled
            glGenerateMipmap(GL_TEXTURE_2D)
            
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
        grid_size = 99
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
    

    def _draw_single_physics_object(self, structure_id, structure, is_transparent=False):
        """Helper method to draw a single physics object with proper transparency handling"""
        # Get position and orientation from physics
        pos = self.physics.get_structure_position(structure_id)
        orn = self.physics.get_structure_orientation(structure_id)
        
        if pos is None or orn is None:
            return
            
        size = structure['size']
        color = structure['color']
        fill_type = structure.get('fill', 'solid')
        
        # Skip rendering for player structures (they're rendered separately)
        if fill_type == 'player':
            return
            
        # Draw circular shadow first (under the object)
        shadow_radius = max(size[0], size[2]) * 0.3
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
        
        # Check if this structure has a model to render
        metadata = structure.get('metadata', {})
        
        if 'model_path' in metadata:
            model_path = metadata['model_path']
            
            if hasattr(self, 'models') and model_path in self.models:
                # Scale the model to fit the collision box
                scale = max(size) * 0.5  # Adjust scale as needed
                glScalef(scale, scale, scale)
                
                # Set color and render the model
                glColor4f(*color)
                self.models[model_path].render()
                
                # Restore matrix and return early since we've handled the rendering
                glPopMatrix()
                return
            else:
                print(f"[RENDER] Model not found or models not loaded: {model_path}")
        
        # Set material color with alpha for transparent objects
        if is_transparent:
            # For transparent objects, ensure we have a 4-component color
            if len(color) == 3:
                color = (color[0], color[1], color[2], 0.5)  # Default alpha if not specified
            glColor4f(*color)
            
            # Enable blending for transparent objects
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(False)  # Disable depth writing for transparent objects
        else:
            glColor4f(*color) if len(color) > 3 else glColor3f(*color)
        
        # Get metadata safely with defaults
        metadata = structure.get('metadata', {})
        is_shard = metadata.get('is_shard_object', False)
        
        # Draw based on texture ID or fill type
        if 'texture_id' in structure and structure['texture_id'] is not None:
            # Use the stored texture ID for this structure
            self.draw_textured_cube(size, structure['texture_id'])
        elif is_shard and fill_type in ['small_bricks', 'rust_metal', 'concrete', 'stone', 'glass', 'tough_glass', 'wood']:
            # For shards, use the stored shard texture or create one if it doesn't exist
            if '_shard_texture' not in metadata:
                base_fill = 'glass' if fill_type == 'tough_glass' else fill_type
                textures = getattr(self, f'{base_fill}_textures', [])
                if textures:
                    metadata['_shard_texture'] = random.choice(textures)
            
            if '_shard_texture' in metadata:
                self.draw_textured_cube(size, metadata['_shard_texture'])
        elif fill_type in ['small_bricks', 'rust_metal', 'concrete', 'stone', 'glass', 'tough_glass', 'wood'] and not is_shard:
            # For non-shard objects with texture types, ensure we have a texture ID
            if 'texture_id' not in structure or structure['texture_id'] is None:
                # Load textures if needed
                if not hasattr(self, f'{fill_type}_textures') or not getattr(self, f'{fill_type}_textures'):
                    self._load_textures(f'textures/{fill_type}', f'{fill_type}_textures')
                
                textures = getattr(self, f'{fill_type}_textures', [])
                if textures:
                    # Store the texture ID for future renders
                    structure['texture_id'] = random.choice(textures)
                    self.draw_textured_cube(size, structure['texture_id'])
                else:
                    # Fall back to solid color if no textures available
                    self.draw_solid_cube(size)
            else:
                self.draw_textured_cube(size, structure['texture_id'])
        elif fill_type == 'tough_glass':
            # For tough_glass, only apply texture to thin faces
            half = [s / 2.0 for s in size]
            
            # Get glass texture - ensure textures are loaded
            if not hasattr(self, 'glass_textures') or not self.glass_textures:
                self._load_textures('textures/glass', 'glass_textures')
            
            textures = getattr(self, 'glass_textures', [])
            texture = random.choice(textures) if textures else None
            
            # Define face normals and their corresponding dimensions
            faces = [
                # Front/Back faces (Z-axis)
                {'normal': (0, 0, 1), 'dims': (0, 1), 'sign': 1},  # Front
                {'normal': (0, 0, -1), 'dims': (0, 1), 'sign': -1},  # Back
                # Top/Bottom faces (Y-axis)
                {'normal': (0, 1, 0), 'dims': (0, 2), 'sign': 1},   # Top
                {'normal': (0, -1, 0), 'dims': (0, 2), 'sign': -1},  # Bottom
                # Left/Right faces (X-axis)
                {'normal': (1, 0, 0), 'dims': (1, 2), 'sign': 1},    # Right
                {'normal': (-1, 0, 0), 'dims': (1, 2), 'sign': -1},  # Left
            ]
            
            # Enable blending for the entire object
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Draw each face
            for face in faces:
                dim1, dim2 = face['dims']
                # Check if this is a thin face (either dimension < 0.2)
                if size[dim1] < 0.2 or size[dim2] < 0.2:
                    # Draw with glass texture if available
                    if texture:
                        glEnable(GL_TEXTURE_2D)
                        glBindTexture(GL_TEXTURE_2D, texture)
                        glColor4f(0.16, 0.51, 0.31, 1)  
                    else:
                        glColor4f(0.7, 0.8, 1.0, 0.7)  # Fallback color if no texture
                else:
                    # Draw as transparent face using the structure's base_color with reduced alpha
                    glDisable(GL_TEXTURE_2D)
                    base_color = structure.get('base_color', (0.5, 0.5, 0.8, 0.3))  # Default fallback color
                    glColor4f(base_color[0], base_color[1], base_color[2], base_color[3] * 0.3)
                
                # Draw the face
                self._draw_face(half, face['normal'], face['sign'])
            
            # Clean up OpenGL state
            glDisable(GL_TEXTURE_2D)
            glDisable(GL_BLEND)
        elif fill_type in ['small_bricks', 'rust_metal', 'concrete', 'stone', 'glass']:
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
        
        # Clean up blending state if this was a transparent object
        if is_transparent:
            glDepthMask(True)
            glDisable(GL_BLEND)
        
        # Restore matrix
        glPopMatrix()
    
    def _draw_face(self, half_size, normal, sign):
        """Helper method to draw a single face of a cube with the given normal and sign"""
        x, y, z = normal
        
        # Determine the vertices based on the normal and sign
        if x != 0:  # Left/Right face
            x_pos = half_size[0] * sign
            v1 = (x_pos, -half_size[1], -half_size[2])
            v2 = (x_pos, -half_size[1], half_size[2])
            v3 = (x_pos, half_size[1], half_size[2])
            v4 = (x_pos, half_size[1], -half_size[2])
        elif y != 0:  # Top/Bottom face
            y_pos = half_size[1] * sign
            v1 = (-half_size[0], y_pos, -half_size[2])
            v2 = (half_size[0], y_pos, -half_size[2])
            v3 = (half_size[0], y_pos, half_size[2])
            v4 = (-half_size[0], y_pos, half_size[2])
        else:  # Front/Back face
            z_pos = half_size[2] * sign
            v1 = (-half_size[0], -half_size[1], z_pos)
            v2 = (half_size[0], -half_size[1], z_pos)
            v3 = (half_size[0], half_size[1], z_pos)
            v4 = (-half_size[0], half_size[1], z_pos)
        
        # Draw the face
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex3f(*v1)
        glTexCoord2f(1, 0)
        glVertex3f(*v2)
        glTexCoord2f(1, 1)
        glVertex3f(*v3)
        glTexCoord2f(0, 1)
        glVertex3f(*v4)
        glEnd()
    
    def draw_physics_objects(self):
        """Draw all physics objects in the scene with proper transparency handling"""
        # First pass: collect and sort transparent objects
        transparent_objects = []
        opaque_objects = []
        
        for structure_id, structure in self.physics.structures.items():
            if not structure.get('visible', True):
                continue
                
            # Check if object is transparent (has alpha < 1.0)
            color = structure.get('color', [0.5, 0.5, 0.5, 1.0])
            is_transparent = len(color) > 3 and color[3] < 0.99
            
            if is_transparent:
                # For transparent objects, calculate distance from camera for sorting
                pos = self.physics.get_structure_position(structure_id)
                if pos is not None:
                    dx = pos[0] - self.player.x
                    dy = pos[1] - self.player.y
                    dz = pos[2] - self.player.z
                    distance = dx*dx + dy*dy + dz*dz  # Squared distance for sorting
                    transparent_objects.append((distance, structure_id, structure))
            else:
                # For opaque objects, just add to the list
                opaque_objects.append((structure_id, structure))
        
        # Sort transparent objects by distance (farthest first for proper blending)
        transparent_objects.sort(reverse=True, key=lambda x: x[0])
        
        # First render all opaque objects
        for structure_id, structure in opaque_objects:
            self._draw_single_physics_object(structure_id, structure, is_transparent=False)
        
        # Then render transparent objects from back to front
        for _, structure_id, structure in transparent_objects:
            self._draw_single_physics_object(structure_id, structure, is_transparent=True)
    
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
        
        # Use the same light direction as our main light (LIGHT0)
        light_dir = [10.0, 15.0, 10.0]  # Match the position of LIGHT0
        length = math.sqrt(sum(x*x for x in light_dir))
        light_dir = [x/length for x in light_dir]  # Normalize the direction
        
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
        
        # Use the same light direction as our main light (LIGHT0)
        light_dir = [10.0, 15.0, 10.0]  # Match the position of LIGHT0
        length = math.sqrt(sum(x*x for x in light_dir))
        light_dir = [x/length for x in light_dir]  # Normalize the direction
        
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
    
    def load_button_textures(self):
        """Load all button textures if not already loaded"""
        if not hasattr(self, 'button_textures'):
            self.button_textures = {}
            
            # Define button texture files and their keys
            button_texture_files = {
                'menu': 'menu_panel_button.png',
                'panel2': 'variable_panel_button.png',
                'player': 'player_panel_button.png',
                'save': 'save_panel_button.png',
                'ai': 'ai_panel_button.png'
            }
            
            # Load each texture
            for key, filename in button_texture_files.items():
                texture_path = os.path.join(self.thumbnails_dir, filename)
                if os.path.exists(texture_path):
                    self.button_textures[key] = self.load_texture(texture_path)
                else:
                    print(f"Warning: Could not find texture {filename}")
                    self.button_textures[key] = None
    
    def draw_textured_button(self, rect, texture_key, is_hovered=False):
        """Draw a button with the specified texture"""
        if texture_key in self.button_textures and self.button_textures[texture_key] is not None:
            # Enable texturing and bind the texture
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.button_textures[texture_key])
            
            # Enable blending for transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Set texture parameters for better quality
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
            # Set color with hover effect - adjust brightness and saturation
            if is_hovered:
                glColor3f(1.4, 1.4, 1.4)  # Brighter when hovered
            else:
                glColor3f(1.1, 1.1, 1.1)  # Slightly brighter than normal
            
            # Draw textured quad with correct texture coordinates
            glBegin(GL_QUADS)
            # Bottom-left (texture bottom-left)
            glTexCoord2f(0, 0)
            glVertex2f(rect.left, rect.top)
            # Bottom-right (texture bottom-right)
            glTexCoord2f(1, 0)
            glVertex2f(rect.right, rect.top)
            # Top-right (texture top-right)
            glTexCoord2f(1, 1)
            glVertex2f(rect.right, rect.bottom)
            # Top-left (texture top-left)
            glTexCoord2f(0, 1)
            glVertex2f(rect.left, rect.bottom)
            glEnd()
            
            # Clean up
            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_2D)
        else:
            # Fallback to colored rectangle if texture not available
            if is_hovered:
                glColor3f(0.3, 0.3, 0.3)
            else:
                glColor3f(0.2, 0.2, 0.2)
            
            glBegin(GL_QUADS)
            glVertex2f(rect.left, rect.top)
            glVertex2f(rect.right, rect.top)
            glVertex2f(rect.right, rect.bottom)
            glVertex2f(rect.left, rect.bottom)
            glEnd()
    
    def draw_menu_buttons(self):
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
        self.is_player_button_hovered = hasattr(self, 'player_button_rect') and self.player_button_rect.collidepoint(mouse_x, mouse_y)
        
        # Draw vertical background for buttons (narrow strip on the right)
        glColor3f(0.1, 0.1, 0.15)  # Panel color
        glBegin(GL_QUADS)
        glVertex2f(self.display[0] - button_bg_width, 0)
        glVertex2f(self.display[0], 0)
        glVertex2f(self.display[0], self.display[1])
        glVertex2f(self.display[0] - button_bg_width, self.display[1])
        glEnd()
        
        # Load button textures if not already loaded
        if not hasattr(self, 'button_textures'):
            self.load_button_textures()
        
        # Draw menu button with texture
        self.draw_textured_button(self.menu_button_rect, 'menu', is_menu_hovered)
        
        # Draw second panel button with texture
        self.draw_textured_button(self.panel2_button_rect, 'panel2', is_panel2_hovered)
        
        # Draw player model button with texture
        if hasattr(self, 'player_button_rect'):
            self.draw_textured_button(self.player_button_rect, 'player', self.is_player_button_hovered)
        
        # Draw save button with texture
        if hasattr(self, 'save_panel_button_rect'):
            self.draw_textured_button(self.save_panel_button_rect, 'save', 
                                    hasattr(self, 'is_save_button_hovered') and self.is_save_button_hovered)
        
        # Draw AI button with texture
        if hasattr(self, 'ai_panel_button_rect'):
            self.draw_textured_button(self.ai_panel_button_rect, 'ai',
                                    hasattr(self, 'is_ai_button_hovered') and self.is_ai_button_hovered)


        

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



    def save_game_state(self, slot):
        """Save the current game state to a file.
        
        Args:
            slot (int): The save slot number (1-5)
        """
        save_data = {
            'version': 1,
            'timestamp': time.time(),
            'structures': []
        }
        
        # Save all structures
        for struct_id, struct in self.physics.structures.items():
            if struct.get('fill') == 'player':
                continue  # Skip player structure
                
            pos = self.physics.get_structure_position(struct_id)
            orn = self.physics.get_structure_orientation(struct_id)
            
            if pos is None or orn is None:
                continue
                
            # Create a clean copy of the structure data without non-serializable fields
            struct_copy = {
                'position': [float(x) for x in pos],
                'orientation': [float(x) for x in orn],
                'size': [float(x) for x in struct['size']],
                'mass': float(struct.get('mass', 1.0)),
                'color': [float(x) for x in struct.get('color', [0.8, 0.2, 0.2, 1.0])],
                'fill': str(struct.get('fill', 'solid')),
                'metadata': {}
            }
            
            # Copy only serializable metadata
            if 'metadata' in struct:
                for key, value in struct['metadata'].items():
                    if isinstance(value, (str, int, float, bool, list, dict)):
                        if isinstance(value, list):
                            # Ensure list items are serializable
                            struct_copy['metadata'][key] = [
                                float(x) if isinstance(x, (int, float)) else x 
                                for x in value if isinstance(x, (str, int, float, bool))
                            ]
                        elif isinstance(value, dict):
                            # Ensure dict values are serializable
                            struct_copy['metadata'][key] = {
                                k: float(v) if isinstance(v, (int, float)) else v 
                                for k, v in value.items() 
                                if isinstance(k, (str, int, float, bool)) and 
                                   isinstance(v, (str, int, float, bool))
                            }
                        else:
                            struct_copy['metadata'][key] = value
            
            save_data['structures'].append(struct_copy)
        
        # Save to file
        save_path = os.path.join(self.saves_dir, f'save_{slot}.json')
        try:
            with open(save_path, 'w') as f:
                import json
                json.dump(save_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving game: {e}")
            return False
    
    def load_game_state(self, slot):
        """Load a game state from a file.
        
        Args:
            slot (int): The save slot number (1-5)
        """
        save_path = os.path.join(self.saves_dir, f'save_{slot}.json')
        if not os.path.exists(save_path):
            print(f"No save file found in slot {slot}")
            return False
            
        try:
            with open(save_path, 'r') as f:
                import json
                save_data = json.load(f)
                
            # Clear existing structures (except player)
            for struct_id in list(self.physics.structures.keys()):
                if self.physics.structures[struct_id].get('fill') != 'player':
                    self.physics.remove_structure(struct_id)
            
            # Load structures
            for struct_data in save_data.get('structures', []):
                # Ensure all required fields are present and have the correct types
                position = [float(x) for x in struct_data.get('position', [0, 0, 0])]
                size = [float(x) for x in struct_data.get('size', [1, 1, 1])]
                mass = float(struct_data.get('mass', 1.0))
                color = [float(x) for x in struct_data.get('color', [0.8, 0.2, 0.2, 1.0])]
                fill = str(struct_data.get('fill', 'solid'))
                metadata = {k: v for k, v in struct_data.get('metadata', {}).items() 
                          if isinstance(k, (str, int, float, bool)) and 
                             isinstance(v, (str, int, float, bool, list, dict))}
                
                # Add the structure with orientation if available
                orientation = struct_data.get('orientation')
                if orientation and len(orientation) == 4:  # It's a quaternion
                    # Convert quaternion to euler for the initial add_structure call
                    euler = p.getEulerFromQuaternion(orientation)
                    struct_id = self.physics.add_structure(
                        position=position,
                        size=size,
                        mass=mass,
                        color=color,
                        rotation=euler,  # Pass euler angles for initial placement
                        fill=fill,
                        metadata=metadata
                    )
                    
                    # Then set the exact quaternion orientation
                    if struct_id is not None:
                        p.resetBasePositionAndOrientation(
                            struct_id,
                            position,
                            orientation
                        )
                else:
                    # No orientation or invalid orientation, just add normally
                    struct_id = self.physics.add_structure(
                        position=position,
                        size=size,
                        mass=mass,
                        color=color,
                        rotation=None,
                        fill=fill,
                        metadata=metadata
                    )
                
                # If this is a textured structure, ensure we have a texture ID
                if struct_id is not None and fill in ['small_bricks', 'rust_metal', 'concrete', 'stone', 'glass', 'tough_glass', 'wood']:
                    if 'texture_id' not in self.physics.structures[struct_id] or self.physics.structures[struct_id]['texture_id'] is None:
                        # Load textures if needed
                        texture_type = 'glass' if fill == 'tough_glass' else fill
                        if not hasattr(self, f'{texture_type}_textures') or not getattr(self, f'{texture_type}_textures'):
                            self._load_textures(f'textures/{texture_type}', f'{texture_type}_textures')
                        
                        textures = getattr(self, f'{texture_type}_textures', [])
                        if textures:
                            self.physics.structures[struct_id]['texture_id'] = random.choice(textures)
            
            return True
            
        except Exception as e:
            print(f"Error loading game: {e}")
            return False
    
    def draw_save_panel(self):
        """Draw the save/load panel for saving and loading game states."""
        if not hasattr(self, 'show_save_panel') or not self.show_save_panel:
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
        panel_width = 300
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
        self.draw_text("Save / Load", panel_x + 15, 20, 20, (0.9, 0.9, 0.9))
        
        # Save slots
        save_slots = 5
        slot_height = 60
        slot_margin = 10
        start_y = header_height + 20
        
        # Store button rects for click detection
        if not hasattr(self, 'save_buttons'):
            self.save_buttons = {}
            self.load_buttons = {}
        
        # Draw save slots
        for i in range(save_slots):
            slot_num = i + 1
            y = start_y + (slot_height + slot_margin) * i
            
            # Check if save file exists
            save_path = os.path.join(self.saves_dir, f'save_{slot_num}.json')
            has_save = os.path.exists(save_path)
            
            # Slot background
            slot_color = (0.25, 0.25, 0.3) if has_save else (0.2, 0.2, 0.25)
            glColor3f(*slot_color)
            glBegin(GL_QUADS)
            glVertex2f(panel_x + 15, y)
            glVertex2f(panel_x + panel_width - 15, y)
            glVertex2f(panel_x + panel_width - 15, y + slot_height)
            glVertex2f(panel_x + 15, y + slot_height)
            glEnd()
            
            # Slot label and timestamp
            if has_save:
                timestamp = os.path.getmtime(save_path)
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                self.draw_text(f"Slot {slot_num}: {time_str}", panel_x + 20, y + 15, 14, (0.9, 0.9, 0.9))
            else:
                self.draw_text(f"Slot {slot_num}: Empty", panel_x + 20, y + 15, 14, (0.7, 0.7, 0.7))
            
            # Save/load buttons
            button_width = 80
            button_height = 20
            
            # Save button
            save_btn_x = panel_x + panel_width - 110
            save_btn_y = y + 10
            
            # Store button rect for click detection
            self.save_buttons[slot_num] = pygame.Rect(save_btn_x, save_btn_y, button_width, button_height)
            
            # Button background (highlight on hover)
            mouse_x, mouse_y = pygame.mouse.get_pos()
            is_hovered = self.save_buttons[slot_num].collidepoint(mouse_x, mouse_y)
            
            save_btn_color = (0.3, 0.6, 0.3) if is_hovered else (0.2, 0.5, 0.2)
            glColor3f(*save_btn_color)
            glBegin(GL_QUADS)
            glVertex2f(save_btn_x, save_btn_y)
            glVertex2f(save_btn_x + button_width, save_btn_y)
            glVertex2f(save_btn_x + button_width, save_btn_y + button_height)
            glVertex2f(save_btn_x, save_btn_y + button_height)
            glEnd()
            
            # Save button text
            self.draw_text("Save", save_btn_x + 30, save_btn_y + 4, 14, (1, 1, 1))
            
            # Load button
            load_btn_x = save_btn_x
            load_btn_y = y + 35
            
            # Store button rect for click detection
            self.load_buttons[slot_num] = pygame.Rect(load_btn_x, load_btn_y, button_width, button_height)
            
            # Button background (highlight on hover and enable/disable based on save existence)
            is_hovered = self.load_buttons[slot_num].collidepoint(mouse_x, mouse_y)
            
            if has_save:
                load_btn_color = (0.3, 0.5, 0.7) if is_hovered else (0.2, 0.4, 0.6)
                text_color = (1, 1, 1)
            else:
                load_btn_color = (0.15, 0.15, 0.15)
                text_color = (0.5, 0.5, 0.5)
                
            glColor3f(*load_btn_color)
            glBegin(GL_QUADS)
            glVertex2f(load_btn_x, load_btn_y)
            glVertex2f(load_btn_x + button_width, load_btn_y)
            glVertex2f(load_btn_x + button_width, load_btn_y + button_height)
            glVertex2f(load_btn_x, load_btn_y + button_height)
            glEnd()
            
            # Load button text
            self.draw_text("Load", load_btn_x + 30, load_btn_y + 4, 14, text_color)
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()


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
    
    def draw_loading_screen(self):
        """Draw a loading screen with the current loading message and progress."""
        # Switch to orthographic projection for 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth testing for 2D rendering
        glDisable(GL_DEPTH_TEST)
        
        # Draw semi-transparent overlay
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.1, 0.1, 0.1, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(self.display[0], 0)
        glVertex2f(self.display[0], self.display[1])
        glVertex2f(0, self.display[1])
        glEnd()
        
        # Draw loading text
        text = f"{self.loading_message}... {int(self.loading_progress * 100)}%"
        text_surface = self.loading_font.render(text, True, (255, 255, 255))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        text_width = text_surface.get_width()
        text_height = text_surface.get_height()
        
        # Position text in the center
        x = (self.display[0] - text_width) // 2
        y = (self.display[1] - text_height) // 2
        
        # Draw progress bar background
        bar_width = 300
        bar_height = 20
        bar_x = (self.display[0] - bar_width) // 2
        bar_y = y + text_height + 20
        
        glColor4f(0.3, 0.3, 0.3, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, bar_y)
        glVertex2f(bar_x + bar_width, bar_y)
        glVertex2f(bar_x + bar_width, bar_y + bar_height)
        glVertex2f(bar_x, bar_y + bar_height)
        glEnd()
        
        # Draw progress bar fill
        fill_width = int(bar_width * self.loading_progress)
        glColor4f(0.2, 0.6, 1.0, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, bar_y)
        glVertex2f(bar_x + fill_width, bar_y)
        glVertex2f(bar_x + fill_width, bar_y + bar_height)
        glVertex2f(bar_x, bar_y + bar_height)
        glEnd()
        
        # Draw text
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)  # Unbind any existing texture
        glRasterPos2i(x, y + text_height)
        glDrawPixels(text_width, text_height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        glDisable(GL_TEXTURE_2D)
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        # Update the display
        pygame.display.flip()
    
    def update_loading(self, progress, message):
        """Update loading progress and message."""
        self.loading_progress = max(0.0, min(1.0, progress))
        self.loading_message = message
        self.draw_loading_screen()
        pygame.event.pump()  # Keep the window responsive
    
    def switch_player_model(self, model_name):
        """Switch to the specified player model with loading screen."""
        try:
            # Set loading state
            self.is_loading = True
            self.loading_message = "Loading model"
            self.loading_progress = 0.0
            self.update_loading(0.1, "Initializing...")
            
            # Load the new model
            self.update_loading(0.3, f"Loading {model_name}")
            new_model = self.model_lib.load_model(model_name)
            
            if new_model:
                self.update_loading(0.6, "Applying model")
                self.currPlayerModel = new_model
                
                # Update camera height based on new model
                if 'camera_height' in self.currPlayerModel.metadata:
                    self.camera_height = self.currPlayerModel.metadata['camera_height']
                
                # Update player collision in physics world
                if hasattr(self, 'player_structure_id'):
                    self.update_loading(0.7, "Updating physics")
                    # Remove old player collision
                    self.physics.remove_structure(self.player_structure_id)
                    
                    # Add new player collision
                    player_mesh_path = self.currPlayerModel.metadata.get('mesh_path')
                    player_mesh_obj = None
                    if player_mesh_path:
                        self.update_loading(0.8, "Loading collision mesh")
                        from models.obj_loader import OBJ
                        player_mesh_obj = OBJ(player_mesh_path)
                    
                    self.update_loading(0.9, "Finalizing")
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
                
                # Update audio
                self.update_loading(0.95, "Loading audio")
                self.load_audio_from_model()
                
                # Update player model in the player class
                if hasattr(self, 'player'):
                    self.player.model = self.currPlayerModel
                
                # Show completion briefly
                self.update_loading(1.0, "Done!")
                pygame.time.delay(200)  # Show 100% for a moment
                
                return True
                
        except Exception as e:
            print(f"Error switching to model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # Always ensure loading is turned off
            self.is_loading = False
    
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

    def resolve_audio_path(self, filename):
        """Resolve audio file path with fallback to default sounds directory"""
        # Check if file exists directly
        if os.path.exists(filename):
            return filename
            
        base_dir = os.path.dirname(__file__)
        
        # List of directories to check for sound files
        search_dirs = [
            os.path.join(base_dir, '..', 'sounds'),
            os.path.join(base_dir, '..', 'textures'),
            os.path.join(base_dir, '..', 'audio'),  # Add audio directory
            os.path.join(base_dir, '..', 'src', 'sounds'),
            os.path.join(base_dir, '..', 'src', 'audio'),  # Add src/audio directory
        ]
        
        # Check in all search directories
        for search_dir in search_dirs:
            # Check exact filename
            path = os.path.join(search_dir, filename)
            if os.path.exists(path):
                return path
                
        # Try with different extensions if needed
        base, ext = os.path.splitext(filename)
        if ext:  # If filename has an extension
            for search_dir in search_dirs:
                # Try .wav if not already
                if ext.lower() != '.wav':
                    wav_path = os.path.join(search_dir, base + '.wav')
                    if os.path.exists(wav_path):
                        return wav_path
                        
                # Try .mp3 if not already
                if ext.lower() != '.mp3':
                    mp3_path = os.path.join(search_dir, base + '.mp3')
                    if os.path.exists(mp3_path):
                        return mp3_path
        
        print(f"Warning: Sound file not found: {filename}")
        return None
        
    def play_sound(self, filename, volume=1.0, channel_num=0):
        """Play a sound effect
        
        Args:
            filename: Name of the sound file to play
            volume: Volume level (0.0 to 1.0)
            channel_num: Audio channel to use (0-7)
        """
        if not self.audio_enabled:
            return None
            
        try:
            # Resolve the sound file path
            sound_file = self.resolve_audio_path(filename)
            if not sound_file:
                print(f"Sound file not found: {filename}")
                return None
                
            # Load the sound
            sound = pygame.mixer.Sound(sound_file)
            sound.set_volume(volume)
            
            # Get or create channel
            if channel_num < 0 or channel_num >= pygame.mixer.get_num_channels():
                channel_num = 0
                
            channel = pygame.mixer.Channel(channel_num)
            channel.play(sound)
            return channel
            
        except Exception as e:
            print(f"Error playing sound {filename}: {e}")
            return None
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

    def calculate_normal(self, v1, v2, v3):
        """Calculate normal vector for a triangle defined by three points."""
        # Convert to numpy arrays for vector operations
        v1 = np.array(v1, dtype=np.float32)
        v2 = np.array(v2, dtype=np.float32)
        v3 = np.array(v3, dtype=np.float32)
        
        # Calculate two vectors in the plane
        u = v2 - v1
        v = v3 - v1
        
        # Calculate cross product
        normal = np.cross(u, v)
        
        # Normalize the vector
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        
        return normal

    def draw_textured_cube(self, size, texture_id):
        """Draw a cuboid with texture applied to all faces and dynamic lighting."""
        # Get current color to check for transparency
        current_color = glGetFloatv(GL_CURRENT_COLOR)
        is_transparent = len(current_color) > 3 and current_color[3] < 0.99
        
        # If the object is transparent, enable blending and disable depth writing
        if is_transparent:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)  # Disable depth writing for transparent objects
        
        # Half sizes for each dimension
        hx = size[0] / 2.0
        hy = size[1] / 2.0
        hz = size[2] / 2.0
        
        # Define the 8 vertices of the cube
        vertices = [
            [-hx, -hy,  hz],  # 0 front bottom left
            [ hx, -hy,  hz],  # 1 front bottom right
            [ hx,  hy,  hz],  # 2 front top right
            [-hx,  hy,  hz],  # 3 front top left
            [-hx, -hy, -hz],  # 4 back bottom left
            [ hx, -hy, -hz],  # 5 back bottom right
            [ hx,  hy, -hz],  # 6 back top right
            [-hx,  hy, -hz]   # 7 back top left
        ]
        
        # Define the 6 faces (4 vertices each, in counter-clockwise order)
        faces = [
            [0, 1, 2, 3],  # Front
            [5, 4, 7, 6],  # Back
            [3, 2, 6, 7],  # Top
            [4, 5, 1, 0],  # Bottom
            [4, 0, 3, 7],  # Left
            [1, 5, 6, 2]   # Right
        ]
        
        # Texture coordinates for each face
        tex_coords = [
            [0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  0.0, 1.0],  # Front
            [0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  0.0, 1.0],  # Back
            [0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  0.0, 1.0],  # Top
            [0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  0.0, 1.0],  # Bottom
            [0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  0.0, 1.0],  # Left
            [0.0, 0.0,  1.0, 0.0,  1.0, 1.0,  0.0, 1.0]   # Right
        ]
        
        # Enable texturing and setup material properties
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        # Enable texture repeating
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        
        # Enable lighting for this object
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        
        # For transparent objects, we need to handle materials differently
        if is_transparent:
            # Disable color material for transparent objects to maintain transparency
            glDisable(GL_COLOR_MATERIAL)
            
            # Set material properties for glass
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (0.1, 0.2, 0.4, 0.4))
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (0.2, 0.3, 0.7, 0.4))
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (0.8, 0.8, 0.9, 0.4))
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 90.0)
            
            # Enable alpha testing to discard fully transparent fragments
            glEnable(GL_ALPHA_TEST)
            glAlphaFunc(GL_GREATER, 0.1)
        else:
            # For opaque objects, use the standard color material
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Use smooth shading for better lighting interpolation
        glShadeModel(GL_SMOOTH)
        
        # Draw each face with proper normals
        for face_idx, face in enumerate(faces):
            # Get the vertices for this face
            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]
            v4 = vertices[face[3]]
            
            # Calculate normal for this face
            normal = self.calculate_normal(v1, v2, v3)
            
            # Begin drawing the quad
            glBegin(GL_QUADS)
            
            # Set the normal for this face
            glNormal3f(normal[0], normal[1], normal[2])
            
            # Draw each vertex with texture coordinates
            for i, vertex_idx in enumerate(face):
                tex_x = tex_coords[face_idx][i*2]
                tex_y = tex_coords[face_idx][i*2 + 1]
                glTexCoord2f(tex_x, tex_y)
                glVertex3f(*vertices[vertex_idx])
            
            glEnd()
        
        # Clean up state
        if is_transparent:
            glDisable(GL_ALPHA_TEST)
            glEnable(GL_COLOR_MATERIAL)  # Re-enable color material for other objects
            glDepthMask(GL_TRUE)  # Re-enable depth writing
            glDisable(GL_BLEND)   # Disable blending
        
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
        
        # Get fill type, base color, and model path
        fill_type = config.get('fill', 'solid')
        base_color = config.get('base_color', (0.8, 0.8, 0.8, 1.0))
        model_path = config.get('path')
        
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
        
        # Load textures if needed for textured materials
        if fill_type in ['small_bricks', 'rust_metal', 'concrete', 'stone', 'glass', 'tough_glass', 'wood']:
            texture_type = 'glass' if fill_type == 'tough_glass' else fill_type
            texture_attr = f'{texture_type}_textures'
            
            # Load textures if not already loaded
            if not hasattr(self, texture_attr) or not getattr(self, texture_attr):
                self._load_textures(f'textures/{texture_type}', texture_attr)
            
            # Get the textures list
            textures = getattr(self, texture_attr, [])
            # Choose a random texture for this cluster
            cluster_texture_id = random.choice(textures) if textures else None
        else:
            cluster_texture_id = None
        
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
                    elif fill_type == 'stone' and hasattr(self, 'stone_textures') and self.stone_textures:
                        texture_id = random.choice(self.stone_textures)
                    elif fill_type == 'glass' and hasattr(self, 'glass_textures') and self.glass_textures:
                        texture_id = random.choice(self.glass_textures)
                    # Prepare metadata for the structure
                    metadata = {}
                    # Get metadata from config if it exists
                    if 'metadata' in config:
                        metadata.update(config['metadata'])
                    
                    # Add other metadata fields
                    if 'hit_sound' in config:
                        metadata['hit_sound'] = config['hit_sound']
                    if 'stiff' in config:
                        metadata['stiff'] = config['stiff']
                    if 'shard_config' in config:
                        metadata['shard_config'] = config['shard_config']
                    
                    # Add model_path to metadata if available
                    if model_path and 'model_path' not in metadata:
                        metadata['model_path'] = model_path

                    
                    # Add to physics world with fill type, texture ID, and metadata
                    mesh_obj = None
                    if 'mesh_path' in config:
                        try:
                            from models.obj_loader import OBJ
                            mesh_obj = OBJ(config['mesh_path'])
                        except Exception as e:
                            mesh_obj = None
                    
                    # Ensure model_path is in metadata if available in config
                    if 'path' in config and 'model_path' not in metadata:
                        metadata['model_path'] = config['path']
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
                        metadata=metadata.copy(),  # Make a copy to avoid modifying the original
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
                    
                    # Use the stored texture ID for rendering if available
                    if texture_id is not None:
                        self.draw_textured_cube(cube_size, texture_id)
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
        if self.show_ai_panel:
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
    
    def spawn_ai_structure(self, structure_data, position=None, rotation=0):
        """
        Spawn an AI-generated structure into the simulation.
        
        Args:
            structure_data: The structure data to spawn (can be a dictionary or JSON string)
            position: Optional position to spawn the structure at. If None, will use cursor position.
            rotation: Rotation in radians around the Y axis
        """
        try:
            
            # Parse structure data if it's a string
            if isinstance(structure_data, str):
                structure = json.loads(structure_data)
            else:
                structure = structure_data
            
            # Get the spawn position (use cursor position if not specified)
            if position is None:
                cursor_pos = self.get_cursor_world_hit()
                if cursor_pos is None:
                    # If no cursor hit, use a position in front of the camera
                    cursor_pos = [
                        self.player.position[0] + math.sin(math.radians(self.player.rotation[1])) * 5,
                        0,
                        self.player.position[2] + math.cos(math.radians(self.player.rotation[1])) * 5
                    ]
                position = cursor_pos
            
            # Convert rotation to quaternion (around Y axis)
            cy = math.cos(rotation * 0.5)
            sy = math.sin(rotation * 0.5)
            orientation = [0, cy, 0, sy]  # Quaternion for Y rotation
            
            # Handle different structure data formats
            if 'structures' in structure:
                # If the structure data has a 'structures' key, use that
                bodies = structure['structures']
            elif 'bodies' in structure:
                # If the structure data has a 'bodies' key, use that
                bodies = structure['bodies']
            elif isinstance(structure, list):
                # If the structure data is a list, treat it as a list of bodies
                bodies = structure
            else:
                # Otherwise, treat the entire structure as a single body
                bodies = [structure]
            
            # Add the structure to the physics world
            for i, body in enumerate(bodies):
                
                # Make a copy of the body to avoid modifying the original
                body = json.loads(json.dumps(body))
                
                # Set default values if they don't exist
                if 'position' not in body or not isinstance(body['position'], (list, tuple)) or len(body['position']) < 3:
                    body['position'] = [0, 0, 0]
                
                if 'size' not in body or not isinstance(body['size'], (list, tuple)) or len(body['size']) < 3:
                    body['size'] = [1, 1, 1]
                
                if 'mass' not in body or not isinstance(body['mass'], (int, float)):
                    body['mass'] = 1.0
                
                if 'color' not in body or not isinstance(body['color'], (list, tuple)) or len(body['color']) < 4:
                    body['color'] = [0.8, 0.2, 0.2, 1.0]  # Default red color
                
                if 'fill' not in body or not isinstance(body['fill'], str):
                    body['fill'] = 'solid'  # Default fill type
                
                # Get relative position
                rel_pos = body['position'].copy()
                
                # Apply rotation to the position
                if rotation != 0:
                    # Rotate around the center (0,0,0)
                    x = rel_pos[0]
                    z = rel_pos[2]
                    rel_pos[0] = x * math.cos(rotation) - z * math.sin(rotation)
                    rel_pos[2] = x * math.sin(rotation) + z * math.cos(rotation)
                
                # Apply the position offset
                rel_pos[0] += position[0]
                rel_pos[1] += position[1]
                rel_pos[2] += position[2]
                
                body['position'] = rel_pos
                
                # Apply rotation to the orientation if it exists
                if 'orientation' in body and isinstance(body['orientation'], (list, tuple)) and len(body['orientation']) == 4:
                    # Combine rotations (quaternion multiplication)
                    q1 = orientation
                    q2 = body['orientation']
                    w1, x1, y1, z1 = q1
                    w2, x2, y2, z2 = q2
                    
                    body['orientation'] = [
                        w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2
                    ]
                else:
                    body['orientation'] = orientation.copy()
                
                
                # Extract parameters from the body dictionary
                try:
                    # Get the base position (cursor or camera position)
                    base_position = position if position is not None else [0, 0, 0]
                    
                    # Get relative position from the body
                    rel_position = body.get('position', [0, 0, 0])
                    
                    # Calculate final position by adding relative position to base position
                    final_position = [
                        base_position[0] + rel_position[0],
                        base_position[1] + rel_position[1],
                        base_position[2] + rel_position[2]
                    ]
                       
                    
                    size = body.get('size', [1, 1, 1])
                    mass = body.get('mass', 1.0)
                    color = body.get('color', [0.8, 0.2, 0.2, 1.0])
                    
                    # Handle orientation (convert quaternion to euler if needed)
                    orientation = body.get('orientation', [0, 0, 0, 1])
                    if len(orientation) == 4:  # It's a quaternion
                        # Convert quaternion to euler angles
                        euler = p.getEulerFromQuaternion(orientation)
                    else:  # Assume it's euler angles
                        euler = orientation
                    
                    # Get metadata and fill type
                    metadata = body.get('metadata', {})
                    # Get fill from the top level of the body, default to 'solid' if not found
                    fill = body.get('fill', 'solid')
                    
                    # Ensure textures are loaded for this fill type if it's a texture type
                    texture_types = ['small_bricks', 'rust_metal', 'concrete', 'stone', 'glass', 'tough_glass', 'wood']
                    if fill in texture_types:
                        texture_attr = f'{fill}_textures'
                        if not hasattr(self, texture_attr) or not getattr(self, texture_attr):
                            texture_dir = os.path.join(os.path.dirname(__file__), 'textures', fill)
                            self._load_textures(texture_dir, texture_attr)
                    
                    # Add the body to the physics world
                    try:
                        struct_id = self.physics.add_structure(
                            position=final_position,
                            size=size,
                            mass=mass,
                            color=color,
                            rotation=euler,
                            fill=fill,
                            metadata=metadata
                        )
                        
                        if struct_id is not None: 
                            
                            # If this is a stiff structure, store its ID
                            if metadata.get('stiff', False):
                                if not hasattr(self.physics, 'stiff_structures'):
                                    self.physics.stiff_structures = set()
                                self.physics.stiff_structures.add(struct_id)
                                
                        else:
                            print("[WARNING] Failed to add body to physics world: add_structure returned None")
                            
                    except Exception as e:
                        print(f"[ERROR] Failed to add body to physics world: {e}")
                        import traceback
                        traceback.print_exc()
                        
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    continue  # Skip this body if there's an error
            
            # Play a sound to indicate successful placement
            self.play_sound("place.wav")
            return True
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False

    def handle_ai_panel_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Calculate panel position
            panel_x = (self.display[0] - 600) // 2
            panel_y = (self.display[1] - 700) // 2  # Using the new panel height
            
            # Define slot input field rect
            slot_input_rect = pygame.Rect(
                panel_x + 70,  # panel_x + 20 (margin) + 50 (label width)
                panel_y + 60,  # panel_y + 60 (top margin)
                50, 30
            )
            
            # Check if click is inside the slot input field
            if slot_input_rect.collidepoint(event.pos):
                self.slot_input_active = True
                self.ai_input_active = False
                # Initialize temp_slot_number if it doesn't exist
                if not hasattr(self, 'temp_slot_number') and hasattr(self, 'ai_slot_number'):
                    self.temp_slot_number = str(self.ai_slot_number)
                elif not hasattr(self, 'temp_slot_number'):
                    self.temp_slot_number = '1'
                return True
                
            # Check if click is inside the description input area
            elif hasattr(self, 'ai_input_rect') and self.ai_input_rect.collidepoint(event.pos):
                self.ai_input_active = True
                self.slot_input_active = False
                # Set cursor position based on click position
                font = pygame.font.Font(None, 18)
                x = event.pos[0] - self.ai_input_rect.x - 5  # 5px padding
                text_so_far = ""
                self.ai_input_cursor_pos = len(self.ai_input_text)  # Default to end of text
                
                for i, char in enumerate(self.ai_input_text):
                    text_so_far += char
                    if font.size(text_so_far)[0] > x:
                        self.ai_input_cursor_pos = max(0, i)
                        break
                return True
                
            # Check for slot button clicks (for spawning only)
            if hasattr(self, 'ai_slot_rects'):
                for i, rect in enumerate(self.ai_slot_rects):
                    if rect.collidepoint(event.pos):
                        slot_num = i + 1
                        filename = f'ai_save_{slot_num:02d}.json'
                        
                        # Check if this slot has content
                        if self.ai_handler and hasattr(self.ai_handler, 'save_exists') and \
                           self.ai_handler.save_exists(filename):
                            try:
                                try:
                                    print(f"[AI PANEL] Loading structure from slot {slot_num}, file: {filename}")
                                    # Load the structure from this slot
                                    structure_data = self.ai_handler.load_ai_structure(filename)
                                    print(f"[AI PANEL] Loaded structure data from {filename}")
                                    print(f"[AI PANEL] Structure data type: {type(structure_data)}")
                                    if isinstance(structure_data, dict):
                                        print(f"[AI PANEL] Structure data keys: {structure_data.keys()}")
                                    
                                    if structure_data:
                                        # Get the cursor's world position for spawning
                                        target_point = self.get_cursor_world_hit()
                                        if target_point is None:
                                            # If no cursor hit, use a position in front of the camera
                                            target_point = [
                                                self.player.position[0] + math.sin(math.radians(self.player.rotation[1])) * 5,
                                                0,
                                                self.player.position[2] + math.cos(math.radians(self.player.rotation[1])) * 5
                                            ]
                                            print(f"[AI PANEL] Using camera-relative position: {target_point}")
                                        else:
                                            print(f"[AI PANEL] Using cursor position: {target_point}")
                                        
                                        # Debug print structure data
                                        if isinstance(structure_data, dict):
                                            if 'bodies' in structure_data:
                                                print(f"[AI PANEL] Found {len(structure_data['bodies'])} bodies in structure data")
                                                if structure_data['bodies'] and isinstance(structure_data['bodies'][0], dict):
                                                    print(f"[AI PANEL] First body keys: {structure_data['bodies'][0].keys()}")
                                                    if 'metadata' in structure_data['bodies'][0]:
                                                        print(f"[AI PANEL] First body metadata: {structure_data['bodies'][0]['metadata']}")
                                            elif 'structures' in structure_data:
                                                print(f"[AI PANEL] Found {len(structure_data['structures'])} structures in structure data")
                                        
                                        # Check if the structure data has 'bodies' or 'structures' key
                                        if 'bodies' in structure_data or 'structures' in structure_data:
                                            print("[AI PANEL] Spawning structure with 'bodies' or 'structures' key")
                                            # Spawn the structure at the target position
                                            if self.spawn_ai_structure(structure_data, position=target_point):
                                                self.ai_status_text = f"Loaded structure from slot {slot_num}"
                                                self.ai_status_timer = 180
                                                self.play_sound("place.wav")
                                                return True
                                            else:
                                                self.ai_status_text = "Failed to spawn structure"
                                                self.ai_status_timer = 180
                                                return True
                                        else:
                                            # Handle the case where the structure data is in a different format
                                            print("[AI PANEL] WARNING: No 'bodies' or 'structures' key found. Wrapping in 'bodies' array.")
                                            wrapped_data = {'bodies': [structure_data]}
                                            print("[AI PANEL] Wrapped data structure:", wrapped_data)
                                            if self.spawn_ai_structure(wrapped_data, position=target_point):
                                                self.ai_status_text = f"Loaded structure from slot {slot_num}"
                                                self.ai_status_timer = 180
                                                self.play_sound("place.wav")
                                                return True
                                    else:
                                        self.ai_status_text = "Aim at a valid position to place the structure"
                                        self.ai_status_timer = 180
                                        return True
                                        
                                except Exception as e:
                                    print(f"[AI PANEL ERROR] Error loading/spawning structure: {str(e)}")
                                    import traceback
                                    print("Traceback:")
                                    traceback.print_exc()
                                    self.ai_status_text = f"Error loading structure: {str(e)}"
                                    self.ai_status_timer = 180
                                    return True
                                else:
                                    self.ai_status_text = f"Failed to load structure from slot {slot_num}"
                                    self.ai_status_timer = 180
                                    return True
                            except Exception as e:
                                self.ai_status_text = f"Error loading structure: {str(e)}"
                                self.ai_status_timer = 180
                                print(f"Error loading AI structure: {e}")
                                return True
                        else:
                            self.ai_status_text = f"Slot {slot_num} is empty"
                            self.ai_status_timer = 180
                            return True
            
            # Check if click is on the Set Slot button
            slot_btn_rect = pygame.Rect(
                (self.display[0] - 600) // 2 + 130,  # panel_x + 20 + 50 + 60
                (self.display[1] - 600) // 2 + 60,   # panel_y + 60
                80, 30
            )
            if slot_btn_rect.collidepoint(event.pos):
                # Save the current slot number
                if hasattr(self, 'temp_slot_number'):
                    try:
                        slot = int(self.temp_slot_number)
                        if 1 <= slot <= 20:
                            self.ai_slot_number = slot
                            self.ai_status_text = f"Slot {slot} selected"
                            self.ai_status_timer = 180
                            # Clear the temporary slot number
                            if hasattr(self, 'temp_slot_number'):
                                delattr(self, 'temp_slot_number')
                        else:
                            self.ai_status_text = "Slot must be between 1-20"
                            self.ai_status_timer = 180
                    except (ValueError, AttributeError):
                        self.ai_status_text = "Invalid slot number"
                        self.ai_status_timer = 180
                return True
            
            # Check delete button clicks
            if hasattr(self, 'ai_delete_rects'):
                for i, rect in enumerate(self.ai_delete_rects):
                    if rect.collidepoint(event.pos):
                        slot_num = i + 1
                        if self.delete_ai_slot(slot_num):
                            return True
            
            # Check generate button click
            if hasattr(self, 'ai_generate_rect') and self.ai_generate_rect.collidepoint(event.pos):
                # Check if a slot is selected
                if not hasattr(self, 'ai_slot_number'):
                    self.ai_status_text = "Please select a slot first"
                    self.ai_status_timer = 180
                    return True
                    
                # Check if slot is already used
                if hasattr(self, 'ai_slot_number') and self.is_slot_used(self.ai_slot_number):
                    self.ai_status_text = f"Slot {self.ai_slot_number} is not empty. Please choose an empty slot."
                    self.ai_status_timer = 180
                    return True
                    
                # Check if AI is properly initialized
                if not hasattr(self, 'ai_initialized') or not self.ai_initialized:
                    self.ai_status_text = "Error: AI not ready yet"
                    if hasattr(self.ai_handler, 'last_error'):
                        self.ai_status_text = f"AI Error: {self.ai_handler.last_error}"
                    self.ai_status_timer = 180
                    return True
                    
                # Use the synchronous method which will handle the async part internally
                self.generate_ai_structure()
                return True  # Event handled
                
            if hasattr(self, 'ai_close_rect') and self.ai_close_rect.collidepoint(event.pos):
                self.show_ai_panel = False
                return True  # Event handled
        
        # Handle keyboard input for slot number
        elif event.type == pygame.KEYDOWN and hasattr(self, 'slot_input_active') and self.slot_input_active:
            if event.key == pygame.K_RETURN:
                # Save the slot number when Enter is pressed
                if hasattr(self, 'temp_slot_number'):
                    try:
                        slot = int(self.temp_slot_number)
                        if 1 <= slot <= 20:
                            self.ai_slot_number = slot
                            self.ai_status_text = f"Slot {slot} selected"
                            self.ai_status_timer = 180
                            delattr(self, 'temp_slot_number')
                        else:
                            self.ai_status_text = "Slot must be between 1-20"
                            self.ai_status_timer = 180
                    except (ValueError, AttributeError):
                        self.ai_status_text = "Invalid slot number"
                        self.ai_status_timer = 180
                self.slot_input_active = False
                return True
            elif event.key == pygame.K_ESCAPE:
                self.slot_input_active = False
                return True
            elif event.key == pygame.K_BACKSPACE:
                if hasattr(self, 'temp_slot_number') and len(self.temp_slot_number) > 0:
                    self.temp_slot_number = self.temp_slot_number[:-1]
                    return True
            elif event.unicode.isdigit() and len(event.unicode) == 1:
                # Only allow single digit input for slots 1-20
                if not hasattr(self, 'temp_slot_number'):
                    self.temp_slot_number = event.unicode
                else:
                    new_num = self.temp_slot_number + event.unicode
                    if 1 <= int(new_num) <= 20:
                        self.temp_slot_number = new_num
                return True
                
        # Handle keyboard input when description input is active
        elif event.type == pygame.KEYDOWN and self.ai_input_active:
            if event.key == pygame.K_RETURN:
                # Check if AI is properly initialized
                if not hasattr(self, 'ai_initialized') or not self.ai_initialized:
                    self.ai_status_text = "Error: AI not ready yet"
                    if hasattr(self.ai_handler, 'last_error'):
                        self.ai_status_text = f"AI Error: {self.ai_handler.last_error}"
                    self.ai_status_timer = 180
                    return True
                # Use the synchronous method which will handle the async part internally
                self.generate_ai_structure()
                return True
            elif event.key == pygame.K_ESCAPE:
                self.show_ai_panel = False
                return True
            elif event.key == pygame.K_BACKSPACE:
                if self.ai_input_cursor_pos > 0:
                    self.ai_input_text = (self.ai_input_text[:self.ai_input_cursor_pos-1] + 
                                       self.ai_input_text[self.ai_input_cursor_pos:])
                    self.ai_input_cursor_pos = max(0, self.ai_input_cursor_pos - 1)
            elif event.key == pygame.K_DELETE:
                if self.ai_input_cursor_pos < len(self.ai_input_text):
                    self.ai_input_text = (self.ai_input_text[:self.ai_input_cursor_pos] + 
                                       self.ai_input_text[self.ai_input_cursor_pos+1:])
            elif event.key == pygame.K_LEFT:
                self.ai_input_cursor_pos = max(0, self.ai_input_cursor_pos - 1)
            elif event.key == pygame.K_RIGHT:
                self.ai_input_cursor_pos = min(len(self.ai_input_text), self.ai_input_cursor_pos + 1)
            elif event.key == pygame.K_HOME:
                self.ai_input_cursor_pos = 0
            elif event.key == pygame.K_END:
                self.ai_input_cursor_pos = len(self.ai_input_text)
            elif event.unicode.isprintable() and not event.key in [pygame.K_RETURN, pygame.K_TAB]:
                # Get the current line start position
                line_start = self.ai_input_text.rfind('\n', 0, self.ai_input_cursor_pos) + 1
                current_line = self.ai_input_text[line_start:self.ai_input_cursor_pos]
                
                # Get the font and calculate the width of the current line
                font = pygame.font.Font(None, 18)
                line_width = font.size(current_line + event.unicode)[0]
                
                # If adding this character would exceed the width, insert a newline
                if line_width > 480:  # 500 - 10*2 (padding)
                    # Insert newline before the current word
                    last_space = current_line.rfind(' ')
                    if last_space > 0:
                        # Insert newline at the last space
                        insert_pos = line_start + last_space
                        self.ai_input_text = (self.ai_input_text[:insert_pos] + 
                                           '\n' + 
                                           self.ai_input_text[insert_pos+1:])
                        # Update cursor position to be after the newline
                        self.ai_input_cursor_pos = insert_pos + 1
                    else:
                        # If no space found, insert newline at the cursor
                        self.ai_input_text = (self.ai_input_text[:self.ai_input_cursor_pos] + 
                                           '\n' + 
                                           self.ai_input_text[self.ai_input_cursor_pos:])
                        self.ai_input_cursor_pos += 1
                
                # Insert the character at the cursor position
                self.ai_input_text = (self.ai_input_text[:self.ai_input_cursor_pos] + 
                                   event.unicode + 
                                   self.ai_input_text[self.ai_input_cursor_pos:])
                self.ai_input_cursor_pos += len(event.unicode)
            return True  # Event handled
            
        return False  # Event not handled
    
    def generate_ai_structure(self):
        """Synchronously start the AI structure generation"""
        if not hasattr(self, 'ai_input_text') or not self.ai_input_text.strip():
            self.ai_status_text = "Please enter a description"
            self.ai_status_timer = 180  # 3 seconds at 60 FPS
            return
            
        if self.is_fetching_genai:
            return  # Don't start a new request if one is already in progress
            
        self.is_fetching_genai = True
        self.ai_status_text = "Generating structure..."
        self.ai_status_timer = 0  # Keep showing until done
        
        # Create a task to run the async generation
        self.ai_generation_future = self._run_async(self._generate_ai_structure_async())

    async def _generate_ai_structure_async(self):
        """Async method to handle the AI structure generation"""
        try:
            structure = await self.ai_handler.generate_from_description(self.ai_input_text)
            
            if not structure:
                self.ai_status_text = "Failed to generate structure. Please try again."
                self.ai_status_timer = 180
                return
                
            # Save the generated structure to the selected slot
            if hasattr(self, 'ai_slot_number') and structure:
                slot_num = self.ai_slot_number
                # Set the generated JSON in the handler
                self.ai_handler.generated_json = json.dumps(structure, indent=2)
                # Save to the selected slot (0-based index)
                success = self.ai_handler.save_ai_structure(slot_num - 1)  # 0-based index
                if success:
                    self.ai_status_text = f"Saved to slot {slot_num}"
                    self.ai_status_timer = 180  # 3 seconds at 60 FPS
                    
                    # Set the structure for immediate placement
                    self.ai_structure_to_place = structure
                    self.ai_show_preview = True
                    self.show_ai_panel = False  # Close the AI panel
                    self.placement_mode = 'ai_structure'
                    self.placement_structure = structure
                    self.placement_rotation = 0
            
            # Show success message
            self.ai_status_text = "Structure generated successfully!"
            self.ai_status_timer = 180
            
            # Clear the input after successful generation
            self.ai_input_text = ""
            self.ai_input_cursor_pos = 0
            
          
            
            # Return the generated structure
            return structure
            
        except json.JSONDecodeError as e:
            error_msg = f"Error: Invalid structure format - {str(e)}"
            print(error_msg)
            self.ai_status_text = error_msg
            self.ai_status_timer = 180
        except Exception as e:
            import traceback
            error_msg = f"Error in generate_ai_structure: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.ai_status_text = f"Error: {str(e)}"
            self.ai_status_timer = 180
        finally:
            # Always reset the loading state when done
            self.is_fetching_genai = False
    
    def is_slot_used(self, slot_num):
        """Check if a slot has saved content"""
        if not hasattr(self, 'ai_handler'):
            return False
        filename = f'ai_save_{slot_num:02d}.json'
        return self.ai_handler.save_exists(filename)

    def delete_ai_slot(self, slot_num):
        """Completely clear the contents of a slot file"""
        if not hasattr(self, 'ai_handler'):
            return False
            
        # Get the directory where AI structures are saved
        ai_structures_dir = os.path.join(os.path.dirname(__file__), 'ai_structures')
        os.makedirs(ai_structures_dir, exist_ok=True)
        
        filename = f'ai_save_{slot_num:02d}.json'
        try:
            filepath = os.path.join(ai_structures_dir, filename)
            if os.path.exists(filepath):
                # Completely truncate the file
                with open(filepath, 'w') as f:
                    f.write('')
                
                self.ai_status_text = f"Cleared slot {slot_num}"
                self.ai_status_timer = 180  # 3 seconds at 60 FPS
                print(f"[AI PANEL] Successfully truncated {filepath}")
                return True
            else:
                print(f"[AI PANEL] File not found: {filepath}")
        except Exception as e:
            print(f"[AI PANEL] Error clearing slot {slot_num}: {e}")
        return False

    def draw_ai_panel(self):
        if not self.show_ai_panel:
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
        
        # Panel dimensions and position
        panel_width = 600
        panel_height = 700  # Increased height to accommodate all elements
        panel_x = (self.display[0] - panel_width) // 2
        panel_y = max(20, (self.display[1] - panel_height) // 2)  # Ensure panel is at least 20px from top
        
        # Initialize slot rects if not already done
        if not hasattr(self, 'ai_slot_rects'):
            self.ai_slot_rects = []
            slot_width = 60  # Increased width for better visibility
            slot_height = 35  # Increased height for better visibility
            slot_margin = 10
            delete_height = 20  # Height of delete button
            vertical_spacing = 5  # Space between slot and delete button
            
            start_x = panel_x + 30  # Slightly more left margin
            start_y = panel_y + 380  # Start a bit higher to make room for larger buttons
            
            # Create 20 slot buttons in a 5x4 grid
            for i in range(20):
                row = i // 5
                col = i % 5
                x = start_x + col * (slot_width + slot_margin)
                y = start_y + row * (slot_height + delete_height + vertical_spacing + slot_margin)
                
                # Main slot button
                self.ai_slot_rects.append(pygame.Rect(x, y, slot_width, slot_height))
                
                # Store delete button rects (one for each slot)
                if not hasattr(self, 'ai_delete_rects'):
                    self.ai_delete_rects = []
                self.ai_delete_rects.append(pygame.Rect(x, y + slot_height + vertical_spacing, slot_width, delete_height))
        
        # Draw panel background (semi-transparent dark)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.1, 0.1, 0.15, 0.9)  # Dark semi-transparent background
        glBegin(GL_QUADS)
        glVertex2f(panel_x, panel_y)
        glVertex2f(panel_x + panel_width, panel_y)
        glVertex2f(panel_x + panel_width, panel_y + panel_height)
        glVertex2f(panel_x, panel_y + panel_height)
        glEnd()
        
        # Draw panel border
        glLineWidth(2.0)
        glColor3f(0.3, 0.5, 0.8)  # Light blue border
        glBegin(GL_LINE_LOOP)
        glVertex2f(panel_x, panel_y)
        glVertex2f(panel_x + panel_width, panel_y)
        glVertex2f(panel_x + panel_width, panel_y + panel_height)
        glVertex2f(panel_x, panel_y + panel_height)
        glEnd()
        
        # Draw header
        header_height = 40
        glColor3f(0.15, 0.15, 0.2)  # Slightly lighter than panel background
        glBegin(GL_QUADS)
        glVertex2f(panel_x, panel_y)
        glVertex2f(panel_x + panel_width, panel_y)
        glVertex2f(panel_x + panel_width, panel_y + header_height)
        glVertex2f(panel_x, panel_y + header_height)
        glEnd()
        
        # Draw title
        self.draw_text("AI Structure Generator", panel_x + 15, panel_y + 20, 20, (0.9, 0.9, 0.9))
        
        # Slot selection
        slot_x = panel_x + 20
        slot_y = panel_y + 60
        slot_width = 100
        slot_height = 30
        
        # Draw slot selection label
        self.draw_text("Slot #:", slot_x, slot_y + 8, 16, (0.9, 0.9, 0.9))
        
        # Slot input field
        slot_input_x = slot_x + 50
        slot_input_rect = pygame.Rect(slot_input_x, slot_y, 50, slot_height)
        
        # Draw slot input background
        glColor3f(0.15, 0.15, 0.2)
        glBegin(GL_QUADS)
        glVertex2f(slot_input_x, slot_y)
        glVertex2f(slot_input_x + 50, slot_y)
        glVertex2f(slot_input_x + 50, slot_y + slot_height)
        glVertex2f(slot_input_x, slot_y + slot_height)
        glEnd()
        
        # Draw slot input border
        glColor3f(0.3, 0.6, 1.0) if hasattr(self, 'slot_input_active') and self.slot_input_active else glColor3f(0.3, 0.3, 0.4)
        glBegin(GL_LINE_LOOP)
        glVertex2f(slot_input_x, slot_y)
        glVertex2f(slot_input_x + 50, slot_y)
        glVertex2f(slot_input_x + 50, slot_y + slot_height)
        glVertex2f(slot_input_x, slot_y + slot_height)
        glEnd()
        
        # Draw current slot number with cursor when active
        if hasattr(self, 'slot_input_active') and self.slot_input_active and hasattr(self, 'temp_slot_number'):
            current_slot = self.temp_slot_number
            cursor_visible = int(time.time() * 2) % 2 == 0  # Blinking cursor
            cursor = "|" if cursor_visible else ""
            self.draw_text(current_slot + cursor, slot_input_x + 5, slot_y + 8, 16, (1.0, 1.0, 1.0), center=False)
        else:
            current_slot = getattr(self, 'ai_slot_number', '1')
            self.draw_text(str(current_slot), slot_input_x + 25, slot_y + 8, 16, (1.0, 1.0, 1.0), center=True)
            
        # Draw help text for slot selection
        self.draw_text("(Press Enter to confirm slot)", slot_input_x + 60, slot_y + 8, 14, (0.7, 0.7, 0.7))
        
        # Description input area
        input_x = panel_x + 20
        input_y = panel_y + 110  # Moved down to make room for slot selection
        input_width = panel_width - 40
        input_height = 150
        
        # Draw input background
        glColor3f(0.15, 0.15, 0.2)
        glBegin(GL_QUADS)
        glVertex2f(input_x, input_y)
        glVertex2f(input_x + input_width, input_y)
        glVertex2f(input_x + input_width, input_y + input_height)
        glVertex2f(input_x, input_y + input_height)
        glEnd()
        
        # Draw input border
        border_color = (0.3, 0.6, 1.0) if self.ai_input_active else (0.3, 0.3, 0.4)
        glColor3f(*border_color)
        glBegin(GL_LINE_LOOP)
        glVertex2f(input_x, input_y)
        glVertex2f(input_x + input_width, input_y)
        glVertex2f(input_x + input_width, input_y + input_height)
        glVertex2f(input_x, input_y + input_height)
        glEnd()
        
        # Initialize font if not exists
        if not hasattr(self, '_ai_panel_font'):
            self._ai_panel_font = pygame.font.Font(None, 18)
            
        font = self._ai_panel_font
        
        # Calculate line breaks and cursor position
        cursor_pos = self.ai_input_cursor_pos
        text_before_cursor = self.ai_input_text[:cursor_pos]
        text_after_cursor = self.ai_input_text[cursor_pos:]
        
        # Split text into lines that fit within the input area
        lines = []
        current_line = ""
        
        # First, split by explicit newlines
        paragraphs = self.ai_input_text.split('\n')
        
        for para in paragraphs:
            current_line = ""
            words = para.split(' ')
            
            for word in words:
                test_line = current_line + (' ' if current_line else '') + word
                if font.size(test_line)[0] < (input_width - 20):  # 20px padding (10px each side)
                    current_line = test_line
                else:
                    if current_line:  # Only add if not empty
                        lines.append(current_line)
                    current_line = word
            
            if current_line:  # Add the last line of the paragraph
                lines.append(current_line)
            
            # Add a newline marker (empty string) to indicate paragraph break
            if para != paragraphs[-1]:  # Don't add after last paragraph
                lines.append("")
        
        # Draw each line of text and track cursor position
        line_height = font.get_linesize()
        cursor_line = 0
        cursor_x = 0
        cursor_found = False
        
        for i, line in enumerate(lines):
            if i * line_height > input_height - 10:  # Don't draw beyond input area
                break
                
            # Draw the line (skip empty lines that are just paragraph breaks)
            if line or i == 0:  # Always draw first line even if empty
                self.draw_text(line, input_x + 5, input_y + 5 + i * line_height, 18, (0.9, 0.9, 0.9))
            
            # Check if cursor is in this line
            if not cursor_found and cursor_pos > 0:
                # Calculate how many characters are in previous lines
                chars_in_prev_lines = sum(len(l) + 1 for l in lines[:i])  # +1 for the newline
                line_start = chars_in_prev_lines
                line_end = line_start + len(line)
                
                if line_start <= cursor_pos <= line_end:
                    cursor_line = i
                    # Calculate x position of cursor within this line
                    cursor_x = font.size(line[:cursor_pos - line_start])[0]
                    cursor_found = True
        
        # Store cursor position for drawing
        if hasattr(self, 'ai_input_active') and self.ai_input_active:
            self.cursor_draw_x = input_x + 5 + cursor_x
            self.cursor_draw_y = input_y + 5 + cursor_line * line_height
        
        # Draw cursor if input is active
        if hasattr(self, 'ai_input_active') and self.ai_input_active and time.time() % 1 > 0.5:
            if hasattr(self, 'cursor_draw_x') and hasattr(self, 'cursor_draw_y'):
                glLineWidth(1.5)
                glColor3f(1.0, 1.0, 1.0)
                glBegin(GL_LINES)
                glVertex2f(self.cursor_draw_x, self.cursor_draw_y)
                glVertex2f(self.cursor_draw_x, self.cursor_draw_y + 15)
                glEnd()
        
        # Button dimensions and positions
        button_width = 150
        button_height = 35
        button_y = panel_y + 280  # Positioned above the save slots
        
        # Generate button
        generate_x = panel_x + 30
        self.ai_generate_rect = pygame.Rect(generate_x, button_y, button_width, button_height)
        
        # Button background - grayed out when generating
        if self.is_fetching_genai:
            glColor3f(0.3, 0.3, 0.3)  # Grayed out when generating
        elif self.ai_generate_rect.collidepoint(pygame.mouse.get_pos()):
            glColor3f(0.2, 0.6, 0.2)  # Lighter green when hovered
        else:
            glColor3f(0.1, 0.5, 0.1)  # Dark green
            
        glBegin(GL_QUADS)
        glVertex2f(generate_x, button_y)
        glVertex2f(generate_x + button_width, button_y)
        glVertex2f(generate_x + button_width, button_y + button_height)
        glVertex2f(generate_x, button_y + button_height)
        glEnd()
        
        # Button border
        glColor3f(0.3, 0.8, 0.3)
        glBegin(GL_LINE_LOOP)
        glVertex2f(generate_x, button_y)
        glVertex2f(generate_x + button_width, button_y)
        glVertex2f(generate_x + button_width, button_y + button_height)
        glVertex2f(generate_x, button_y + button_height)
        glEnd()
        
        # Button text - show loading spinner if generating
        if self.is_fetching_genai:
            # Draw loading spinner
            spinner_radius = 10
            spinner_center_x = generate_x + button_width // 2 - 20
            spinner_center_y = button_y + button_height // 2
            spinner_angle = (time.time() * 5) % (2 * 3.14159)  # Rotating angle
            
            # Draw spinner circle outline
            glLineWidth(2.0)
            glColor3f(0.8, 0.8, 0.8)
            glBegin(GL_LINE_LOOP)
            for i in range(12):
                angle = i * (2 * 3.14159 / 12)
                x = spinner_center_x + spinner_radius * math.cos(angle)
                y = spinner_center_y + spinner_radius * math.sin(angle)
                glVertex2f(x, y)
            glEnd()
            
            # Draw spinner arm
            end_x = spinner_center_x + spinner_radius * math.cos(spinner_angle)
            end_y = spinner_center_y + spinner_radius * math.sin(spinner_angle)
            glBegin(GL_LINES)
            glVertex2f(spinner_center_x, spinner_center_y)
            glVertex2f(end_x, end_y)
            glEnd()
            
            # Draw "Generating..." text next to spinner
            self.draw_text("Generating...", generate_x + button_width // 2 , 
                          button_y + button_height // 2 - 5, 18, (0.9, 0.9, 0.9), center=True)
        else:
            # Normal generate button text
            self.draw_text("Generate", generate_x + button_width//2, button_y + button_height//2 - 5, 
                          20, (1.0, 1.0, 1.0), center=True)
        
        # Close button
        close_x = panel_x + panel_width - 30 - button_width
        self.ai_close_rect = pygame.Rect(close_x, button_y, button_width, button_height)
        
        # Button background
        if self.ai_close_rect.collidepoint(pygame.mouse.get_pos()):
            glColor3f(0.7, 0.2, 0.2)  # Lighter red when hovered
        else:
            glColor3f(0.5, 0.1, 0.1)  # Dark red
            
        glBegin(GL_QUADS)
        glVertex2f(close_x, button_y)
        glVertex2f(close_x + button_width, button_y)
        glVertex2f(close_x + button_width, button_y + button_height)
        glVertex2f(close_x, button_y + button_height)
        glEnd()
        
        # Button border
        glColor3f(0.9, 0.3, 0.3)
        glBegin(GL_LINE_LOOP)
        glVertex2f(close_x, button_y)
        glVertex2f(close_x + button_width, button_y)
        glVertex2f(close_x + button_width, button_y + button_height)
        glVertex2f(close_x, button_y + button_height)
        glEnd()
        
        # Button text
        self.draw_text("Close", close_x + button_width//2, button_y + button_height//2 - 5, 
                      20, (1.0, 1.0, 1.0), center=True)
        
        # Draw save slot buttons (for spawning only)
        if hasattr(self, 'ai_slot_rects'):
            # Draw save slots title
            self.draw_text("Load Structure (Click to spawn):", panel_x + 20, panel_y + 360, 16, (0.9, 0.9, 0.9))
            
            # Draw each slot button
            for i, rect in enumerate(self.ai_slot_rects):
                # Check if this slot has a saved structure
                slot_has_content = self.ai_handler and hasattr(self.ai_handler, 'save_exists') and \
                                 self.ai_handler.save_exists(f'ai_save_{i+1:02d}.json')
                
                # Button background - only show filled slots as clickable
                if slot_has_content:
                    if rect.collidepoint(pygame.mouse.get_pos()):
                        glColor3f(0.3, 0.4, 0.6)  # Lighter blue when hovered
                    else:
                        glColor3f(0.2, 0.3, 0.5)  # Dark blue
                    
                    glBegin(GL_QUADS)
                    glVertex2f(rect.x, rect.y)
                    glVertex2f(rect.x + rect.width, rect.y)
                    glVertex2f(rect.x + rect.width, rect.y + rect.height)
                    glVertex2f(rect.x, rect.y + rect.height)
                    glEnd()
                    
                    # Button border
                    glColor3f(0.4, 0.8, 0.4)  # Green border for filled slots
                    glBegin(GL_LINE_LOOP)
                    glVertex2f(rect.x, rect.y)
                    glVertex2f(rect.x + rect.width, rect.y)
                    glVertex2f(rect.x + rect.width, rect.y + rect.height)
                    glVertex2f(rect.x, rect.y + rect.height)
                    glEnd()
                    
                    # Slot number (1-20)
                    self.draw_text(str(i+1), rect.x + rect.width//2, rect.y + rect.height//2 - 2, 
                                 16, (1.0, 1.0, 1.0), center=True)
                    
                    # Draw delete button for this slot
                    delete_rect = self.ai_delete_rects[i]
                    
                    # Button background - red when hovered, darker red when not
                    if delete_rect.collidepoint(pygame.mouse.get_pos()):
                        glColor3f(0.9, 0.3, 0.3)  # Bright red when hovered
                        if pygame.mouse.get_pressed()[0]:  # If left mouse button is pressed
                            glColor3f(0.7, 0.2, 0.2)  # Darker red when clicked
                    else:
                        glColor3f(0.7, 0.2, 0.2)  # Darker red normally
                    
                    glBegin(GL_QUADS)
                    glVertex2f(delete_rect.x, delete_rect.y)
                    glVertex2f(delete_rect.x + delete_rect.width, delete_rect.y)
                    glVertex2f(delete_rect.x + delete_rect.width, delete_rect.y + delete_rect.height)
                    glVertex2f(delete_rect.x, delete_rect.y + delete_rect.height)
                    glEnd()
                    
                    # Button border
                    glColor3f(1.0, 1.0, 1.0)  # White border
                    glBegin(GL_LINE_LOOP)
                    glVertex2f(delete_rect.x, delete_rect.y)
                    glVertex2f(delete_rect.x + delete_rect.width, delete_rect.y)
                    glVertex2f(delete_rect.x + delete_rect.width, delete_rect.y + delete_rect.height)
                    glVertex2f(delete_rect.x, delete_rect.y + delete_rect.height)
                    glEnd()
                    
                    # Delete text
                    self.draw_text("Delete", delete_rect.x + delete_rect.width//2, 
                                 delete_rect.y + delete_rect.height//2 - 8, 
                                 14, (1.0, 1.0, 1.0), center=True)
                else:
                    # Draw empty slot as grayed out
                    glColor3f(0.15, 0.15, 0.2)
                    glBegin(GL_QUADS)
                    glVertex2f(rect.x, rect.y)
                    glVertex2f(rect.x + rect.width, rect.y)
                    glVertex2f(rect.x + rect.width, rect.y + rect.height)
                    glVertex2f(rect.x, rect.y + rect.height)
                    glEnd()
                    
                    # Gray border for empty slots
                    glColor3f(0.3, 0.3, 0.3)
                    glBegin(GL_LINE_LOOP)
                    glVertex2f(rect.x, rect.y)
                    glVertex2f(rect.x + rect.width, rect.y)
                    glVertex2f(rect.x + rect.width, rect.y + rect.height)
                    glVertex2f(rect.x, rect.y + rect.height)
                    glEnd()
                    
                    # Slot number (1-20) in gray
                    self.draw_text(str(i+1), rect.x + rect.width//2, rect.y + rect.height//2 - 2, 
                                 16, (0.5, 0.5, 0.5), center=True)
                    
                    # Draw disabled delete button for empty slots
                    delete_rect = self.ai_delete_rects[i]
                    
                    # Grayed out delete button
                    glColor3f(0.15, 0.15, 0.2)  # Match panel background
                    glBegin(GL_QUADS)
                    glVertex2f(delete_rect.x, delete_rect.y)
                    glVertex2f(delete_rect.x + delete_rect.width, delete_rect.y)
                    glVertex2f(delete_rect.x + delete_rect.width, delete_rect.y + delete_rect.height)
                    glVertex2f(delete_rect.x, delete_rect.y + delete_rect.height)
                    glEnd()
                    
                    # Gray border for disabled delete button
                    glColor3f(0.3, 0.3, 0.3)
                    glBegin(GL_LINE_LOOP)
                    glVertex2f(delete_rect.x, delete_rect.y)
                    glVertex2f(delete_rect.x + delete_rect.width, delete_rect.y)
                    glVertex2f(delete_rect.x + delete_rect.width, delete_rect.y + delete_rect.height)
                    glVertex2f(delete_rect.x, delete_rect.y + delete_rect.height)
                    glEnd()
                    
                    # Grayed out delete text
                    self.draw_text("Delete", delete_rect.x + delete_rect.width//2, 
                                 delete_rect.y + delete_rect.height//2 - 8, 
                                 14, (0.4, 0.4, 0.4), center=True)
        
        # Status message
        if self.ai_status_text and self.ai_status_timer > 0:
            status_y = panel_y + panel_height - 30
            self.draw_text(self.ai_status_text, panel_x + panel_width//2, status_y, 
                          18, (1.0, 1.0, 0.5), center=True)
            if self.ai_status_timer > 0:
                self.ai_status_timer -= 1
        
        # Re-enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Restore the projection and modelview matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        font = pygame.font.SysFont('Arial', 18)
        words = [word.split(' ') for word in self.ai_input_text.splitlines()]
        space = font.size(' ')[0]
        x, y = self.display[0]//2 - 280, self.display[1]//2 - 80
        max_width = 560
        
        for line in words:
            for word in line:
                word_surface = font.render(word, True, (255, 255, 255))
                word_width, word_height = word_surface.get_size()
                if x + word_width >= self.display[0]//2 + 280:
                    x = self.display[0]//2 - 280
                    y += word_height
                self.screen.blit(word_surface, (x, y))
                x += word_width + space
            x = self.display[0]//2 - 280
            y += word_height
        
        # Draw cursor if input is active
        if self.ai_input_active and time.time() % 1 > 0.5:
            cursor_x = x + 5 if x > self.display[0]//2 - 280 else self.display[0]//2 - 275
            pygame.draw.line(self.screen, (255, 255, 255), 
                           (cursor_x, y + 5), 
                           (cursor_x, y + 20), 2)
        
        # Status message is already drawn above with OpenGL
        # No need to draw it again with Pygame
    
    def handle_input(self):
        # Handle AI panel events first if it's visible
        if self.show_ai_panel:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                if self.handle_ai_panel_events(event):
                    continue  # Event was handled by AI panel
                
                # Forward other events to the rest of the input handling
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.show_ai_panel = False
                        continue
                    
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
                # Handle AI panel key events
                if self.show_ai_panel:
                    self.handle_ai_panel_events(event)
                    continue
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
                                self.show_save_panel = False
                                self.show_ai_panel = False
                        # Check if second panel button was clicked
                        elif self.panel2_button_rect.collidepoint(event.pos):
                            self.show_side_panel2 = not self.show_side_panel2
                            if self.show_side_panel2:
                                self.show_side_panel = False
                                self.show_save_panel = False
                                self.show_player_panel = False
                                self.show_ai_panel = False
                        # Check if player button was clicked
                        elif hasattr(self, 'player_button_rect') and self.player_button_rect.collidepoint(event.pos):
                            self.show_player_panel = not self.show_player_panel
                            if self.show_player_panel:
                                self.show_side_panel = False
                                self.show_side_panel2 = False
                                self.show_save_panel = False
                                self.show_ai_panel = False
                        # Check if save button was clicked
                        elif self.save_panel_button_rect.collidepoint(event.pos):
                            self.show_save_panel = not self.show_save_panel
                            if self.show_save_panel:
                                self.show_side_panel = False
                                self.show_side_panel2 = False                             
                                self.show_player_panel = False
                        # Check if AI button was clicked
                        elif self.ai_panel_button_rect.collidepoint(event.pos):
                            self.show_ai_panel = not self.show_ai_panel
                            if self.show_ai_panel:
                                self.show_side_panel = False
                                self.show_side_panel2 = False
                                self.show_save_panel = False
                                # Initialize AI panel rects
                                self._init_ai_panel_rects()
                                self.show_player_panel = False
                        # Check for clicks on save/load buttons
                        elif hasattr(self, 'show_save_panel') and self.show_save_panel:
                            # Check save buttons
                            for slot_num, rect in self.save_buttons.items():
                                if rect.collidepoint(event.pos):
                                    if self.save_game_state(slot_num):
                                        print(f"Game saved to slot {slot_num}")
                                    else:
                                        print(f"Failed to save to slot {slot_num}")
                                    break
                            
                            # Check load buttons
                            for slot_num, rect in self.load_buttons.items():
                                if rect.collidepoint(event.pos):
                                    save_path = os.path.join(self.saves_dir, f'save_{slot_num}.json')
                                    if os.path.exists(save_path):
                                        if self.load_game_state(slot_num):
                                            print(f"Game loaded from slot {slot_num}")
                                        else:
                                            print(f"Failed to load from slot {slot_num}")
                                    break       
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
        
        # Handle continuous key states for movement - only if AI panel is not open
        if not self.show_ai_panel:
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
                
            # Q key moves up 
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
        
        # Draw AI panel if visible
        if self.show_ai_panel:
            # Switch to orthographic projection for UI
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            # Draw the AI panel
            self.draw_ai_panel()
            
            # Switch back to perspective projection
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
        
        # Draw other UI elements
        if self.show_side_panel:
            self.draw_side_panel()
        elif self.show_side_panel2:
            self.draw_side_panel2()
        elif self.show_save_panel:
            self.draw_save_panel()
        elif self.show_player_panel:
            self.draw_player_panel()
        elif self.show_ai_panel:
            self.draw_ai_panel()
        self.draw_crosshair()
        self.draw_menu_buttons()
        self.draw_reset_button()
        self.draw_lock_button()
    
    def _run_async(self, coro):
        """Helper method to run an async function from sync context"""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def run(self):
        # Initialize mouse settings
        pygame.mouse.set_visible(False)
        pygame.mouse.set_pos(self.display[0] // 2, self.display[1] // 2)  # Center mouse
        
        try:
            while self.running:
                # Handle input and update
                self.handle_input()
                
                # Run pending asyncio tasks
                self.loop.call_soon(self.loop.stop)
                self.loop.run_forever()
                
                self.update()  # Update simulation state
                self.update_camera()
                
                # Render the scene
                self.render_scene()
                
                # Update the display
                pygame.display.flip()
                
                # Cap the frame rate
                self.clock.tick(self.fps)
                
                # Check if AI generation is complete
                if hasattr(self, 'ai_generation_future') and self.ai_generation_future and self.ai_generation_future.done():
                    try:
                        result = self.ai_generation_future.result()
                        # Handle the result if needed
                        if result:
                            print("AI Generation completed successfully!")
                    except Exception as e:
                        print(f"Error in AI generation: {e}")
                    finally:
                        self.ai_generation_future = None
        finally:
            # Clean up
            if hasattr(self, 'physics'):
                self.physics.cleanup()
            self.loop.close()
            pygame.quit()

if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()
