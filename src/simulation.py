import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import os
import sys
import math

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.player import Player
from models.obj_loader import OBJ

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
        self.display = (WINDOW_WIDTH, WINDOW_HEIGHT)
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("Simulation")
        
        # Mouse control settings
        self.mouse_grabbed = True
        pygame.event.set_grab(True)  # Grab the cursor for relative motion
        pygame.mouse.set_visible(False)  # Hide cursor
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.VIDEORESIZE])
        self.relative_mouse_mode = True
        self.last_mouse_pos = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)  # Initialize last mouse position
        
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
        self.model_lib = ModelLibrary()
        self.currPlayerModel = self.model_lib.load_model('Sci Fi AA Turret')
        
        # Set up the perspective
        self.setup_projection()
        
        # Enable depth testing and other OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Create player with vertical offset (3x player height below z=0)
        player_height = 1.0  # Assuming the player's height is roughly 1 unit
        self.player = Player(y=1.0, z=-5 * player_height)  # Slightly above the ground
        
        # Camera settings
        self.camera_distance = 5.0
        self.camera_height = 2.0
        self.min_zoom = 2.0  # Minimum zoom distance
        self.max_zoom = 20.0  # Maximum zoom distance
        self.zoom_speed = 0.2  # Zoom speed multiplier
        
        # Mouse control
        self.mouse_x, self.mouse_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        pygame.mouse.set_pos(self.mouse_x, self.mouse_y)
        
        # Simulation variables
        self.running = True
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.delta_time = 0.0
        self.show_side_panel = False  # Initially hidden
        
        # Colors
        self.colors = {
            'grid': (0.3, 0.3, 0.3),
            'player': (0.0, 1.0, 0.0),
            'background': (0.1, 0.1, 0.15),
            'menu_button': (0.2, 0.2, 0.25),
            'menu_button_hover': (0.3, 0.3, 0.35),
            'lock_button': (0.2, 0.2, 0.25),  # Default dark color
            'lock_button_active': (0.2, 0.6, 0.2),  # Green when active
            'lock_button_hover': (0.3, 0.3, 0.35)  # Lighter on hover
        }
        
        # UI elements
        button_size = 40
        margin = 10
        button_bg_width = 60
        
        # Menu button (top right)
        self.menu_button_rect = pygame.Rect(
            self.display[0] - button_size - margin,
            margin,
            button_size,
            button_size
        )
        
        # Lock button (bottom right)
        self.lock_button_rect = pygame.Rect(
            self.display[0] - button_size - margin,
            self.display[1] - button_size - margin,
            button_size,
            button_size
        )
    
    def setup_projection(self):
        glViewport(0, 0, self.display[0], self.display[1])
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(FOV, (self.display[0] / self.display[1]), NEAR_PLANE, FAR_PLANE)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def draw_grid(self):
        glBegin(GL_LINES)
        glColor3f(*self.colors['grid'])
        
        # Draw grid lines on the XZ plane (offset below z=0)
        grid_z_offset = 0.0  # Same as player's initial z-offset
        grid_height = 1.0  # Height to raise the grid by
        for i in range(-GRID_SIZE, GRID_SIZE + 1):
            # Lines along X axis
            glVertex3f(i * TILE_SIZE, grid_height, -GRID_SIZE * TILE_SIZE + grid_z_offset)
            glVertex3f(i * TILE_SIZE, grid_height, GRID_SIZE * TILE_SIZE + grid_z_offset)
            # Lines along Z axis
            glVertex3f(-GRID_SIZE * TILE_SIZE, grid_height, i * TILE_SIZE + grid_z_offset)
            glVertex3f(GRID_SIZE * TILE_SIZE, grid_height, i * TILE_SIZE + grid_z_offset)
        
        # Draw some vertical lines to show height (offset below z=0)
        grid_z_offset = -5.0  # Same as player's initial z-offset
        for i in range(0, 5):
            glVertex3f(0, i + grid_height, grid_z_offset)
            glVertex3f(0, i + grid_height, grid_z_offset + 1)
            
        glEnd()
    
    def draw_player(self):
        glPushMatrix()
        # Position the model at the player's position
        glTranslatef(self.player.x, self.player.y, self.player.z)
        
        # Apply yaw rotation to make model follow left/right camera movement
        # with an additional 45-degree left rotation
        glRotatef(self.player.rotation_y + 45, 0, 1, 0)  # Yaw with 45-degree left offset
        
        # Apply model's own rotation (roll)
        glRotatef(self.player.rotation_z, 0, 0, 1)  # Roll (tilt) around Z axis
        
        # Apply model's offset (x, y, z)
        if hasattr(self.currPlayerModel, 'metadata') and 'offset' in self.currPlayerModel.metadata:
            offset = self.currPlayerModel.metadata['offset']
            glTranslatef(offset[0], offset[1], offset[2])
        
        # Apply additional scale if needed
        model_scale = 0.5
        glScalef(model_scale, model_scale, model_scale)
        
        # Fixed model orientation (Y forward, X right, Z up)
        glRotatef(-90, 1, 0, 0)  # Rotate -90 degrees around X to make Y point forward
        glRotatef(90, 1, 0, 0)   # Additional 90 degrees around X (red axis)
        
        # Draw the model with its own scaling and centering
        glColor3f(0.3, 0.3, 0.3)  # Dark gray color for the model
        self.currPlayerModel.render()
        
        # Draw coordinate axes for debugging (without text)
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)  # Make the lines thicker
        glBegin(GL_LINES)
        # X axis (Red) - Pitch
        glColor3f(1, 0, 0)  # Red
        glVertex3f(0, 0, 0)
        glVertex3f(2, 0, 0)
        # Y axis (Green) - Yaw
        glColor3f(0, 1, 0)  # Green
        glVertex3f(0, 0, 0)
        glVertex3f(0, 2, 0)
        # Z axis (Blue) - Roll
        glColor3f(0, 0, 1)  # Blue
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 2)
        glEnd()
        glLineWidth(1.0)  # Reset line width
        glEnable(GL_LIGHTING)
        
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
    
    def draw_menu_button(self):
        # Define button and panel sizes
        button_size = 40
        margin = 10
        button_bg_width = 60  # Width of the vertical strip for the button
        panel_width = 250  # Panel will be this wide
        
        # Position button on right side with margin
        self.menu_button_rect.x = self.display[0] - button_size - margin
        
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
        
        # Check if mouse is hovering over the menu button
        mouse_x, mouse_y = pygame.mouse.get_pos()
        is_hovered = self.menu_button_rect.collidepoint(mouse_x, mouse_y)
        
        # Draw vertical background for menu button (narrow strip on the right)
        glColor3f(0.1, 0.1, 0.15)  # Panel color
        glBegin(GL_QUADS)
        glVertex2f(self.display[0] - button_bg_width, 0)
        glVertex2f(self.display[0], 0)
        glVertex2f(self.display[0], self.display[1])
        glVertex2f(self.display[0] - button_bg_width, self.display[1])
        glEnd()
        
        # Draw menu button background
        button_color = self.colors['menu_button_hover'] if is_hovered else self.colors['menu_button']
        glColor3f(*button_color)
        glBegin(GL_QUADS)
        glVertex2f(self.menu_button_rect.left, self.menu_button_rect.top)
        glVertex2f(self.menu_button_rect.right, self.menu_button_rect.top)
        glVertex2f(self.menu_button_rect.right, self.menu_button_rect.bottom)
        glVertex2f(self.menu_button_rect.left, self.menu_button_rect.bottom)
        glEnd()
        
        # Draw menu icon (three horizontal lines)
        glColor3f(1.0, 1.0, 1.0)
        line_height = 2
        line_spacing = 5
        line_width = 20
        
        for i in range(3):
            y = self.menu_button_rect.centery - 6 + i * line_spacing
            glBegin(GL_QUADS)
            glVertex2f(self.menu_button_rect.centerx - line_width//2, y)
            glVertex2f(self.menu_button_rect.centerx + line_width//2, y)
            glVertex2f(self.menu_button_rect.centerx + line_width//2, y + line_height)
            glVertex2f(self.menu_button_rect.centerx - line_width//2, y + line_height)
            glEnd()
        
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
        
        # Re-enable depth testing
        glEnable(GL_DEPTH_TEST)
        
        # Restore the projection and modelview matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def update_camera(self):
        glLoadIdentity()
        
        # Calculate camera position behind and above the player
        rad_yaw = math.radians(self.player.rotation_y)
        rad_pitch = math.radians(self.player.rotation_x)
        
        # Calculate camera offset from player
        horizontal_distance = self.camera_distance * math.cos(rad_pitch)
        vertical_offset = self.camera_distance * math.sin(rad_pitch)
        
        # Calculate camera position (2 units to the left of the player)
        camera_x = self.player.x - math.sin(rad_yaw) * horizontal_distance + math.cos(rad_yaw) * 2.0
        camera_z = self.player.z - math.cos(rad_yaw) * horizontal_distance - math.sin(rad_yaw) * 2.0
        camera_y = self.player.y + self.camera_height + vertical_offset
        
        # Calculate look-at point (2 units to the right of the player)
        look_x = self.player.x + math.sin(rad_yaw) * 2.0 - math.cos(rad_yaw) * 2.0
        look_z = self.player.z + math.cos(rad_yaw) * 2.0 + math.sin(rad_yaw) * 2.0
        look_y = self.player.y + 1.0  # Look slightly above the player
        
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
        size = 5
        center_x, center_y = self.display[0] // 2, self.display[1] // 2
        glColor3f(1.0, 1.0, 1.0)  # White color
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # Horizontal line
        glVertex2f(center_x - size, center_y)
        glVertex2f(center_x + size, center_y)
        # Vertical line
        glVertex2f(center_x, center_y - size)
        glVertex2f(center_x, center_y + size)
        glEnd()
        
        # Restore OpenGL state
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_TAB:
                    # Toggle mouse grab to allow clicking UI elements
                    self.toggle_mouse_grab()
                elif event.key == pygame.K_m:  # Add M key to toggle side panel
                    self.show_side_panel = not self.show_side_panel
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Check if menu button was clicked
                    if self.menu_button_rect.collidepoint(event.pos):
                        self.show_side_panel = not self.show_side_panel
                    # Check if lock button was clicked
                    elif self.lock_button_rect.collidepoint(event.pos):
                        self.toggle_mouse_grab()
            elif event.type == pygame.MOUSEWHEEL:
                # Check if CTRL is being held down
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # Zoom in/out based on scroll direction
                    zoom_amount = event.y * self.zoom_speed
                    self.camera_distance = max(self.min_zoom, min(self.max_zoom, self.camera_distance - zoom_amount))
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                self.display = (event.w, event.h)
                self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL | RESIZABLE)
                self.setup_projection()
                # Update menu button position
                button_size = 40
                margin = 10
                self.menu_button_rect.x = self.display[0] - button_size - margin

    def handle_input(self):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_TAB:
                    # Toggle mouse grab to allow clicking UI elements
                    self.toggle_mouse_grab()
                elif event.key == pygame.K_m:  # Add M key to toggle side panel
                    self.show_side_panel = not self.show_side_panel
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Check if menu button was clicked
                    if self.menu_button_rect.collidepoint(event.pos):
                        self.show_side_panel = not self.show_side_panel
                    # Check if lock button was clicked
                    elif self.lock_button_rect.collidepoint(event.pos):
                        self.toggle_mouse_grab()
            elif event.type == pygame.MOUSEWHEEL:
                # Check if CTRL is being held down
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    # Zoom in/out based on scroll direction
                    zoom_amount = event.y * self.zoom_speed
                    self.camera_distance = max(self.min_zoom, min(self.max_zoom, self.camera_distance - zoom_amount))
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                self.display = (event.w, event.h)
                self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL | RESIZABLE)
                self.setup_projection()
                # Update menu button position
                button_size = 40
                margin = 10
                self.menu_button_rect.x = self.display[0] - button_size - margin
            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_grabbed:
                    # Only process mouse movement if mouse is grabbed
                    rel_x, rel_y = event.rel
                    
                    # Update player rotation based on relative movement
                    if rel_x != 0 or rel_y != 0:
                        # Apply rotation based on relative movement
                        self.player.rotation_y -= rel_x * self.player.mouse_sensitivity * 0.5
                        self.player.rotation_x += rel_y * self.player.mouse_sensitivity * 0.5
                        
                        # Keep rotations within 0-360 range to prevent floating point overflow
                        self.player.rotation_y %= 360
                        self.player.rotation_x %= 360
        
        # Get time since last frame
        self.delta_time = self.clock.tick(self.fps) / 1000.0
        
        # Handle continuous key states for movement
        keys = pygame.key.get_pressed()
        move_speed = self.player.move_speed * self.delta_time
        
        # Forward/backward movement (relative to box orientation)
        if keys[pygame.K_s]:  # Changed from K_w to K_s
            self.player.move_forward(move_speed)
        if keys[pygame.K_w]:  # Changed from K_s to K_w
            self.player.move_forward(-move_speed)
            
        # Strafing (left/right movement relative to box orientation)
        if keys[pygame.K_d]:  # Changed from K_a to K_d
            self.player.move_right(move_speed)
        if keys[pygame.K_a]:  # Changed from K_d to K_a
            self.player.move_right(-move_speed)   # D moves left
            
        # Vertical movement (up/down in world space)
        if keys[pygame.K_SPACE]:
            self.player.move_up(move_speed)
        if keys[pygame.K_LSHIFT]:
            self.player.move_up(-move_speed)
    
    def run(self):
        # Initialize mouse settings
        pygame.mouse.set_visible(False)
        pygame.mouse.set_pos(self.display[0] // 2, self.display[1] // 2)  # Center mouse
        
        while self.running:
            # Handle input and update
            self.handle_input()
            self.update_camera()
            
            # Clear the screen and depth buffer
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Draw the scene
            self.draw_grid()
            self.draw_player()
            self.draw_side_panel()
            self.draw_crosshair()
            self.draw_menu_button()
            self.draw_lock_button()
            
            # Update the display
            pygame.display.flip()
            # Cap the frame rate
            self.clock.tick(self.fps)
        
        # Clean up
        pygame.quit()

if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()
