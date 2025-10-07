import math

class Player:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.speed = 0.1
        self.rotation_x = 0  # Vertical rotation (pitch)
        self.rotation_y = 0  # Horizontal rotation (yaw)
        self.rotation_z = 0  # Vertical rotation (roll)
        self.move_speed = 10.0
        self.mouse_sensitivity = 0.2
        self.min_pitch = -100.0  # Minimum pitch angle (looking up)
        self.max_pitch = 70.0   # Maximum pitch angle (looking down)
        self.camera_height = 1.8  # Height of camera above player position
        
        # Physics properties
        self.velocity_y = 0
        self.gravity = -20.0  # Increased gravity for faster fall
        self.jump_force = 16.0  # Increased jump force for higher jumps
        self.is_jumping = False
        self.ground_level = 0.0  # Ground level (y-coordinate)
        self.player_height = 1.0  # Height of the player's collision box
        self.jump_cooldown = 0  # Cooldown to prevent multiple jumps in one press

    def move_forward(self, distance):
        rad = math.radians(self.rotation_y)
        new_x = self.x - math.sin(rad) * distance
        new_z = self.z - math.cos(rad) * distance
        # Check for collision with structures
        if not self.check_collision(new_x, self.y, new_z):
            self.x = new_x
            self.z = new_z

    def move_right(self, distance):
        rad = math.radians(self.rotation_y + 90)
        new_x = self.x - math.sin(rad) * distance
        new_z = self.z - math.cos(rad) * distance
        # Check for collision with structures
        if not self.check_collision(new_x, self.y, new_z):
            self.x = new_x
            self.z = new_z

    def move_left(self, distance):
        self.move_right(-distance)
        
    def jump(self):
        if not self.is_jumping and not self.jump_cooldown:
            self.velocity_y = self.jump_force
            self.is_jumping = True
            self.jump_cooldown = 0.1  # Small cooldown to prevent multiple jumps on one press
            
    def check_collision(self, x, y, z):
        """Check if the player would collide with any structure at the given position."""
        # This will be implemented in the simulation to use the physics world
        return False  # Placeholder
        
    def update_position(self, delta_time):
        # Update jump cooldown
        if self.jump_cooldown > 0:
            self.jump_cooldown = max(0, self.jump_cooldown - delta_time)
            
        # Apply gravity
        self.velocity_y += self.gravity * delta_time
        self.y += self.velocity_y * delta_time
        
        # Check for ground collision
        if self.y <= self.ground_level + self.player_height:
            self.y = self.ground_level + self.player_height
            self.velocity_y = 0
            self.is_jumping = False
            self.jump_cooldown = 0  # Reset cooldown when on ground
