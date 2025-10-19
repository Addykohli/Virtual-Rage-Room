import math

class Player:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
        #self.speed = 0.1
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
        self.gravity = -10.0 *3 # Gravity for falling
        self.jump_height = 1.0  # Fixed jump height in units
        self.is_jumping = False
        self.ground_level = 0.0  # Ground level (y-coordinate)
        self.player_height = 1.0  # Height of the player's collision box
        self.jump_cooldown = 0.0  # Cooldown timer for jumping
    def move_forward(self, distance):
        rad = math.radians(self.rotation_y)
        new_x = self.x - math.sin(rad) * distance
        new_z = self.z - math.cos(rad) * distance
        
        # Check collision at new position
        collision = self.check_collision(new_x, self.y, new_z)
        
        if not collision:
            self.x = new_x
            self.z = new_z

    def move_right(self, distance):
        rad = math.radians(self.rotation_y + 90)
        new_x = self.x - math.sin(rad) * distance
        new_z = self.z - math.cos(rad) * distance
        
        # Check collision at new position
        collision = self.check_collision(new_x, self.y, new_z)
        
        if not collision:
            self.x = new_x
            self.z = new_z

    def move_left(self, distance):
        self.move_right(-distance)
        
    def jump(self):
        # Only allow jumping if not already jumping and cooldown is 0
        if not self.is_jumping and self.jump_cooldown <= 0:
            # Calculate initial velocity needed to reach desired height under gravity
            # Using v = sqrt(2gh) from physics
            self.velocity_y = math.sqrt(-2 * self.gravity * self.jump_height)
            self.is_jumping = True
            self.jump_cooldown = 0.4  # Small cooldown to prevent multiple jumps on one press
        
    def update_position(self, delta_time):
        # Update jump cooldown
        if self.jump_cooldown > 0:
            self.jump_cooldown = max(0, self.jump_cooldown - delta_time)

        # Check if player is standing on ground or object below
        on_ground_or_object = (
            self.y <= self.ground_level + self.player_height or
            self.check_collision(self.x, self.y - 0.01, self.z)
        )

        # Only disable gravity if standing
        if on_ground_or_object and self.velocity_y <= 0:
            self.velocity_y = 0
            self.is_jumping = False
            new_y = max(self.y, self.ground_level + self.player_height)
        else:
            # Apply gravity or jump movement
            self.velocity_y += self.gravity * delta_time
            new_y = self.y + self.velocity_y * delta_time
        # Ground collision check
        if new_y <= self.ground_level + self.player_height:
            self.y = self.ground_level + self.player_height
            self.velocity_y = 0
            self.is_jumping = False
        # Structure collision check from below
        elif self.check_collision(self.x, new_y, self.z):
            # Find the highest non-colliding position
            step = 0.1
            test_y = self.y
            while test_y < new_y and not self.check_collision(self.x, test_y, self.z):
                test_y += step
            self.y = test_y
            self.velocity_y = 0
            self.is_jumping = False
        else:
            self.y = new_y
            
            # Additional ground check after movement
            if self.y <= self.ground_level + self.player_height:
                self.y = self.ground_level + self.player_height
                self.velocity_y = 0
                self.is_jumping = False
