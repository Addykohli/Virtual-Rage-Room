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
        self.move_speed = 5.0
        self.mouse_sensitivity = 0.2

    def move_forward(self, distance):
        rad = math.radians(self.rotation_y-30)
        self.x -= math.sin(rad) * distance
        self.z -= math.cos(rad) * distance

    def move_right(self, distance):
        rad = math.radians(self.rotation_y + 60)
        self.x -= math.sin(rad) * distance
        self.z -= math.cos(rad) * distance

    def move_left(self, distance):
        self.move_right(-distance)

    def move_up(self, distance):
        self.y += distance
