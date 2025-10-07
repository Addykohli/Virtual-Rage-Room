import pybullet as p
import numpy as np

class PhysicsWorld:
    def __init__(self, gravity=(0, -9.81, 0), grid_height=0.0):
        # Use DIRECT mode for headless simulation (faster)
        self.physicsClient = p.connect(p.DIRECT)  # or p.GUI for visualization
        p.setGravity(*gravity)
        p.setTimeStep(1/60.0)  # 60 FPS physics simulation
        
        # Store references to physics bodies
        self.structures = {}
        self.next_structure_id = 0
        self.grid_height = grid_height
        self.stiff_structures = set()  # Track stiff structures
        
        # Enable contact points for collision detection
        p.setPhysicsEngineParameter(enableConeFriction=1)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        
        # Create a ground plane at the specified grid height
        # For PyBullet, the plane is always at y=0, so we'll position it using basePosition
        self.ground_plane = p.createCollisionShape(
            p.GEOM_PLANE,
            planeNormal=[0, 1, 0]  # Y-up normal
        )
        # Create a static (mass=0) multi-body for the ground plane
        # Position the plane at the specified grid height
        self.ground_id = p.createMultiBody(
            baseMass=0,  # Mass of 0 makes it static
            baseCollisionShapeIndex=self.ground_plane,
            basePosition=[0, grid_height, 0],  # Position at grid height
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])  # No rotation
        )
        
        # Set friction and restitution for the ground
        p.changeDynamics(
            self.ground_id, -1,
            lateralFriction=1.0,  # Friction in the plane
            spinningFriction=0.5,  # Resistance against spinning
            rollingFriction=0.1,   # Resistance against rolling
            restitution=0.5        # Bounciness (0.0 to 1.0)
        )
        
    def add_structure(self, position, size=(1, 1, 1), mass=1.0, color=(0.8, 0.2, 0.2, 1), rotation=None, fill='solid', metadata=None, mesh_obj=None):
        """Add a shootable structure to the physics world
        
        Args:
            position: [x, y, z] position of the structure
            size: [width, height, depth] size of the structure
            mass: mass of the structure (0 for static)
            color: [r, g, b, a] color of the structure
            rotation: [roll, pitch, yaw] in radians, or None for no rotation
            fill: type of fill for rendering ('solid', 'small_bricks', 'rust_metal', etc.)
            metadata: dictionary containing additional metadata for the structure (e.g., hit sounds)
            mesh_obj: OBJ instance for mesh-based collision (optional)
        Returns:
            int: The ID of the created structure, or None if a structure already exists at the position
        """
        # Check for existing structures at this position
        nearby = self.check_nearby_objects(position, radius=0.1)  # Small radius to check exact position
        if nearby:
            return None  # Structure already exists at this position
        
        if mesh_obj is not None:
            # Use mesh-based collision
            vertices, indices = mesh_obj.get_bullet_mesh()
            shape = p.createCollisionShape(
                p.GEOM_MESH,
                vertices=vertices,
                indices=indices,
                meshScale=[s for s in size]
            )
        else:
            # Only create collision shape, not visual shape since we'll render it ourselves
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=np.array(size)/2)
        
        # Convert rotation to quaternion if provided
        if rotation is not None:
            orientation = p.getQuaternionFromEuler(rotation)
        else:
            orientation = [0, 0, 0, 1]  # No rotation (identity quaternion)
        
        # Check if this is a 'stiff' structure (unmovable)
        is_stiff = metadata and metadata.get('stiff', False)
        
        # Create body with no visual shape
        body = p.createMultiBody(
            baseMass=0.0 if is_stiff else mass,  # Mass of 0 makes it static/immovable
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=-1,  # No visual shape
            basePosition=position,
            baseOrientation=orientation
        )
        
        structure_id = self.next_structure_id
        # Initialize structure data with provided metadata or empty dict
        structure_data = {
            'body': body,
            'size': list(size),
            'color': list(color),
            'rotation': list(rotation) if rotation is not None else [0, 0, 0],
            'angular_velocity': [0, 0, 0],  # Initialize with no rotation
            'fill': fill,  # Store the fill type for rendering
            'texture_id': None,  # Will be set by the simulation
            'metadata': metadata or {},  # Store any additional metadata
            'stiff': is_stiff,  # Mark if this is a stiff (unmovable) structure
            'mesh_obj': mesh_obj  # Store mesh_obj for rendering if needed
        }
        
        # Handle stiff structures
        if is_stiff:
            # Store the initial position and orientation
            structure_data['initial_position'] = list(position)
            structure_data['initial_orientation'] = list(orientation)
            self.stiff_structures.add(structure_id)
            
            # Check if this is a kinematic object (like the player)
            is_kinematic = metadata and metadata.get('kinematic', False)
            
            if is_kinematic:
                # For kinematic objects, set collision flags to make them solid to other objects
                p.changeDynamics(
                    body,
                    -1,  # -1 for the base
                    mass=0.0,  # Mass of 0 makes it kinematic
                    localInertiaDiagonal=[0, 0, 0],  # No inertia
                    collisionMargin=0.0  # No collision margin
                )
            else:
                # For static structures, use high damping
                p.changeDynamics(
                    body,
                    -1,  # -1 for the base
                    mass=0.0,  # Mass of 0 makes it static
                    localInertiaDiagonal=[0, 0, 0],  # No inertia
                    linearDamping=1.0,  # High damping to prevent any movement
                    angularDamping=1.0   # High damping to prevent any rotation
                )
        
        # Store the structure data
        self.structures[structure_id] = structure_data
        
        # Set angular damping to make rotations more stable
        p.changeDynamics(
            body,
            -1,  # -1 for the base
            angularDamping=0.1,  # Add some angular damping
            restitution=0.5,     # Bounciness
            lateralFriction=0.7, # Friction
            rollingFriction=0.1, # Rolling resistance
            spinningFriction=0.1 # Spinning resistance
        )
        self.next_structure_id += 1
        return self.next_structure_id - 1  # Return the ID we just used
    def step(self):
        """Step the physics simulation"""
        self.step_simulation()
        
    def get_structure_position(self, structure_id):
        """Get the position of a structure by its ID"""
        if structure_id in self.structures:
            pos, _ = p.getBasePositionAndOrientation(self.structures[structure_id]['body'])
            return pos
        return None
        
    def move_kinematic_body(self, body_id, position, orientation):
        """
        Moves a kinematic body to the specified position and orientation.
        This is used for player movement where the player is a kinematic object.
        """
        p.resetBasePositionAndOrientation(body_id, position, orientation)
        p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0])  # Ensure no residual velocity
        
    def check_nearby_objects(self, position, radius=2.0):
        """Check for objects near the given position within the specified radius
        
        Args:
            position: [x, y, z] position to check around
            radius: search radius in meters
            
        Returns:
            list: List of (object_id, distance) tuples for objects within radius
        """
        nearby_objects = []
        
        # Check against all structures
        for obj_id, info in self.structures.items():
            obj_pos, _ = p.getBasePositionAndOrientation(info['body'])
            distance = ((obj_pos[0] - position[0])**2 + 
                       (obj_pos[1] - position[1])**2 + 
                       (obj_pos[2] - position[2])**2)**0.5
            
            if distance < radius:
                nearby_objects.append((obj_id, distance))
                
        # Sort by distance (closest first)
        nearby_objects.sort(key=lambda x: x[1])
        return nearby_objects
        
    def get_structure_rotation(self, structure_id):
        """Get the current rotation of a structure as [roll, pitch, yaw] in radians"""
        if structure_id in self.structures:
            _, orn = p.getBasePositionAndOrientation(self.structures[structure_id]['body'])
            return p.getEulerFromQuaternion(orn)
        return None
        
    def get_structure_orientation(self, structure_id):
        if structure_id in self.structures:
            _, orn = p.getBasePositionAndOrientation(self.structures[structure_id]['body'])
            return orn
        return None

    def get_structure(self, structure_id):
        """Get structure data by ID."""
        structure = self.structures.get(structure_id)
        return structure

    def get_structure_properties(self, structure_id):
        if structure_id in self.structures:
            info = self.structures[structure_id]
            return {
                'size': info.get('size', [1, 1, 1]),
                'color': info.get('color', [0.8, 0.2, 0.2, 1])
            }
        return None
    def apply_impulse(self, structure_id, impulse, position=None):
        """Apply an impulse to a structure (for shooting)
        
        Args:
            structure_id: ID of the structure to apply impulse to
            impulse: [x, y, z] impulse vector in world coordinates
            position: [x, y, z] position to apply the impulse (in world coordinates)
                     If None, uses the center of mass
        """
        if structure_id in self.structures:
            body = self.structures[structure_id]['body']
            if position is None:
                position, _ = p.getBasePositionAndOrientation(body)
                
            # Apply linear impulse (force)
            p.applyExternalForce(
                body, 
                -1,  # -1 for the base
                impulse, 
                position, 
                p.WORLD_FRAME
            )
            
            # Apply some angular velocity based on where the bullet hit
            # This creates a more realistic rotation effect
            com_pos, _ = p.getBasePositionAndOrientation(body)
            r = [position[i] - com_pos[i] for i in range(3)]  # Vector from COM to hit point
            
            # Calculate torque as cross product of r and impulse
            torque = [
                r[1] * impulse[2] - r[2] * impulse[1],
                r[2] * impulse[0] - r[0] * impulse[2],
                r[0] * impulse[1] - r[1] * impulse[0]
            ]
            
            # Scale down the torque to prevent excessive spinning
            torque = [t * 0.5 for t in torque]
            
            # Apply the torque as an angular impulse
            p.applyExternalTorque(
                body,
                -1,  # -1 for the base
                torque,
                p.WORLD_FRAME
            )
    
    def cleanup(self):
        """Clean up physics resources"""
        p.disconnect()

    def ray_test(self, from_pos, to_pos):
        """Cast a ray and return the first structure hit"""
        # Perform the ray test with all objects and the ground plane
        ray = p.rayTest(from_pos, to_pos)
        
        if ray and ray[0][0] != -1:  # -1 means no hit
            hit_object_id = ray[0][0]
            hit_fraction = ray[0][2]
            hit_position = ray[0][3]
            # Check if we hit the ground plane
            if hit_object_id == self.ground_id:
                return -1, hit_position  # Special ID for ground plane
                
            # Check if we hit any other objects
            for struct_id, info in self.structures.items():
                if info['body'] == hit_object_id:
                    return struct_id, hit_position
                    
        return None, None
        
    def step_simulation(self):
        """Step the physics simulation and ensure stiff structures stay in place"""
        # First step the simulation
        p.stepSimulation()
        
        # Ensure all stiff structures stay in their initial positions/orientations
        # Skip kinematic objects (like the player) as they are controlled by code
        for structure_id in self.stiff_structures:
            if structure_id in self.structures:
                structure = self.structures[structure_id]
                
                # Skip kinematic objects - they are controlled by code, not physics
                if structure.get('metadata', {}).get('kinematic', False):
                    continue
                    
                body = structure['body']
                
                # Reset position and orientation to initial values
                p.resetBasePositionAndOrientation(
                    body,
                    structure['initial_position'],
                    structure['initial_orientation']
                )
                
                # Reset all velocities to zero
                p.resetBaseVelocity(
                    body,
                    linearVelocity=[0, 0, 0],
                    angularVelocity=[0, 0, 0]
                )
    
    def remove_structure(self, structure_id):
        """Remove a structure from the physics world"""
        if structure_id in self.structures:
            p.removeBody(self.structures[structure_id]['body'])
            if structure_id in self.stiff_structures:
                self.stiff_structures.remove(structure_id)
            del self.structures[structure_id]
    
    def add_bullet(self, position, direction, speed, radius=0.1, mass=0.1):
        # Create a sphere collision shape for the bullet
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius
        )
        
        # Create a bright white visual shape for the bullet
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[1, 1, 1, 1],  # Bright white color
            specularColor=[1, 1, 1]   # White specular highlight
        )
        
        # Create the bullet with the specified mass and position
        bullet_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        # Set initial linear velocity in the firing direction
        velocity = [d * speed for d in direction]
        p.resetBaseVelocity(bullet_id, linearVelocity=velocity)
        
        # Set bullet properties
        p.changeDynamics(
            bullet_id,
            -1,  # -1 for the base
            linearDamping=0.0,  # No air resistance
            angularDamping=0.0,  # No angular damping
            restitution=0.5,     # Some bounciness
            lateralFriction=0.0,  # No friction
            rollingFriction=0.0,  # No rolling friction
            spinningFriction=0.0  # No spinning friction
        )
        
        # Store bullet in the structures dictionary
        self.structures[self.next_structure_id] = {
            'body': bullet_id,
            'size': [radius * 2] * 3,
            'color': [1, 1, 1, 1]
        }
        self.next_structure_id += 1
        return self.next_structure_id - 1
    
    def remove_bullet(self, bullet_id):
        """Remove a bullet from the physics world"""
        self.remove_structure(bullet_id)
    
    def get_bullet_position(self, bullet_id):
        """Get the current position of a bullet"""
        if bullet_id in self.structures:
            pos, _ = p.getBasePositionAndOrientation(self.structures[bullet_id]['body'])
            return pos
        return None
