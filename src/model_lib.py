import os
from models.obj_loader import OBJ

class ModelLibrary:
    def __init__(self):
        # Base paths
        self.models_dir = os.path.join('models', 'model_objs')
        self.thumbnails_dir = 'thumbnails'
        
        # Ensure thumbnails directory exists
        os.makedirs(self.thumbnails_dir, exist_ok=True)
        
        # Model library
        self.available_models = {
            'player': [
                {
                    'name': 'Sci Fi AA Turret',
                    'path': os.path.join(self.models_dir, 'Sci Fi AA Turret.obj'),
                    'mesh_path': os.path.join(self.models_dir, 'Sci Fi AA Turret.obj'),
                    'thumbnail': os.path.join(self.thumbnails_dir, 'sci_fi_aa_turret.png'),
                    'type': 'player',
                    'scale': 1.5,
                    'offset': (0, 1.25, 0),  # x, y, z offset for model positioning
                    'bullet_origin': [(0.0, -0.12, 0.0)],  # x, y, z offset from model center for bullet spawn
                    'camera_height': -4.5,
                    'shoot_sound': 'turret-shoot.mp3',
                    'hit_sound': 'turret-hit.mp3',
                    'Yaw_offset': 0
                },
                {
                    'name': 'Punisher',
                    'path': os.path.join(self.models_dir, 'Untitled.obj'),
                    'mesh_path': os.path.join(self.models_dir, 'Untitled.obj'),
                    'thumbnail': os.path.join(self.thumbnails_dir, 'Untitled.png'),
                    'type': 'player',
                    'scale': 1.5,
                    'offset': (0, 1.85, 0),  # x, y, z offset for model positioning 
                    'bullet_origin': [(-1.1, 0.6, -0.1), (-0.8, 0.6, -1.2)],  # x, y, z offset from model center for bullet spawn , (-1.0, 0.6, 0.2)]
                    'camera_height': 2.5,
                    'shoot_sound': 'punisher_shoot.mp3',
                    'hit_sound': 'turret-hit.mp3',
                    'Yaw_offset': -90
                }
            ],
            'environment': [
                {
                    'name': 'Brick wall',
                    'type': 'structure',
                    'thumbnail': os.path.join(self.thumbnails_dir, 'cube_125pc.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [0.24, 0.24, 0.6],
                        'grid_count': (3, 10, 5),
                        'mass': 3.5,
                        'base_color': (0.8, 0.8, 0.8, 1.0),  
                        'tint_strength': 0.1,  
                        'tint_variation': (0.8, 0.8, 0.8),  
                        'fill': 'small_bricks',
                        'hit_sound': ['brick_hit.mp3'],
                        'stiff': False
                    }
                },
                {
                    'name': 'Rusted Tin boxes',
                    'type': 'structure',
                    'thumbnail': os.path.join(self.thumbnails_dir, 'cube_125pc.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [0.15, 0.45, 0.15],
                        'grid_count': (1, 7, 8),
                        'mass': 1.0,
                        'base_color': (0.8, 0.8, 0.8, 1.0),
                        'tint_strength': 0.1,
                        'tint_variation': (1.0, 1.0, 1.0),
                        'fill': 'rust_metal',
                        'hit_sound': ['metal_hit1.mp3', 'metal_hit2.mp3', 'metal_hit3.mp3','metal_hit4.mp3'],
                        'stiff': False                      
                    }
                },
                {
                    'name': 'concrete wall',
                    'type': 'structure',
                    'thumbnail': os.path.join(self.thumbnails_dir, 'cube_125pc.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [2.15, 3, 0.7],
                        'grid_count': (1, 1, 1),
                        'mass': 10.0,
                        'base_color': (1.0, 1.0, 1.0, 1.0),
                        'tint_strength': 0.7,
                        'tint_variation': (1.0, 1.0, 1.0),
                        'fill': 'concrete',
                        'hit_sound': ['concrete-hit.mp3'],  
                        'stiff': True
                    }
                },
                {
                    'name': 'concrete step',
                    'type': 'structure',
                    'thumbnail': os.path.join(self.thumbnails_dir, 'cube_125pc.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [2.15, 0.7, 0.7],
                        'grid_count': (1, 1, 1),
                        'mass': 10.0,
                        'base_color': (1.0, 1.0, 1.0, 1.0),
                        'tint_strength': 0.7,
                        'tint_variation': (1.0, 1.0, 1.0),
                        'fill': 'concrete',
                        'hit_sound': ['concrete-hit.mp3'],  
                        'stiff': True
                    }
                }
            
            ],
            'complex': [
                    {'name': 'Sports Car Mk1',
                    'type': 'structure',
                    'generator': 'compound',
                    'thumbnail': 'thumbs/sportscar.png',
                    'base': {
                        'shape': 'box',
                        'size': [1.0, 0.45, 0.18],
                        'mass': 600.0,
                        'position': [0, 0, 0.6],
                        'orientation_euler': [0, 0, 0],
                        'texture': 'chassis.jpg',
                        'tint': [0.9, 0.9, 0.95, 1.0],
                        'friction': 0.9,
                        'restitution': 0.05,
                        'linearDamping': 0.02,
                        'angularDamping': 0.02,
                        'breakable': false
                    },
                            "children": [
                                { "name": "wheel_fr", "shape": "cylinder", "size": [0.28, 0.18], "mass": 15.0,
                                    "relative_position": [0.75, 0.55, -0.08],
                                    "relative_orientation_euler": [0, 0, 0],
                                    "joint": { "type": "revolute", "axis": [0, 1, 0], "lower": -1000, "upper": 1000 },
                                    "texture": "wheel.jpg", "tint": [0.2, 0.2, 0.2, 1.0],
                                    "friction": 1.2, "restitution": 0.1,
                                    "breakable": true, "break": { "impulseThreshold": 140.0, "cumulativeDamageLimit": 280.0, "onBreak": "detach" }
                                },
                                { "name": "wheel_fl", "shape": "cylinder", "size": [0.28, 0.18], "mass": 15.0,
                                    "relative_position": [0.75, -0.55, -0.08],
                                    "relative_orientation_euler": [0, 0, 0],
                                    "joint": { "type": "revolute", "axis": [0, 1, 0], "lower": -1000, "upper": 1000 },
                                    "texture": "wheel.jpg", "tint": [0.2, 0.2, 0.2, 1.0],
                                    "friction": 1.2, "restitution": 0.1,
                                    "breakable": true, "break": { "impulseThreshold": 140.0, "cumulativeDamageLimit": 280.0, "onBreak": "detach" }
                                },
                                { "name": "wheel_rr", "shape": "cylinder", "size": [0.28, 0.18], "mass": 15.0,
                                     "relative_position": [-0.75, 0.55, -0.08],
                                    "relative_orientation_euler": [0, 0, 0],
                                    "joint": { "type": "revolute", "axis": [0, 1, 0], "lower": -1000, "upper": 1000 },
                                    "texture": "wheel.jpg", "tint": [0.2, 0.2, 0.2, 1.0],
                                    "friction": 1.2, "restitution": 0.1,
                                    "breakable": true, "break": { "impulseThreshold": 140.0, "cumulativeDamageLimit": 280.0, "onBreak": "detach" }
                                },
                                { "name": "wheel_rl", "shape": "cylinder", "size": [0.28, 0.18], "mass": 15.0,
                                    "relative_position": [-0.75, -0.55, -0.08],
                                    "relative_orientation_euler": [0, 0, 0],
                                    "joint": { "type": "revolute", "axis": [0, 1, 0], "lower": -1000, "upper": 1000 },
                                    "texture": "wheel.jpg", "tint": [0.2, 0.2, 0.2, 1.0],
                                    "friction": 1.2, "restitution": 0.1,
                                    "breakable": true, "break": { "impulseThreshold": 140.0, "cumulativeDamageLimit": 280.0, "onBreak": "detach" }
                                },
                                {   "name": "window_top", "shape": "box", "size": [0.6, 0.40, 0.12], "mass": 2.0,
                                    "relative_position": [0.0, 0.0, 0.35],
                                    "relative_orientation_euler": [0, 0, 0],
                                    "joint": { "type": "fixed", "axis": [0, 0, 0] },
                                    "texture": "window.jpg",
                                    "tint": [0.6, 0.75, 0.95, 0.85],
                                    "friction": 0.3, "restitution": 0.01,
                                    "breakable": true,
                                    "break": {
                                        "impulseThreshold": 45.0,
                                        "cumulativeDamageLimit": 60.0,
                                        "onBreak": "shatter",
                                        "shards": [
                                            { "shape": "box", "size": [0.08, 0.02, 0.04], "mass": 0.05, "texture": "glass_shard.jpg", "tint": [0.7,0.9,1.0,0.6], "localOffset": [ 0.05, 0.02, 0.02], "randomImpulse": 3.2 },
                                            { "shape": "box", "size": [0.06, 0.02, 0.03], "mass": 0.05, "texture": "glass_shard.jpg", "tint": [0.7,0.9,1.0,0.6], "localOffset": [-0.03,-0.04, 0.01], "randomImpulse": 3.0 },
                                            { "shape": "box", "size": [0.07, 0.02, 0.03], "mass": 0.05, "texture": "glass_shard.jpg", "tint": [0.7,0.9,1.0,0.6], "localOffset": [ 0.00, 0.05,-0.01], "randomImpulse": 3.1 },
                                            { "shape": "box", "size": [0.05, 0.02, 0.02], "mass": 0.05, "texture": "glass_shard.jpg", "tint": [0.7,0.9,1.0,0.6], "localOffset": [-0.06, 0.01, 0.00], "randomImpulse": 3.5 },
                                            { "shape": "box", "size": [0.06, 0.02, 0.02], "mass": 0.05, "texture": "glass_shard.jpg", "tint": [0.7,0.9,1.0,0.6], "localOffset": [ 0.04,-0.02,-0.02], "randomImpulse": 3.3 }
                                        ]
                                    }
                                },
                                { "name": "hood", "shape": "box", "size": [0.6, 0.45, 0.08], "mass": 5.0,
                                    "relative_position": [0.55, 0, 0.22],
                                    "relative_orientation_euler": [0, 0, 0],
                                    "joint": { "type": "fixed", "axis": [0, 0, 0] },
                                    "texture": "hood.jpg",
                                    "tint": [0.95, 0.1, 0.1, 1.0],
                                    "friction": 0.5, "restitution": 0.02,
                                    "breakable": true, "break": { "impulseThreshold": 110.0, "cumulativeDamageLimit": 200.0, "onBreak": "detach" }
                                },
                                { "name": "spoiler", "shape": "box", "size": [0.35, 0.15, 0.05], "mass": 1.5,
                                    "relative_position": [-0.9, 0, 0.28],
                                    "relative_orientation_euler": [0, 0, 0],
                                    "joint": { "type": "fixed", "axis": [0, 0, 0] },
                                    "texture": "spoiler.jpg",
                                    "tint": [0.2, 0.2, 0.2, 1.0],
                                    "friction": 0.6, "restitution": 0.03,
                                    "breakable": true, "break": { "impulseThreshold": 80.0, "cumulativeDamageLimit": 120.0, "onBreak": "detach" }
                                }
                            ],
                            "effects": {
                                "hitSounds": ["metal_hit.mp3"],
                                "breakSounds": ["metal_break.mp3"],
                                "debris": { "enabled": true, "count": 8, "size": [0.06,0.06,0.06], "massEach": 0.2, "texture": "debris.jpg", "tint": [0.6,0.6,0.6,1.0] }
                            }
                }
            ]
        }
    
    def get_player_models(self):
        """Return a list of available player models"""
        return self.available_models['player']
    
    def get_environment_models(self):
        """Return a list of available environment models"""
        return self.available_models['environment']
    
    def get_model_by_name(self, name, model_type='player'):
        """Get model info by name and type"""
        for model in self.available_models.get(model_type, []):
            if model['name'] == name:
                return model
        return None
    
    def load_model(self, name, model_type='player'):
        """Load a model by name and type"""
        model_info = self.get_model_by_name(name, model_type)
        if not model_info:
            raise ValueError(f"Model {name} of type {model_type} not found")
        
        # Load the OBJ model
        model = OBJ(model_info['path'])
        model.generate()
        
        # Store model metadata
        model.metadata = {
            'name': model_info['name'],
            'type': model_info['type'],
            'scale': model_info['scale'],
            'offset': model_info['offset'],
            'bullet_origin': model_info.get('bullet_origin', (0, 0, 1.0)),  # Default forward if not specified
            'camera_height': model_info.get('camera_height', 1.0),  # Default camera height if not specified
            'shoot_sound': model_info.get('shoot_sound', None),
            'hit_sound': model_info.get('hit_sound', None),
            'Yaw_offset': model_info.get('Yaw_offset', 0)
        }
        
        return model
