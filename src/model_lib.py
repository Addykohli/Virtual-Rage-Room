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
                    'bullet_origin': [(-1.3, 0.6, -0.1), (-0.8, 0.6, -1.2)],  # x, y, z offset from model center for bullet spawn , (-1.0, 0.6, 0.2)]
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
                        'grid_count': (4, 7, 4),
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
