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
                    'path': os.path.join(self.models_dir, 'turret','Sci Fi AA Turret.obj'),
                    'mesh_path': os.path.join(self.models_dir, 'turret','Sci Fi AA Turret.obj'),
                    'thumbnail': os.path.join(self.thumbnails_dir, 'sci_fi_aa_turret.png'),
                    'type': 'player',
                    'scale': 1.5,
                    'offset': (0, 1.25, 0),  # x, y, z offset for model positioning
                    'bullet_origin': [(0.0, -0.12, 0.0)],  # x, y, z offset from model center for bullet spawn
                    'camera_height': -4.5,
                    'shoot_sound': 'turret_shoot.mp3',
                    'hit_sound': 'turret_hit.mp3',
                    'Yaw_offset': 15
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
                },
                {
                    'name': 'Cannon',
                    'path': os.path.join(self.models_dir, 'Cannon','Cannon.obj'),
                    'mesh_path': os.path.join(self.models_dir, 'Cannon','Cannon.obj'),
                    'thumbnail': os.path.join(self.thumbnails_dir, 'Cannon.png'),
                    'type': 'player',
                    'scale': 1.3,
                    'offset': (0, 1.35, 0),  # x, y, z offset for model positioning
                    'bullet_origin': [(0.6, 0.4, -0.3)],  # x, y, z offset from model center for bullet spawn
                    'camera_height': -4.0,
                    'shoot_sound': 'cannon_shoot.mp3',
                    'hit_sound': 'turret-hit.mp3',
                    'Yaw_offset': 10
                }
            ],
            'environment': [
                {
                    'name': 'Brick wall',
                    'type': 'structure',
                    'thumbnail': os.path.join(self.thumbnails_dir, 'brick_wall.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [0.21, 0.21, 0.5],
                        'grid_count': (3, 10, 5),
                        'mass': 1.5,
                        'base_color': (0.8, 0.8, 0.8, 1.0),  
                        'tint_strength': 0.1,  
                        'tint_variation': (0.8, 0.8, 0.8),  
                        'fill': 'small_bricks',
                        'hit_sound': ['brick-hit.mp3'],
                        'stiff': False,
                        'shard_config': {
                            'count': 29,           # Number of shards to create
                            'size_scale': 0.38,    # Shard size relative to original brick
                            'mass': 0.6,          # Mass of each shard
                            'velocity_scale': 0.5, # How fast shards fly
                            'impulse_threshold': 600.0  # Minimum impulse required to break the brick
                        }
                    }
                },
                {
                    'name': 'Rusted Tin boxes',
                    'type': 'structure',
                    'thumbnail': os.path.join(self.thumbnails_dir, 'rusted_tin.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [0.2, 0.45, 0.2],
                        'grid_count': (1, 7, 8),
                        'mass': 1.0,
                        'base_color': (0.8, 0.8, 0.8, 1.0),
                        'tint_strength': 0.1,
                        'tint_variation': (1.0, 1.0, 1.0),
                        'fill': 'rust_metal',
                        'hit_sound': ['metal_hit1.mp3', 'metal_hit2.mp3', 'metal_hit3.mp3', 'metal_hit4.mp3'],
                        'stiff': False
                    }
                },
                {
                    'name': 'Concrete Wall',
                    'type': 'structure',
                    'thumbnail': os.path.join(self.thumbnails_dir, 'concrete_wall.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [8, 8, 0.3],
                        'grid_count': (1, 1, 1),
                        'mass': 10.0,
                        'base_color': (0.7, 0.7, 0.7, 1.0),
                        'tint_strength': 0.7,
                        'tint_variation': (0.1, 0.1, 0.1),
                        'fill': 'concrete',
                        'hit_sound': ['concrete-hit.mp3'],
                        'stiff': True
                    }
                },
                {
                    'name': 'stone wall',
                    'type': 'structure',
                    'thumbnail': os.path.join(self.thumbnails_dir, 'stone_wall.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [0.3, 0.1, 0.6],
                        'grid_count': (20, 25, 1),
                        'mass': 30.0,
                        'base_color': (1.0, 1.0, 1.0, 1.0),
                        'tint_strength': 0.7,
                        'tint_variation': (1.0, 1.0, 1.0),
                        'fill': 'stone',
                        'hit_sound': ['concrete-hit.mp3'],  
                        'stiff': True,
                        'shard_config': {
                            'count': 12,           # Number of shards to create
                            'size_scale': 0.45,    # Shard size relative to original brick
                            'mass': 3.6,          # Mass of each shard
                            'velocity_scale': 0.3, # How fast shards fly
                            'impulse_threshold': 1000.0  # Minimum impulse required to break the brick
                        }
                    }
                },
                {
                    'name': 'glass',
                    'type': 'structure',
                    'thumbnail': os.path.join(self.thumbnails_dir, 'glass.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [1.5, 1, 0.1],
                        'grid_count': (1, 1, 1),
                        'mass': 30.0,
                        'base_color': (0.13, 0.13, 0.54, 0.34),
                        'tint_strength': 0.2,
                        'tint_variation': (1.0, 1.0, 1.0),
                        'fill': 'glass',
                        'stiff': False,
                        'shard_config': {
                            'count': 18,           # Number of shards to create
                            'size_scale': 0.3,    # Shard size relative to original brick
                            'mass': 3.6,          # Mass of each shard
                            'velocity_scale': 0.3, # How fast shards fly
                            'shatter_sound': ['glass_break1.mp3', 'glass_break2.mp3'],  
                            'impulse_threshold': 50.0  # Minimum impulse required to break the brick
                        }
                    }
                },
                {
                    'name': 'tough_glass',
                    'type': 'structure',
                    'thumbnail': os.path.join(self.thumbnails_dir, 'glass.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [1.5, 1, 0.1],
                        'grid_count': (1, 1, 1),
                        'mass': 30.0,
                        'base_color': (0.13, 0.13, 0.54, 0.34),
                        'tint_strength': 0.2,
                        'tint_variation': (1.0, 1.0, 1.0),
                        'fill': 'tough_glass',
                        'stiff': True,
                        'shard_config': {
                            'count': 16,           # Number of shards to create
                            'size_scale': 0.35,    # Shard size relative to original brick
                            'mass': 3.6,          # Mass of each shard
                            'velocity_scale': 0.3, # How fast shards fly
                            'shatter_sound': ['glass_break1.mp3', 'glass_break2.mp3'],  
                            'impulse_threshold': 500.0  # Minimum impulse required to break the brick
                        }
                    }
                },
                {
                    'name': 'Fence',
                    'type': 'structure',
                    'path': os.path.join(self.models_dir, 'fence','Fence.obj'),
                    'mesh_path': os.path.join(self.models_dir, 'fence','Fence.obj'),
                    'thumbnail': os.path.join(self.thumbnails_dir, 'Fence.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [1, 5.8, 5.85],
                        'grid_count': (1, 1, 1),
                        'mass': 10.0,
                        'base_color': (1.0, 1.0, 1.0, 1.0),
                        'tint_strength': 0.7,
                        'tint_variation': (1.0, 1.0, 1.0),
                        'hit_sound': ['concrete-hit.mp3'],  
                        'stiff': True
                    }
                },
                {
                    'name': 'Glass building',
                    'type': 'structure',
                    'path': os.path.join(self.models_dir, 'building','building 5.obj'),
                    'mesh_path': os.path.join(self.models_dir, 'building','building 5.obj'),
                    'thumbnail': os.path.join(self.thumbnails_dir, 'building_structure.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [2.9, 8, 2.9],
                        'grid_count': (1, 1, 1),
                        'mass': 10.0,
                        'base_color': (1.0, 1.0, 1.0, 1.0),
                        'tint_strength': 0.7,
                        'tint_variation': (1.0, 1.0, 1.0),
                        'hit_sound': ['concrete-hit.mp3'],  
                        'stiff': True
                    }
                },
                {
                    'name': 'Modular building',
                    'type': 'structure',
                    'path': os.path.join(self.models_dir, 'modular_tower','building 5.obj'),
                    'mesh_path': os.path.join(self.models_dir, 'modular_tower','building 5.obj'),
                    'thumbnail': os.path.join(self.thumbnails_dir, 'Modular_building.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [2.5, 14, 2.5],
                        'grid_count': (1, 1, 1),
                        'mass': 10.0,
                        'base_color': (1.0, 1.0, 1.0, 1.0),
                        'tint_strength': 0.7,
                        'tint_variation': (1.0, 1.0, 1.0),
                        'hit_sound': ['concrete-hit.mp3'],  
                        'stiff': True
                    }
                },
                {
                    'name': 'Office building',
                    'type': 'structure',
                    'path': os.path.join(self.models_dir, 'office_building','Building 5.obj'),
                    'mesh_path': os.path.join(self.models_dir, 'office_building','Building 5.obj'),
                    'thumbnail': os.path.join(self.thumbnails_dir, 'office_building.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [9, 16, 9],
                        'grid_count': (1, 1, 1),
                        'mass': 10.0,
                        'base_color': (1.0, 1.0, 1.0, 1.0),
                        'tint_strength': 0.7,
                        'tint_variation': (1.0, 1.0, 1.0),
                        'hit_sound': ['concrete-hit.mp3'],  
                        'stiff': True
                    }
                },
                {
                    'name': 'Big building',
                    'type': 'structure',
                    'path': os.path.join(self.models_dir, 'big_building','building 5.obj'),
                    'mesh_path': os.path.join(self.models_dir, 'big_building','building 5.obj'),
                    'thumbnail': os.path.join(self.thumbnails_dir, 'building_structure.png'),
                    'generator': 'cluster',
                    'cluster': {
                        'size': [6, 24, 6],
                        'grid_count': (1, 1, 1),
                        'mass': 10.0,
                        'base_color': (1.0, 1.0, 1.0, 1.0),
                        'tint_strength': 0.7,
                        'tint_variation': (1.0, 1.0, 1.0),
                        'hit_sound': ['concrete-hit.mp3'],  
                        'stiff': True
                    }
                }
            ],
            
        }
    
    def get_player_models(self):
        """Return a list of available player models"""
        return self.available_models['player']
    
    def get_environment_models(self):
        """Return a list of available environment models with model_path in metadata"""
        models = []
        for model in self.available_models['environment']:
            # Create a deep copy to avoid modifying the original
            model = model.copy()
            if 'cluster' in model:
                # Make sure we have a copy of the cluster config
                model['cluster'] = model['cluster'].copy()
                
                # Ensure metadata exists in the cluster
                if 'metadata' not in model['cluster']:
                    model['cluster']['metadata'] = {}
                
                # Add model_path to the cluster metadata if it's not already there
                if 'path' in model and 'model_path' not in model['cluster']['metadata']:
                    model['cluster']['metadata']['model_path'] = model['path']
                
                # Also copy any model_path from the root to metadata if not present
                elif 'model_path' in model and 'model_path' not in model['cluster']['metadata']:
                    model['cluster']['metadata']['model_path'] = model['model_path']
            
            models.append(model)
        return models
    
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
            'thumbnail': model_info['thumbnail'],
            'scale': model_info['scale'],
            'offset': model_info['offset'],
            'bullet_origin': model_info.get('bullet_origin', (0, 0, 1.0)),  # Default forward if not specified
            'camera_height': model_info.get('camera_height', 1.0),  # Default camera height if not specified
            'shoot_sound': model_info.get('shoot_sound', None),
            'hit_sound': model_info.get('hit_sound', None),
            'Yaw_offset': model_info.get('Yaw_offset', 0)
        }
        
        return model
