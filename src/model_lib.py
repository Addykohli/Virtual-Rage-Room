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
                    'thumbnail': os.path.join(self.thumbnails_dir, 'sci_fi_aa_turret.png'),
                    'type': 'player',
                    'scale': 1.0,
                    'offset': (0, 0.2, 0)  # x, y, z offset (0.5 units up)
                }
            ],
            'environment': [
                # Will be populated with environment models
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
            'offset': model_info['offset']
        }
        
        return model
