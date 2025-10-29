from typing import Optional, Dict, Any, List, Union
import os
import time
import json
import asyncio
import google.generativeai as genai
import pygame

class AIDescriptionHandler:
    """
    Handles AI-powered structure generation from text descriptions using Google's Gemini API.
    """
    
    def __init__(self, saves_dir: str, gemini_api_key: str = None):
        """
        Initialize the AI Description Handler.
        
        Args:
            saves_dir: Directory to save generated configurations
            gemini_api_key: Optional Google Gemini API key. If not provided, will try to load from environment.
        """
      
        # Store the API key as a class attribute
        self._gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self._model_initialized = False
        self.is_generating = False
        self.last_error = None
        self.last_response = None
        self.generated_json = None
        self.show_ai_panel = True
        
        # AI save files management - use ai_structures in the same directory as this file
        self.saves_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_structures')
        os.makedirs(self.saves_dir, exist_ok=True)
        
        # UI elements
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 14)
        self.text_color = (255, 255, 255)
        self.bg_color = (40, 44, 52, 230)
        self.input_color = (60, 64, 72)
        self.button_color = (70, 130, 180)
        self.button_hover_color = (100, 150, 200)
        self.slot_color = (80, 80, 90)
        self.slot_hover_color = (100, 100, 120)
        self.slot_active_color = (120, 140, 200)
        self.button_pressed = None
        
        # Panel dimensions
        self.panel_width = 440
        self.panel_height = 600
        
        # Input field rects
        self.input_rect = pygame.Rect(20, 100, 400, 120)
        self.generate_btn_rect = pygame.Rect(20, 240, 120, 30)
        self.save_btn_rect = pygame.Rect(160, 240, 120, 30)
        self.load_btn_rect = pygame.Rect(300, 240, 120, 30)
        self.close_btn_rect = pygame.Rect(380, 20, 40, 30)
        
        # Create 20 slot buttons in a 5x4 grid
        self.slot_rects = []
        slot_size = 60
        slot_margin = 10
        start_x = 20
        start_y = 300
        
        for i in range(20):
            row = i // 5
            col = i % 5
            x = start_x + col * (slot_size + slot_margin)
            y = start_y + row * (30 + slot_margin)
            self.slot_rects.append(pygame.Rect(x, y, slot_size, 30))
            
        self.save_slot = 0
        self.max_save_slots = 20
        self.description_input = ""
        
        # Initialize model if API key is available
        if self._gemini_api_key:
            self._initialize_model()

        # Available textures for structures    
        self.available_textures = ["small_bricks", "rust_metal", "concrete", "stone", "glass", "tough_glass", "wood"]
    def _get_ai_save_files(self) -> List[str]:
        """
        Get list of AI save files, ensuring all 20 slots exist.
        
        Returns:
            List of 20 filenames (ai_save_01.json to ai_save_20.json)
        """
        # Ensure saves directory exists
        os.makedirs(self.saves_dir, exist_ok=True)
        
        # Generate all 20 filenames
        filenames = [f'ai_save_{i:02d}.json' for i in range(1, 21)]
        
        # Create any missing files
        for filename in filenames:
            filepath = os.path.join(self.saves_dir, filename)
            if not os.path.exists(filepath):
                try:
                    with open(filepath, 'w') as f:
                        f.write('')
                except IOError as e:
                    print(f"[AI] Error creating save file {filename}: {e}")
        
        return filenames
    
    def save_exists(self, filename: str) -> bool:
        """
        Check if a save file exists
        
        Args:
            filename: Name of the file to check (e.g., 'ai_save_01.json')
            
        Returns:
            bool: True if the file exists and is not empty, False otherwise
        """
        filepath = os.path.join(self.saves_dir, filename)
        return os.path.exists(filepath) and os.path.getsize(filepath) > 0
        
    def save_ai_structure(self, slot: Optional[int] = None) -> Optional[str]:
        """
        Save the current structure to a file
        
        Args:
            slot: If provided, save to this specific slot (0-19).
                  If None, save to the currently selected slot.
                  
        Returns:
            Path to the saved file if successful, None otherwise
        """
             
        if not self.generated_json:
            return None
            
        try:
            # If no slot provided, use the currently selected one
            if slot is None:
                slot = self.save_slot
            
            # Ensure slot is within bounds
            slot = max(0, min(19, slot))
            
            # Ensure the saves directory exists
            saves_dir = os.path.abspath(self.saves_dir)
            os.makedirs(saves_dir, exist_ok=True)
            
            # Generate filename and path
            filename = f'ai_save_{slot+1:02d}.json'
            filepath = os.path.join(saves_dir, filename)
            
            # Ensure we have a string to write
            if isinstance(self.generated_json, dict):
                json_str = json.dumps(self.generated_json, indent=2)
            else:
                json_str = str(self.generated_json)
            
            # Validate JSON before saving
            try:
                json_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                return None
            
            # Save to file
            try:
                with open(filepath, 'w') as f:
                    f.write(json_str)
                
                file_size = os.path.getsize(filepath)
                return filepath
                
            except IOError as e:
                return None
            
        except Exception as e:
            return None
    
    def load_ai_structure(self, filename: str) -> Optional[Dict]:
        """
        Load a structure from a file
        
        Args:
            filename: Name of the file to load (e.g., 'ai_save_01.json')
            
        Returns:
            Parsed JSON data if successful, None otherwise
        """
        try:
            filepath = os.path.join(self.saves_dir, filename)
            
            # Check if file exists
            if not os.path.exists(filepath):
                return None
            
            # Check file size
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                return None
            
            # Read file content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                if not content:
                    return None
                
                # Store the raw content for debugging
                self.generated_json = content
                
                try:
                    # Parse JSON
                    data = json.loads(content)
                    
                    # Validate the structure
                    if not isinstance(data, dict):
                        return None
                        
                    # Check for required fields
                    if 'structures' not in data and 'bodies' not in data:
                        data = {'bodies': [data]}
                    
                    return data
                    
                except json.JSONDecodeError as e:
                    return None
                    
        except Exception as e:
            return None
            
    @property
    def gemini_api_key(self) -> str:
        """Get the current API key."""
        return self._gemini_api_key
        
    @gemini_api_key.setter
    def gemini_api_key(self, value: str):
        """Set the API key and reinitialize the model if it changes."""
        if value != self._gemini_api_key:
            self._gemini_api_key = value
            if value:  # Only reinitialize if we have a new key
                self._model_initialized = False
                self._initialize_model()
        
        # Create saves directory if it doesn't exist
        os.makedirs(saves_dir, exist_ok=True)
        
        # UI state
        self.show_ai_panel = False
        self.description_input = ""
        self.generated_json = ""
        self.save_slot = 0
        self.max_save_slots = 10
        
        # UI elements
        self.font = pygame.font.SysFont('Arial', 20)
        self.small_font = pygame.font.SysFont('Arial', 16)
        self.text_color = (255, 255, 255)
        self.bg_color = (40, 44, 52, 230)
        self.input_color = (60, 64, 72)
        self.button_color = (70, 130, 180)
        self.button_hover_color = (100, 150, 200)
        
        # Input field rects
        self.input_rect = pygame.Rect(20, 100, 400, 120)
        self.generate_btn_rect = pygame.Rect(20, 240, 120, 30)
        self.load_btn_rect = pygame.Rect(300, 240, 120, 30)
        self.close_btn_rect = pygame.Rect(380, 20, 40, 30)
        self.slot_rects = [pygame.Rect(20 + i * 40, 290, 35, 35) for i in range(self.max_save_slots)]
        
    @property
    def gemini_api_key(self) -> Optional[str]:
        """Get the current API key"""
        return self._gemini_api_key
        
    @gemini_api_key.setter
    def gemini_api_key(self, api_key: str) -> None:
        """Set the API key and reset model initialization state"""
        self._gemini_api_key = api_key
        self._model_initialized = False
        self.model = None

    def set_api_key(self, api_key: str) -> bool:
        """
        Set the Gemini API key and initialize the model
        
        Args:
            api_key: The Gemini API key to use for authentication
            
        Returns:
            bool: True if the API key was set and the model was initialized successfully, False otherwise
        """

        # Basic validation
        if not api_key or len(api_key) < 30:
            error_msg = "API key is too short or empty"
            self.last_error = error_msg
            self.gemini_api_key = None  # Clear any invalid key
            return False
            
        # Store the API key using the property setter
        self.gemini_api_key = api_key

        try:
            genai.configure(api_key=api_key)
            
            try:
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                
                response = self.model.generate_content("Say 'verification successful'")
                
                if response and hasattr(response, 'text') and response.text:
                    self._model_initialized = True
                    return True
                    
                self._model_initialized = False
                return False
                
            except Exception as model_error:
                error_msg = f"Failed to initialize Gemini model: {str(model_error)}"
                self.last_error = error_msg
                self.model = None
                return False
                
        except Exception as e:
            error_msg = str(e)
            
            # Add more specific error handling for Gemini
            if "API_KEY" in error_msg and "invalid" in error_msg.lower():
                self.last_error = "The provided Gemini API key is invalid"
            elif "quota" in error_msg.lower():
                self.last_error = "API quota exceeded - check your Google Cloud account"
            elif "permission" in error_msg.lower():
                self.last_error = "Permission denied - check your API key permissions"
            else:
                self.last_error = f"API error: {error_msg}"
                
            self.model = None
            return False

    async def _test_connection(self):
        """Test the Gemini connection with a simple request"""
        if not hasattr(self, 'client') or not self.client:
            return False
        try:
            start_time = time.time()
            
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say 'connection test successful'"}],
                max_tokens=10
            )
            
            elapsed = (time.time() - start_time) * 1000
            
            if response and hasattr(response, 'choices') and response.choices:
                result = response.choices[0].message.content
                return True
                
            return False
            
        except Exception as e:
            error_msg = str(e)
            
            # Add more specific error handling
            if "Incorrect API key" in error_msg:
                self.last_error = "The provided API key is invalid"
            elif "Rate limit" in error_msg:
                print("[AI] Error: Rate limit exceeded")
            elif "authentication" in error_msg.lower():
                print("[AI] Error: Authentication failed - please check your API key")
                
            self.last_error = f"Connection test failed: {error_msg}"
            return False
    
    def _initialize_model(self) -> bool:
        """Initialize the Gemini model with the current API key"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self._gemini_api_key)
            
            # Try to initialize the model
            try:
                self.model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as flash_error:
                self.model = genai.GenerativeModel('gemini-pro')
            
            response = self.model.generate_content("Test connection")
            
            if response and hasattr(response, 'text') and response.text:
                self._model_initialized = True
                self.last_error = None
                return True
                
            self.last_error = "Empty or invalid response from Gemini API"
            return False
            
        except Exception as e:
            error_msg = str(e) 
            self.last_error = f"Failed to initialize model: {error_msg}"
            self._model_initialized = False
            return False
    
    def _ensure_model_initialized(self) -> bool:
        """Ensure the model is properly initialized with a valid API key"""
        # If already initialized and we have a model, return True
        if self._model_initialized and hasattr(self, 'model') and self.model is not None:
            return True
            
        
        
        # Check if we have an API key
        if not self._gemini_api_key:
            error_msg = "Error: No API key available. Please set the API key first."
            self.last_error = error_msg
            return False
        
        # If we have a model but it's not marked as initialized, try using it
        if hasattr(self, 'model') and self.model is not None:
            self._model_initialized = True
            return True
            
        # If we have an API key but no model, try to initialize it
        if self._gemini_api_key:
            return self._initialize_model()
            
        return False
        
    async def generate_from_description(self, description: str) -> Optional[Dict[str, Any]]:
        """
        Generate a structure configuration from a text description using Google's Gemini.
        
        Args:
            description: Text description of the structure
            
        Returns:
            Dictionary containing the structure configuration, or None if generation failed
        """
        
        # Ensure the model is properly initialized
        if not self._ensure_model_initialized():
            return None
            
        self.is_generating = True
        self.last_error = None
        
        try:
            print(f"[AI] Generating structure for description: {description}")
            
            system_prompt = """
            You are a 3D structure generator that creates configurations in a specific JSON format.
            Generate a structure based on the user's description following these exact specifications:
            
            REQUIRED FIELDS (MUST include all marked with (M)):
            1. version: (M) Schema version number (always 1)
            2. timestamp: (M) Current Unix timestamp in seconds with decimals
            3. structures[]: (M) Array of structure objects, each with:
               - position: (M) [x, y, z] in meters
               - orientation: (M) Quaternion [x, y, z, w] (normalized)
               - size: (M) [width, height, depth] in meters
               - mass: (M) Mass in kg (typical range: 0.1-10000)
               - color: (M) RGBA values [r, g, b, a] (0.0-1.0)
               - metadata: (M) Object containing:
                 - stiff: (M) boolean (true for immovable objects)

            OPTIONAL FIELDS (include if applicable):
            - fill: Optional texture type (use one of: small_bricks, rust_metal, concrete, stone, glass, tough_glass, wood)
            - metadata.shard_config: Configuration for breakable objects
              - count: Number of shards to create
              - size_scale: Size multiplier for shards (0.1-1.0)
              - mass: Mass of each shard
              - velocity_scale: Velocity multiplier for shard ejection
              - impulse_threshold: Minimum force needed to break

            RULES:
            1. Use realistic dimensions (e.g., walls: 0.2-0.5m thick, floors: 0.1-0.3m)
            2. Keep structures within reasonable bounds (e.g., ±50m from origin)
            3. Use quaternions for orientation (default [0,0,0,1] for no rotation)
            4. Set mass based on volume and material density
            5. For 'stiff: true' objects, ensure they're properly anchored
            6. Use consistent units (meters, kg, seconds)
            7. For the 'fill' property, use one of these exact values if applicable: small_bricks, rust_metal, concrete, stone, glass, tough_glass, wood

            Example materials and properties:
            - Concrete: density ~2400 kg/m³, color [0.7, 0.7, 0.7, 1.0]
            - Glass: density ~2500 kg/m³, color [0.9, 0.9, 1.0, 0.3]
            - Wood: density ~700 kg/m³, color [0.5, 0.3, 0.1, 1.0]
            - Metal: density ~7800 kg/m³, color [0.8, 0.8, 0.9, 1.0]
            
            Respond ONLY with valid JSON, no additional text or markdown formatting.
            """
            
            # Use a raw string for the JSON template to avoid f-string nesting issues
            json_template = r'''
            {
              "position": [0, 1, 0],
              "orientation": [0, 0, 0, 1],
              "size": [1, 1, 1],
              "mass": 1.0,
              "color": [0.8, 0.2, 0.2, 1.0],
              "metadata": {
                "stiff": false,
                "shard_config": {
                  "count": 16.0,
                  "size_scale": 0.35,
                  "mass": 0.5,
                  "velocity_scale": 0.5,
                  "impulse_threshold": 500.0
                }
              }
            }
            '''
            
            user_prompt = f"""
            Generate a 3D structure based on this description: "{description}"
            
            Requirements:
            1. Create a structure that matches the description
            2. Use appropriate materials and dimensions
            3. Include all required fields for each structure
            4. Set realistic physics properties
            5. Keep the structure within reasonable bounds
            6. Use proper quaternion rotations
            7. For the 'fill' property, use one of these exact values if applicable: small_bricks, rust_metal, concrete, stone, glass, tough_glass, wood
            8. Return ONLY valid JSON, no additional text or markdown
            
            Example of a single structure (your response should include multiple structures):
            {json_template}
            """    
            try:
                # Combine system prompt and user prompt for Gemini
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                # Ensure we have a valid model instance
                if not hasattr(self, 'model') or self.model is None:
                    error_msg = "Model not initialized. Please check your API key and try again."
                    print(f"[AI] {error_msg}")
                    self.last_error = error_msg
                    return None
                
                # Make the API call to Gemini using the pre-initialized model
                try:
                    # Use the instance's model instead of creating a new one
                    response = await self.model.generate_content_async(full_prompt)
                except Exception as e:
                    error_msg = f"Error calling Gemini API: {str(e)}"
                    print(f"[AI] {error_msg}")
                    print(f"[AI] Model type: {type(self.model).__name__}")
                    print(f"[AI] API Key: {'Set' if self._gemini_api_key else 'Not set'}")
                    self.last_error = error_msg
                    return None
                
                if not response or not hasattr(response, 'text'):
                    raise Exception("Invalid response from Gemini API")
                
                if not response.text:
                    error_msg = "Empty response from Gemini API"
                    if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
                        error_msg += f" (Block reason: {response.prompt_feedback.block_reason})"
                    print(f"[AI] {error_msg}")
                    self.last_error = error_msg
                    return None
                
                generated_text = response.text
                print(f"[AI] Generated text: {generated_text[:5000]}...")  # Log first 5000 chars
                
                # Clean the response to extract just the JSON part
                if '```json' in generated_text:
                    generated_text = generated_text.split('```json')[1].split('```')[0].strip()
                elif '```' in generated_text:
                    generated_text = generated_text.split('```')[1].split('```')[0].strip()
                else:
                    # If no code blocks, try to extract JSON
                    json_start = generated_text.find('{')
                    json_end = generated_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        generated_text = generated_text[json_start:json_end]
                
                self.generated_json = generated_text
                
                # Try to parse the JSON response
                try:
                    structure = json.loads(generated_text)
                    print("[AI] Successfully parsed JSON structure")
                    
                    if not isinstance(structure, dict):
                        error_msg = "Generated structure is not a dictionary"
                        print(f"[AI] {error_msg}")
                        self.last_error = error_msg
                        return None
                        
                    if 'structures' not in structure:
                        error_msg = "Generated structure is missing 'structures' key"
                        print(f"[AI] {error_msg}")
                        self.last_error = error_msg
                        return None
                    
                    self.last_response = structure
                    return structure
                    
                except json.JSONDecodeError as e:
                    error_msg = f"Failed to parse generated JSON: {str(e)}"
                    print(f"[AI] {error_msg}")
                    print(f"[AI] Generated text was: {generated_text}")
                    self.last_error = error_msg
                    return None
                    
            except Exception as api_error:
                error_msg = f"Error calling Gemini API: {str(api_error)}"
                print(f"[AI] {error_msg}")
                import traceback
                traceback.print_exc()
                self.last_error = error_msg
                return None
                
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"[AI] {error_msg}")
            import traceback
            traceback.print_exc()
            self.last_error = error_msg
            return None
            
        finally:
            self.is_generating = False
    
    def save_to_slot(self, slot: int) -> bool:
        """Save the last generated configuration to a slot"""
        if not hasattr(self, 'last_response') or not self.last_response:
            self.last_error = "No configuration to save"
            return False
            
        try:
            save_path = os.path.join(self.saves_dir, f"ai_slot_{slot}.json")
            with open(save_path, 'w') as f:
                json.dump(self.last_response, f, indent=2)
            return True
        except Exception as e:
            self.last_error = str(e)
            return False
    
    def load_from_slot(self, slot: int) -> Optional[Dict[str, Any]]:
        """Load a configuration from a slot"""
        try:
            save_path = os.path.join(self.saves_dir, f"ai_slot_{slot}.json")
            if not os.path.exists(save_path):
                self.last_error = f"No save found in slot {slot}"
                return None
                
            with open(save_path, 'r') as f:
                config = json.load(f)
                self.last_response = config
                self.generated_json = json.dumps(config, indent=2)
                return config
        except Exception as e:
            self.last_error = str(e)
            return None
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame events. Returns True if the event was handled."""
        if not self.show_ai_panel:
            return False
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.show_ai_panel = False
                return True
                
            # Handle text input
            if event.key == pygame.K_BACKSPACE:
                self.description_input = self.description_input[:-1]
            elif event.key == pygame.K_RETURN:
                # Start generation on Enter
                if self.description_input.strip():
                    asyncio.create_task(self.generate_from_description(self.description_input))
                # Add the typed character to the input
                if event.unicode.isprintable():
                    self.description_input += event.unicode
                    
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_pos = pygame.mouse.get_pos()
                
                # Check close button
                if self.close_btn_rect.collidepoint(mouse_pos):
                    self.show_ai_panel = False
                    return True
                    
                # Check generate button
                if self.generate_btn_rect.collidepoint(mouse_pos) and self.description_input.strip():
                    asyncio.create_task(self.generate_from_description(self.description_input))
                    return True
                    
                # Check save button
                if self.save_btn_rect.collidepoint(mouse_pos) and self.generated_json:
                    # Save to the currently selected slot
                    saved_file = self.save_ai_structure(self.save_slot)
                    if saved_file:
                        print(f"[AI] Structure saved to slot {self.save_slot + 1}")
                    return True
                    
                # Check load button
                if self.load_btn_rect.collidepoint(mouse_pos):
                    # Load from the currently selected slot
                    filename = f'ai_save_{self.save_slot + 1:02d}.json'
                    structure = self.load_ai_structure(filename)
                    if structure:
                        print(f"[AI] Loaded structure from slot {self.save_slot + 1}")
                        # Here you would typically return the structure to be spawned
                        # For example: return {'type': 'ai_structure', 'data': structure}
                    return True
                
                # Check if a slot was clicked
                for i, rect in enumerate(self.slot_rects):
                    if rect.collidepoint(mouse_pos):
                        self.save_slot = i
                        return True
                        
        return False
    
    def draw(self, surface: pygame.Surface):
        """Draw the AI description panel"""
        if not self.show_ai_panel:
            return None
            
        mouse_pos = pygame.mouse.get_pos()
        
        # Draw semi-transparent background
        s = pygame.Surface((self.panel_width, self.panel_height), pygame.SRCALPHA)
        s.fill(self.bg_color)
        surface.blit(s, (10, 10))
        
        # Draw title
        title = self.font.render("AI Structure Generator", True, self.text_color)
        surface.blit(title, (30, 30))
        
        # Draw close button
        close_btn_color = (220, 70, 70) if self.close_btn_rect.collidepoint(mouse_pos) else (200, 50, 50)
        pygame.draw.rect(surface, close_btn_color, self.close_btn_rect, 0, 5)
        close_text = self.small_font.render("X", True, (255, 255, 255))
        surface.blit(close_text, (self.close_btn_rect.x + 15, self.close_btn_rect.y + 5))
        
        # Draw description input
        pygame.draw.rect(surface, self.input_color, self.input_rect, 0, 5)
        pygame.draw.rect(surface, (100, 100, 100), self.input_rect, 2, 5)
        
        # Draw input text
        text_surface = self.small_font.render(self.description_input, True, self.text_color)
        # Render cursor if input is focused
        cursor_pos = len(self.description_input)
        cursor_x = self.input_rect.x + 10 + self.small_font.size(self.description_input[:cursor_pos])[0]
        pygame.draw.line(surface, self.text_color, (cursor_x, self.input_rect.y + 10), 
                        (cursor_x, self.input_rect.y + self.input_rect.height - 10), 2)
        
        # Draw buttons
        self._draw_button(surface, "Generate", self.generate_btn_rect, 
                         hover_color=self.button_hover_color if self.generate_btn_rect.collidepoint(mouse_pos) else self.button_color)
        
        self._draw_button(surface, f"Save to {self.save_slot+1}", self.save_btn_rect,
                         hover_color=self.button_hover_color if self.save_btn_rect.collidepoint(mouse_pos) else self.button_color,
                         enabled=bool(self.generated_json))
        
        self._draw_button(surface, "Load", self.load_btn_rect,
                         hover_color=self.button_hover_color if self.load_btn_rect.collidepoint(mouse_pos) else self.button_color)
        
        # Draw slot buttons
        for i, rect in enumerate(self.slot_rects):
            # Check if slot has content
            filename = f'ai_save_{i+1:02d}.json'
            filepath = os.path.join(self.saves_dir, filename)
            has_content = os.path.exists(filepath) and os.path.getsize(filepath) > 0
            
            # Determine button color
            if i == self.save_slot:
                color = self.slot_active_color
            elif rect.collidepoint(mouse_pos):
                color = self.slot_hover_color
            else:
                color = self.slot_color
                
            # Draw button
            pygame.draw.rect(surface, color, rect, 0, 5)
            pygame.draw.rect(surface, (100, 100, 100), rect, 1, 5)
            
            # Draw slot number
            slot_text = self.small_font.render(str(i+1), True, self.text_color)
            text_rect = slot_text.get_rect(center=rect.center)
            surface.blit(slot_text, text_rect)
            
            # Draw indicator if slot has content
            if has_content:
                pygame.draw.circle(surface, (50, 200, 50), 
                                 (rect.right - 8, rect.top + 8), 4)
        
        # Draw help text
        help_text = self.small_font.render("Click a slot to select, then Save or Load", True, (200, 200, 200))
        surface.blit(help_text, (20, 270))
        
        # Draw status/error message
        if self.last_error:
            error_text = self.small_font.render(f"Error: {self.last_error}", True, (255, 100, 100))
            surface.blit(error_text, (20, self.panel_height - 30))
        elif self.is_generating:
            status_text = self.small_font.render("Generating...", True, (100, 200, 100))
            surface.blit(status_text, (20, self.panel_height - 30))
        
        # Draw description text
        y_offset = 5
        for line in self._wrap_text(self.description_input, self.small_font, self.input_rect.width - 20):
            text_surface = self.small_font.render(line, True, self.text_color)
            surface.blit(text_surface, (self.input_rect.x + 10, self.input_rect.y + y_offset))
            y_offset += 20
        
        # Draw buttons
        self._draw_button(surface, "Generate", self.generate_btn_rect)
        self._draw_button(surface, "Save", self.save_btn_rect)
        self._draw_button(surface, "Load", self.load_btn_rect)
        
        # Draw save/load buttons
        self._draw_button(surface, "Generate", self.generate_btn_rect)
        self._draw_button(surface, "Save", self.save_btn_rect)
        self._draw_button(surface, "Load", self.load_btn_rect)
        
        # Draw saved structures grid
        save_text = self.small_font.render("Saved Structures:", True, self.text_color)
        surface.blit(save_text, (20, 280))
        
        # Draw saved structure buttons in a grid
        for i, (rect, save_file) in enumerate(zip(self.slot_rects, self.ai_save_files)):
            mouse_pos = pygame.mouse.get_pos()
            is_hovered = rect.collidepoint(mouse_pos)
            
            # Draw button background
            button_color = self.button_hover_color if is_hovered else self.button_color
            pygame.draw.rect(surface, button_color, rect, 0, 3)
            pygame.draw.rect(surface, (100, 100, 100), rect, 1, 3)
            
            # Display shortened filename
            display_name = save_file[:-5]  # Remove .json
            if len(display_name) > 8:  # Truncate if too long
                display_name = display_name[:8] + '..'
                
            text_surf = self.small_font.render(display_name, True, self.text_color)
            surface.blit(text_surf, (rect.centerx - text_surf.get_width() // 2, 
                                   rect.centery - text_surf.get_height() // 2))
        
        # Draw generated JSON
        if self.generated_json:
            json_surface = pygame.Surface((400, 200), pygame.SRCALPHA)
            json_surface.fill((50, 54, 62, 200))
            
            y_offset = 10
            for line in self.generated_json.split('\n'):
                text_surface = self.small_font.render(line, True, self.text_color)
                json_surface.blit(text_surface, (10, y_offset))
                y_offset += 18
                
            surface.blit(json_surface, (20, 340))
        
        # Draw error message if any
        if self.last_error:
            error_text = self.small_font.render(f"Error: {self.last_error}", True, (255, 100, 100))
            surface.blit(error_text, (20, 550))
    
    def _draw_button(self, surface: pygame.Surface, text: str, rect: pygame.Rect):
        """Helper method to draw a button"""
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = rect.collidepoint(mouse_pos)
        color = self.button_hover_color if is_hovered else self.button_color
        
        pygame.draw.rect(surface, color, rect, 0, 5)
        pygame.draw.rect(surface, (100, 100, 100), rect, 2, 5)
        
        text_surface = self.small_font.render(text, True, self.text_color)
        text_rect = text_surface.get_rect(center=rect.center)
        surface.blit(text_surface, text_rect)
    
    @staticmethod
    def _wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list:
        """Wrap text to fit within a given width"""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if font.size(test_line)[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
            
        return lines
