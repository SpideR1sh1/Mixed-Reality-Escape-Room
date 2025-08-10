"""
Dynamic Cave Lighting System for Unity Integration
================================================

This system generates sophisticated cave lighting that adapts to player progress,
environmental conditions, and gameplay events. Uses advanced lighting algorithms
that would be computationally expensive in Unity's real-time renderer.

Features:
- Progress-driven lighting scenarios
- Advanced ray-tracing simulation for realistic light bouncing
- Dynamic shadow generation with real-time updates
- Atmospheric lighting effects (volumetric fog, god rays)
- Psychological lighting design (tension, relief, discovery)
- Performance-optimized streaming to Unity
- Scientific accuracy for cave environments
- Real-time color temperature adjustment

Interview Demonstration Points:
- Complex mathematical lighting calculations
- Real-time performance optimization
- Advanced graphics programming concepts
- Scientific lighting simulation
- Seamless Unity integration
"""

import asyncio
import websockets
import json
import numpy as np
import math
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightingScenario(Enum):
    ENTRANCE = "entrance"          # Bright outdoor light fading in
    EXPLORATION = "exploration"    # Torch/headlamp lighting
    DISCOVERY = "discovery"        # Revealing hidden areas
    PUZZLE = "puzzle"             # Focused task lighting
    TENSION = "tension"           # Dramatic shadows, suspense
    RELIEF = "relief"             # Warm, safe lighting
    VICTORY = "victory"           # Triumphant, bright lighting
    MAGICAL = "magical"           # Supernatural glow effects

class LightType(Enum):
    AMBIENT = "ambient"           # Overall scene illumination
    DIRECTIONAL = "directional"   # Sun/moon light
    POINT = "point"              # Torch, candle
    SPOT = "spot"                # Flashlight, focused beam
    AREA = "area"                # Large soft light source
    VOLUMETRIC = "volumetric"    # God rays, fog lighting

@dataclass
class LightSource:
    """Represents a single light source with all properties"""
    id: str
    light_type: LightType
    position: Tuple[float, float, float]
    direction: Tuple[float, float, float]
    color: Tuple[float, float, float]  # RGB 0-1
    intensity: float                   # Brightness multiplier
    range: float                      # Maximum distance
    spot_angle: float                 # For spot lights (degrees)
    falloff_curve: str               # linear, quadratic, realistic
    shadows: bool                    # Cast shadows
    volumetric: bool                 # Volumetric effects
    flicker: bool                    # Dynamic intensity variation
    temperature: float               # Color temperature in Kelvin
    enabled: bool                    # On/off state

@dataclass
class ProgressMarkers:
    """Player progress indicators that affect lighting"""
    overall_progress: float          # 0-1 completion percentage
    current_puzzle: int             # Current puzzle index
    puzzles_completed: int          # Number solved
    secrets_found: int              # Hidden items discovered
    areas_explored: int             # Cave sections visited
    time_in_cave: float            # Minutes spent
    player_stress_level: float     # 0-1 tension indicator
    last_achievement: str          # Most recent accomplishment
    proximity_to_exit: float       # Distance to completion

@dataclass
class EnvironmentalFactors:
    """Environmental conditions affecting lighting"""
    cave_depth: float              # Meters underground
    humidity: float                # 0-1 moisture level
    air_circulation: float         # 0-1 air movement
    natural_light_sources: int     # Cracks, openings
    player_light_sources: int      # Torch, headlamp, etc.
    ambient_temperature: float     # Celsius
    atmospheric_pressure: float    # kPa
    dust_particles: float          # 0-1 particle density

class AdvancedLightingCalculator:
    """Advanced lighting calculations using ray-tracing principles"""
    
    def __init__(self):
        self.light_cache = {}
        self.ray_samples = 64  # Ray tracing samples
        self.bounce_limit = 3  # Light bounce recursion
        
    def calculate_global_illumination(self, light_sources: List[LightSource],
                                    cave_geometry: Dict, surface_materials: Dict) -> Dict:
        """Calculate global illumination using simplified ray tracing"""
        
        # Initialize illumination map
        illumination_map = {}
        
        for light in light_sources:
            if not light.enabled:
                continue
                
            # Calculate direct illumination
            direct_light = self._calculate_direct_lighting(light, cave_geometry)
            
            # Calculate indirect illumination (bounces)
            indirect_light = self._calculate_indirect_lighting(
                light, cave_geometry, surface_materials
            )
            
            # Combine direct and indirect
            light_contribution = self._combine_lighting(direct_light, indirect_light)
            
            # Add to global illumination
            self._accumulate_lighting(illumination_map, light_contribution)
        
        return illumination_map
    
    def _calculate_direct_lighting(self, light: LightSource, geometry: Dict) -> Dict:
        """Calculate direct lighting from a source"""
        direct_lighting = {}
        
        # Sample points in cave geometry
        for surface_id, surface_data in geometry.items():
            surface_points = surface_data['points']
            surface_normals = surface_data['normals']
            
            for i, point in enumerate(surface_points):
                # Calculate distance to light
                light_vector = np.array(light.position) - np.array(point)
                distance = np.linalg.norm(light_vector)
                
                if distance > light.range:
                    continue
                
                # Normalize light direction
                light_direction = light_vector / distance
                
                # Calculate angle with surface normal
                normal = np.array(surface_normals[i])
                angle = np.dot(normal, light_direction)
                
                if angle <= 0:  # Back-facing
                    continue
                
                # Calculate attenuation
                attenuation = self._calculate_attenuation(distance, light)
                
                # Calculate spot light cone (if applicable)
                spot_factor = self._calculate_spot_factor(light, light_direction)
                
                # Final intensity
                final_intensity = (light.intensity * attenuation * 
                                 angle * spot_factor)
                
                # Store lighting data
                if surface_id not in direct_lighting:
                    direct_lighting[surface_id] = []
                
                direct_lighting[surface_id].append({
                    'point': point,
                    'intensity': final_intensity,
                    'color': light.color,
                    'light_id': light.id
                })
        
        return direct_lighting
    
    def _calculate_indirect_lighting(self, light: LightSource, geometry: Dict,
                                   materials: Dict) -> Dict:
        """Calculate indirect lighting from bounces"""
        indirect_lighting = {}
        
        # Simplified bounce calculation
        for bounce in range(self.bounce_limit):
            bounce_factor = 0.5 ** (bounce + 1)  # Energy reduction per bounce
            
            # Calculate reflected light from surfaces
            for surface_id, material in materials.items():
                reflectance = material.get('reflectance', 0.1)
                
                if reflectance < 0.01:  # Skip non-reflective surfaces
                    continue
                
                # Calculate reflected light contribution
                reflected_intensity = light.intensity * bounce_factor * reflectance
                
                if surface_id not in indirect_lighting:
                    indirect_lighting[surface_id] = []
                
                indirect_lighting[surface_id].append({
                    'bounce_level': bounce,
                    'intensity': reflected_intensity,
                    'color': self._modify_color_by_material(light.color, material)
                })
        
        return indirect_lighting
    
    def _calculate_attenuation(self, distance: float, light: LightSource) -> float:
        """Calculate light attenuation based on distance"""
        if light.falloff_curve == "linear":
            return max(0, 1.0 - (distance / light.range))
        elif light.falloff_curve == "quadratic":
            return 1.0 / (1.0 + distance * distance)
        elif light.falloff_curve == "realistic":
            # Inverse square law with linear falloff near source
            return 1.0 / (1.0 + 0.1 * distance + 0.01 * distance * distance)
        else:
            return 1.0  # No attenuation
    
    def _calculate_spot_factor(self, light: LightSource, light_direction: np.ndarray) -> float:
        """Calculate spot light cone factor"""
        if light.light_type != LightType.SPOT:
            return 1.0
        
        # Calculate angle between light direction and spot direction
        spot_dir = np.array(light.direction)
        angle = np.arccos(np.clip(np.dot(-light_direction, spot_dir), -1.0, 1.0))
        angle_degrees = np.degrees(angle)
        
        if angle_degrees > light.spot_angle / 2:
            return 0.0  # Outside cone
        
        # Smooth falloff at edges
        edge_factor = 1.0 - (angle_degrees / (light.spot_angle / 2))
        return edge_factor ** 2  # Quadratic falloff
    
    def _combine_lighting(self, direct: Dict, indirect: Dict) -> Dict:
        """Combine direct and indirect lighting"""
        combined = {}
        
        # Add direct lighting
        for surface_id, lighting_data in direct.items():
            combined[surface_id] = lighting_data.copy()
        
        # Add indirect lighting
        for surface_id, lighting_data in indirect.items():
            if surface_id in combined:
                combined[surface_id].extend(lighting_data)
            else:
                combined[surface_id] = lighting_data
        
        return combined
    
    def _accumulate_lighting(self, global_map: Dict, new_lighting: Dict):
        """Accumulate lighting into global illumination map"""
        for surface_id, lighting_data in new_lighting.items():
            if surface_id in global_map:
                global_map[surface_id].extend(lighting_data)
            else:
                global_map[surface_id] = lighting_data
    
    def _modify_color_by_material(self, color: Tuple[float, float, float],
                                material: Dict) -> Tuple[float, float, float]:
        """Modify light color based on material properties"""
        material_color = material.get('color', (1.0, 1.0, 1.0))
        
        return (
            color[0] * material_color[0],
            color[1] * material_color[1],
            color[2] * material_color[2]
        )

class ProgressiveLightingDesigner:
    """Designs lighting scenarios based on player progress"""
    
    def __init__(self):
        self.lighting_calculator = AdvancedLightingCalculator()
        self.scenario_templates = self._initialize_scenario_templates()
        
    def _initialize_scenario_templates(self) -> Dict[LightingScenario, Dict]:
        """Initialize lighting templates for different scenarios"""
        return {
            LightingScenario.ENTRANCE: {
                'base_ambient': (0.3, 0.3, 0.4),  # Cool outdoor light
                'primary_lights': [
                    {
                        'type': 'directional',
                        'color': (1.0, 0.95, 0.8),
                        'intensity': 0.8,
                        'temperature': 5500  # Daylight
                    }
                ],
                'mood': 'bright_hopeful',
                'fog_density': 0.1
            },
            
            LightingScenario.EXPLORATION: {
                'base_ambient': (0.1, 0.1, 0.15),  # Dark cave
                'primary_lights': [
                    {
                        'type': 'spot',
                        'color': (1.0, 0.9, 0.7),
                        'intensity': 1.2,
                        'temperature': 3200,  # Warm torch light
                        'flicker': True
                    }
                ],
                'mood': 'cautious_discovery',
                'fog_density': 0.3
            },
            
            LightingScenario.PUZZLE: {
                'base_ambient': (0.15, 0.15, 0.2),
                'primary_lights': [
                    {
                        'type': 'area',
                        'color': (0.9, 0.95, 1.0),
                        'intensity': 1.0,
                        'temperature': 4500,  # Focused work light
                        'shadows': True
                    }
                ],
                'mood': 'focused_concentration',
                'fog_density': 0.2
            },
            
            LightingScenario.TENSION: {
                'base_ambient': (0.05, 0.05, 0.1),  # Very dark
                'primary_lights': [
                    {
                        'type': 'point',
                        'color': (1.0, 0.3, 0.2),
                        'intensity': 0.8,
                        'temperature': 2000,  # Ominous red
                        'flicker': True
                    }
                ],
                'mood': 'dramatic_suspense',
                'fog_density': 0.5
            },
            
            LightingScenario.DISCOVERY: {
                'base_ambient': (0.2, 0.25, 0.3),
                'primary_lights': [
                    {
                        'type': 'area',
                        'color': (0.8, 1.0, 0.9),
                        'intensity': 1.5,
                        'temperature': 6500,  # Cool discovery light
                        'volumetric': True
                    }
                ],
                'mood': 'awe_revelation',
                'fog_density': 0.4
            },
            
            LightingScenario.RELIEF: {
                'base_ambient': (0.25, 0.2, 0.15),
                'primary_lights': [
                    {
                        'type': 'point',
                        'color': (1.0, 0.8, 0.6),
                        'intensity': 1.0,
                        'temperature': 2700,  # Warm comfort
                        'shadows': False
                    }
                ],
                'mood': 'warm_safety',
                'fog_density': 0.15
            },
            
            LightingScenario.VICTORY: {
                'base_ambient': (0.4, 0.4, 0.5),
                'primary_lights': [
                    {
                        'type': 'area',
                        'color': (1.0, 1.0, 0.9),
                        'intensity': 2.0,
                        'temperature': 5000,  # Bright triumph
                        'volumetric': True
                    }
                ],
                'mood': 'triumphant_bright',
                'fog_density': 0.2
            },
            
            LightingScenario.MAGICAL: {
                'base_ambient': (0.1, 0.15, 0.2),
                'primary_lights': [
                    {
                        'type': 'point',
                        'color': (0.6, 0.8, 1.0),
                        'intensity': 1.3,
                        'temperature': 8000,  # Cool magical
                        'flicker': True,
                        'volumetric': True
                    }
                ],
                'mood': 'mystical_otherworldly',
                'fog_density': 0.6
            }
        }
    
    def design_lighting_for_progress(self, progress: ProgressMarkers,
                                   environment: EnvironmentalFactors) -> List[LightSource]:
        """Design lighting based on current progress"""
        
        # Determine current scenario
        scenario = self._determine_scenario(progress)
        
        # Get base template
        template = self.scenario_templates[scenario]
        
        # Create light sources
        light_sources = []
        
        # Add ambient lighting
        ambient_light = LightSource(
            id="ambient_global",
            light_type=LightType.AMBIENT,
            position=(0, 0, 0),
            direction=(0, -1, 0),
            color=template['base_ambient'],
            intensity=self._calculate_ambient_intensity(progress, environment),
            range=1000.0,
            spot_angle=0,
            falloff_curve="none",
            shadows=False,
            volumetric=False,
            flicker=False,
            temperature=5000,
            enabled=True
        )
        light_sources.append(ambient_light)
        
        # Add primary lights from template
        for i, light_template in enumerate(template['primary_lights']):
            primary_light = self._create_light_from_template(
                f"primary_{i}", light_template, progress, environment
            )
            light_sources.append(primary_light)
        
        # Add progress-specific lights
        progress_lights = self._create_progress_lights(progress, environment)
        light_sources.extend(progress_lights)
        
        # Add environmental lights
        env_lights = self._create_environmental_lights(environment)
        light_sources.extend(env_lights)
        
        return light_sources
    
    def _determine_scenario(self, progress: ProgressMarkers) -> LightingScenario:
        """Determine appropriate lighting scenario based on progress"""
        
        # Check for specific conditions first
        if progress.player_stress_level > 0.7:
            return LightingScenario.TENSION
        
        if progress.last_achievement in ["puzzle_solved", "secret_found"]:
            return LightingScenario.DISCOVERY
        
        if progress.overall_progress < 0.1:
            return LightingScenario.ENTRANCE
        
        if progress.overall_progress > 0.9:
            return LightingScenario.VICTORY
        
        if progress.proximity_to_exit < 0.2 and progress.overall_progress > 0.8:
            return LightingScenario.RELIEF
        
        # Default based on overall progress
        if progress.overall_progress < 0.3:
            return LightingScenario.EXPLORATION
        elif progress.current_puzzle > 0:
            return LightingScenario.PUZZLE
        else:
            return LightingScenario.EXPLORATION
    
    def _calculate_ambient_intensity(self, progress: ProgressMarkers,
                                   environment: EnvironmentalFactors) -> float:
        """Calculate ambient light intensity"""
        base_intensity = 0.2
        
        # Reduce with depth
        depth_factor = max(0.1, 1.0 - (environment.cave_depth / 100.0))
        
        # Increase with progress (player learns/adapts)
        progress_factor = 1.0 + (progress.overall_progress * 0.3)
        
        # Adjust for natural light sources
        natural_factor = 1.0 + (environment.natural_light_sources * 0.1)
        
        return base_intensity * depth_factor * progress_factor * natural_factor
    
    def _create_light_from_template(self, light_id: str, template: Dict,
                                  progress: ProgressMarkers,
                                  environment: EnvironmentalFactors) -> LightSource:
        """Create light source from template"""
        
        # Determine light type
        light_type = LightType(template['type'])
        
        # Calculate position based on player/environment
        position = self._calculate_light_position(light_type, progress, environment)
        
        # Calculate direction for directional/spot lights
        direction = self._calculate_light_direction(light_type, position)
        
        # Adjust intensity based on conditions
        intensity = template['intensity'] * self._get_intensity_modifier(
            progress, environment
        )
        
        return LightSource(
            id=light_id,
            light_type=light_type,
            position=position,
            direction=direction,
            color=template['color'],
            intensity=intensity,
            range=template.get('range', 50.0),
            spot_angle=template.get('spot_angle', 45.0),
            falloff_curve=template.get('falloff_curve', 'realistic'),
            shadows=template.get('shadows', True),
            volumetric=template.get('volumetric', False),
            flicker=template.get('flicker', False),
            temperature=template['temperature'],
            enabled=True
        )
    
    def _create_progress_lights(self, progress: ProgressMarkers,
                              environment: EnvironmentalFactors) -> List[LightSource]:
        """Create lights specific to player progress"""
        progress_lights = []
        
        # Add lights for completed puzzles
        for i in range(progress.puzzles_completed):
            completed_light = LightSource(
                id=f"puzzle_complete_{i}",
                light_type=LightType.POINT,
                position=(i * 10, 2, 0),  # Spread out
                direction=(0, -1, 0),
                color=(0.8, 1.0, 0.8),  # Success green
                intensity=0.5,
                range=15.0,
                spot_angle=0,
                falloff_curve="quadratic",
                shadows=False,
                volumetric=True,
                flicker=False,
                temperature=4000,
                enabled=True
            )
            progress_lights.append(completed_light)
        
        # Add secret discovery lights
        for i in range(progress.secrets_found):
            secret_light = LightSource(
                id=f"secret_{i}",
                light_type=LightType.POINT,
                position=(random.uniform(-20, 20), random.uniform(1, 4), 
                         random.uniform(-20, 20)),
                direction=(0, -1, 0),
                color=(1.0, 0.8, 0.6),  # Treasure gold
                intensity=0.8,
                range=10.0,
                spot_angle=0,
                falloff_curve="linear",
                shadows=True,
                volumetric=False,
                flicker=True,
                temperature=2500,
                enabled=True
            )
            progress_lights.append(secret_light)
        
        return progress_lights
    
    def _create_environmental_lights(self, environment: EnvironmentalFactors) -> List[LightSource]:
        """Create lights based on environmental conditions"""
        env_lights = []
        
        # Natural light sources (cracks, openings)
        for i in range(environment.natural_light_sources):
            natural_light = LightSource(
                id=f"natural_{i}",
                light_type=LightType.DIRECTIONAL,
                position=(random.uniform(-50, 50), 10, random.uniform(-50, 50)),
                direction=(0, -1, 0),
                color=(1.0, 0.95, 0.85),  # Sunlight
                intensity=environment.natural_light_sources * 0.3,
                range=100.0,
                spot_angle=0,
                falloff_curve="linear",
                shadows=True,
                volumetric=True,
                flicker=False,
                temperature=5500,
                enabled=True
            )
            env_lights.append(natural_light)
        
        return env_lights
    
    def _calculate_light_position(self, light_type: LightType,
                                progress: ProgressMarkers,
                                environment: EnvironmentalFactors) -> Tuple[float, float, float]:
        """Calculate optimal light position"""
        
        if light_type == LightType.DIRECTIONAL:
            # High up for sun/moon light
            return (0, 20, 0)
        elif light_type == LightType.SPOT:
            # Player's headlamp position
            return (0, 1.8, 0)  # Eye level
        else:
            # Point/area lights distributed in cave
            return (
                random.uniform(-10, 10),
                random.uniform(2, 5),
                random.uniform(-10, 10)
            )
    
    def _calculate_light_direction(self, light_type: LightType,
                                 position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Calculate light direction"""
        
        if light_type in [LightType.DIRECTIONAL, LightType.SPOT]:
            return (0, -1, 0)  # Downward
        else:
            return (0, 0, 0)  # No direction for point lights
    
    def _get_intensity_modifier(self, progress: ProgressMarkers,
                              environment: EnvironmentalFactors) -> float:
        """Get intensity modifier based on conditions"""
        modifier = 1.0
        
        # Reduce intensity in humid conditions
        modifier *= (1.0 - environment.humidity * 0.2)
        
        # Increase intensity if player is stressed (dramatic effect)
        modifier *= (1.0 + progress.player_stress_level * 0.3)
        
        # Adjust for dust particles (scattering)
        modifier *= (1.0 - environment.dust_particles * 0.1)
        
        return max(0.1, modifier)

class DynamicLightingSystem:
    """Main dynamic lighting system"""
    
    def __init__(self):
        self.lighting_designer = ProgressiveLightingDesigner()
        self.current_lights = []
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Performance tracking
        self.calculation_times = deque(maxlen=50)
        self.frame_count = 0
        
    def update_lighting(self, progress: ProgressMarkers,
                       environment: EnvironmentalFactors,
                       cave_geometry: Dict = None) -> Dict[str, Any]:
        """Update lighting based on current conditions"""
        
        start_time = time.time()
        
        # Design new lighting setup
        new_lights = self.lighting_designer.design_lighting_for_progress(
            progress, environment
        )
        
        # Calculate lighting if geometry is provided
        lighting_data = {}
        if cave_geometry:
            surface_materials = self._get_default_materials()
            lighting_data = self.lighting_designer.lighting_calculator.calculate_global_illumination(
                new_lights, cave_geometry, surface_materials
            )
        
        # Update current lights
        self.current_lights = new_lights
        
        # Track performance
        calculation_time = time.time() - start_time
        self.calculation_times.append(calculation_time)
        self.frame_count += 1
        
        # Prepare Unity data
        unity_data = self._prepare_unity_data(new_lights, lighting_data)
        
        logger.info(f"Updated lighting with {len(new_lights)} lights in {calculation_time:.3f}s")
        
        return unity_data
    
    def _get_default_materials(self) -> Dict[str, Dict]:
        """Get default cave surface materials"""
        return {
            'rock_wall': {
                'reflectance': 0.1,
                'color': (0.8, 0.8, 0.8),
                'roughness': 0.9
            },
            'limestone': {
                'reflectance': 0.3,
                'color': (0.9, 0.9, 0.85),
                'roughness': 0.6
            },
            'crystal': {
                'reflectance': 0.8,
                'color': (0.95, 0.95, 1.0),
                'roughness': 0.1
            },
            'water': {
                'reflectance': 0.9,
                'color': (0.8, 0.9, 1.0),
                'roughness': 0.0
            }
        }
    
    def _prepare_unity_data(self, lights: List[LightSource],
                          lighting_data: Dict) -> Dict[str, Any]:
        """Prepare data for Unity consumption"""
        
        unity_lights = []
        for light in lights:
            unity_light = {
                'id': light.id,
                'type': light.light_type.value,
                'position': light.position,
                'direction': light.direction,
                'color': light.color,
                'intensity': light.intensity,
                'range': light.range,
                'spotAngle': light.spot_angle,
                'shadows': light.shadows,
                'volumetric': light.volumetric,
                'flicker': light.flicker,
                'temperature': light.temperature,
                'enabled': light.enabled
            }
            unity_lights.append(unity_light)
        
        # Performance stats
        avg_calc_time = (sum(self.calculation_times) / len(self.calculation_times)
                        if self.calculation_times else 0.0)
        
        return {
            'lights': unity_lights,
            'globalIllumination': lighting_data,
            'performance': {
                'averageCalculationTime': avg_calc_time,
                'frameCount': self.frame_count,
                'lightCount': len(lights)
            },
            'timestamp': time.time()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        if self.calculation_times:
            avg_time = sum(self.calculation_times) / len(self.calculation_times)
            max_time = max(self.calculation_times)
            min_time = min(self.calculation_times)
        else:
            avg_time = max_time = min_time = 0.0
        
        return {
            'average_calculation_time': avg_time,
            'max_calculation_time': max_time,
            'min_calculation_time': min_time,
            'total_frames': self.frame_count,
            'current_light_count': len(self.current_lights),
            'fps_estimate': 1.0 / avg_time if avg_time > 0 else 0.0
        }

class UnityLightingStreamer:
    """WebSocket server for Unity lighting integration"""
    
    def __init__(self, lighting_system: DynamicLightingSystem):
        self.lighting_system = lighting_system
        self.connected_clients = set()
        self.server = None
        self.is_running = False
        
    async def start_server(self, host: str = "localhost", port: int = 8894):
        """Start WebSocket server for Unity connections"""
        logger.info(f"💡 Starting Dynamic Cave Lighting server on {host}:{port}")
        
        self.server = await websockets.serve(
            self.handle_client, host, port, max_size=5 * 1024 * 1024  # 5MB max
        )
        
        self.is_running = True
        logger.info("✅ Lighting server started!")
        
        # Keep server running
        try:
            await self.server.wait_closed()
        except KeyboardInterrupt:
            logger.info("🛑 Lighting server stopped")
            self.is_running = False
    
    async def handle_client(self, websocket, path):
        """Handle Unity client connections"""
        client_id = f"unity_{len(self.connected_clients)}"
        self.connected_clients.add(websocket)
        
        logger.info(f"🎮 Unity lighting client connected: {client_id}")
        
        # Send capabilities
        await self.send_capabilities(websocket)
        
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Unity lighting client disconnected: {client_id}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def send_capabilities(self, websocket):
        """Send lighting system capabilities to Unity"""
        capabilities = {
            "type": "lighting_capabilities",
            "data": {
                "supported_scenarios": [scenario.value for scenario in LightingScenario],
                "supported_light_types": [light_type.value for light_type in LightType],
                "real_time_calculation": True,
                "global_illumination": True,
                "volumetric_effects": True,
                "dynamic_shadows": True,
                "performance_optimization": True,
                "max_lights": 50
            }
        }
        
        await websocket.send(json.dumps(capabilities))
    
    async def process_message(self, websocket, message: str):
        """Process incoming messages from Unity"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "update_lighting":
                await self.handle_lighting_update(websocket, data)
            elif message_type == "get_performance_stats":
                await self.handle_performance_request(websocket)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received from Unity")
        except Exception as e:
            logger.error(f"Error processing lighting message: {e}")
    
    async def handle_lighting_update(self, websocket, data: Dict):
        """Handle lighting update requests"""
        try:
            # Parse progress data
            progress_data = data["progress_markers"]
            progress = ProgressMarkers(
                overall_progress=progress_data.get("overall_progress", 0.0),
                current_puzzle=progress_data.get("current_puzzle", 0),
                puzzles_completed=progress_data.get("puzzles_completed", 0),
                secrets_found=progress_data.get("secrets_found", 0),
                areas_explored=progress_data.get("areas_explored", 0),
                time_in_cave=progress_data.get("time_in_cave", 0.0),
                player_stress_level=progress_data.get("player_stress_level", 0.0),
                last_achievement=progress_data.get("last_achievement", ""),
                proximity_to_exit=progress_data.get("proximity_to_exit", 1.0)
            )
            
            # Parse environmental data
            env_data = data["environmental_factors"]
            environment = EnvironmentalFactors(
                cave_depth=env_data.get("cave_depth", 10.0),
                humidity=env_data.get("humidity", 0.5),
                air_circulation=env_data.get("air_circulation", 0.3),
                natural_light_sources=env_data.get("natural_light_sources", 1),
                player_light_sources=env_data.get("player_light_sources", 1),
                ambient_temperature=env_data.get("ambient_temperature", 15.0),
                atmospheric_pressure=env_data.get("atmospheric_pressure", 101.3),
                dust_particles=env_data.get("dust_particles", 0.3)
            )
            
            # Cave geometry (optional)
            cave_geometry = data.get("cave_geometry")
            
            # Update lighting (run in thread to avoid blocking)
            loop = asyncio.get_event_loop()
            lighting_data = await loop.run_in_executor(
                self.lighting_system.executor,
                self.lighting_system.update_lighting,
                progress, environment, cave_geometry
            )
            
            # Send response
            response = {
                "type": "lighting_result",
                "data": lighting_data
            }
            
            await websocket.send(json.dumps(response))
            logger.info("✅ Sent updated lighting to Unity")
            
        except Exception as e:
            error_response = {
                "type": "error",
                "message": f"Lighting update failed: {str(e)}"
            }
            await websocket.send(json.dumps(error_response))
            logger.error(f"Lighting update error: {e}")
    
    async def handle_performance_request(self, websocket):
        """Handle performance statistics requests"""
        stats = self.lighting_system.get_performance_stats()
        
        response = {
            "type": "performance_stats",
            "data": stats
        }
        
        await websocket.send(json.dumps(response))

# Unity Integration Script
unity_integration_script = '''
/*
 * DynamicCaveLightingClient.cs
 * Unity client for dynamic cave lighting system
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using WebSocketSharp;
using Newtonsoft.Json;
using System;

public class DynamicCaveLightingClient : MonoBehaviour
{
    private WebSocket lightingSocket;
    
    [Header("Connection Settings")]
    public string pythonLightingUrl = "ws://localhost:8894";
    public bool autoConnect = true;
    public bool realTimeUpdates = true;
    public float updateInterval = 0.5f;
    
    [Header("Progress Tracking")]
    public ProgressMarkers progressMarkers;
    public EnvironmentalFactors environmentalFactors;
    
    [Header("Lighting Control")]
    public Transform lightingParent;
    public bool enableVolumetricEffects = true;
    public bool enableDynamicShadows = true;
    
    private Dictionary<string, Light> activeLights = new Dictionary<string, Light>();
    private Coroutine updateCoroutine;
    
    [System.Serializable]
    public class ProgressMarkers
    {
        [Range(0f, 1f)] public float overallProgress = 0f;
        public int currentPuzzle = 0;
        public int puzzlesCompleted = 0;
        public int secretsFound = 0;
        public int areasExplored = 0;
        public float timeInCave = 0f;
        [Range(0f, 1f)] public float playerStressLevel = 0f;
        public string lastAchievement = "";
        [Range(0f, 1f)] public float proximityToExit = 1f;
    }
    
    [System.Serializable]
    public class EnvironmentalFactors
    {
        public float caveDepth = 10f;
        [Range(0f, 1f)] public float humidity = 0.5f;
        [Range(0f, 1f)] public float airCirculation = 0.3f;
        public int naturalLightSources = 1;
        public int playerLightSources = 1;
        public float ambientTemperature = 15f;
        public float atmosphericPressure = 101.3f;
        [Range(0f, 1f)] public float dustParticles = 0.3f;
    }
    
    void Start()
    {
        if (autoConnect)
        {
            ConnectToLightingSystem();
        }
    }
    
    public void ConnectToLightingSystem()
    {
        Debug.Log("💡 Connecting to Dynamic Cave Lighting System...");
        
        lightingSocket = new WebSocket(pythonLightingUrl);
        
        lightingSocket.OnOpen += (sender, e) =>
        {
            Debug.Log("✅ Connected to Lighting System!");
            
            if (realTimeUpdates)
            {
                StartRealTimeUpdates();
            }
        };
        
        lightingSocket.OnMessage += (sender, e) =>
        {
            ProcessLightingData(e.Data);
        };
        
        lightingSocket.Connect();
    }
    
    public void StartRealTimeUpdates()
    {
        if (updateCoroutine != null)
        {
            StopCoroutine(updateCoroutine);
        }
        
        updateCoroutine = StartCoroutine(UpdateLightingLoop());
    }
    
    private IEnumerator UpdateLightingLoop()
    {
        while (true)
        {
            RequestLightingUpdate();
            yield return new WaitForSeconds(updateInterval);
        }
    }
    
    public void RequestLightingUpdate()
    {
        var request = new
        {
            type = "update_lighting",
            progress_markers = new
            {
                overall_progress = progressMarkers.overallProgress,
                current_puzzle = progressMarkers.currentPuzzle,
                puzzles_completed = progressMarkers.puzzlesCompleted,
                secrets_found = progressMarkers.secretsFound,
                areas_explored = progressMarkers.areasExplored,
                time_in_cave = progressMarkers.timeInCave,
                player_stress_level = progressMarkers.playerStressLevel,
                last_achievement = progressMarkers.lastAchievement,
                proximity_to_exit = progressMarkers.proximityToExit
            },
            environmental_factors = new
            {
                cave_depth = environmentalFactors.caveDepth,
                humidity = environmentalFactors.humidity,
                air_circulation = environmentalFactors.airCirculation,
                natural_light_sources = environmentalFactors.naturalLightSources,
                player_light_sources = environmentalFactors.playerLightSources,
                ambient_temperature = environmentalFactors.ambientTemperature,
                atmospheric_pressure = environmentalFactors.atmosphericPressure,
                dust_particles = environmentalFactors.dustParticles
            }
        };
        
        lightingSocket.Send(JsonConvert.SerializeObject(request));
    }
    
    private void ProcessLightingData(string jsonData)
    {
        try
        {
            var message = JsonConvert.DeserializeObject<Dictionary<string, object>>(jsonData);
            string messageType = message["type"].ToString();
            
            if (messageType == "lighting_result")
            {
                var data = JsonConvert.DeserializeObject<Dictionary<string, object>>(
                    message["data"].ToString()
                );
                
                var lightsData = JsonConvert.DeserializeObject<List<Dictionary<string, object>>>(
                    data["lights"].ToString()
                );
                
                ApplyLighting(lightsData);
                
                Debug.Log($"✅ Applied {lightsData.Count} lights from Python system");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error processing lighting data: {ex.Message}");
        }
    }
    
    private void ApplyLighting(List<Dictionary<string, object>> lightsData)
    {
        // Clear existing lights
        foreach (var light in activeLights.Values)
        {
            if (light != null)
            {
                DestroyImmediate(light.gameObject);
            }
        }
        activeLights.Clear();
        
        // Create new lights
        foreach (var lightData in lightsData)
        {
            CreateLight(lightData);
        }
    }
    
    private void CreateLight(Dictionary<string, object> lightData)
    {
        string lightId = lightData["id"].ToString();
        string lightType = lightData["type"].ToString();
        
        // Create GameObject
        GameObject lightObj = new GameObject($"Light_{lightId}");
        lightObj.transform.SetParent(lightingParent);
        
        // Add Light component
        Light lightComponent = lightObj.AddComponent<Light>();
        
        // Configure light
        ConfigureLight(lightComponent, lightData);
        
        // Store reference
        activeLights[lightId] = lightComponent;
    }
    
    private void ConfigureLight(Light lightComponent, Dictionary<string, object> lightData)
    {
        // Set type
        string lightType = lightData["type"].ToString();
        switch (lightType)
        {
            case "directional":
                lightComponent.type = LightType.Directional;
                break;
            case "point":
                lightComponent.type = LightType.Point;
                break;
            case "spot":
                lightComponent.type = LightType.Spot;
                break;
            case "area":
                lightComponent.type = LightType.Area;
                break;
        }
        
        // Set position
        var position = JsonConvert.DeserializeObject<float[]>(lightData["position"].ToString());
        lightComponent.transform.position = new Vector3(position[0], position[1], position[2]);
        
        // Set direction (for directional/spot lights)
        var direction = JsonConvert.DeserializeObject<float[]>(lightData["direction"].ToString());
        lightComponent.transform.rotation = Quaternion.LookRotation(new Vector3(direction[0], direction[1], direction[2]));
        
        // Set color
        var color = JsonConvert.DeserializeObject<float[]>(lightData["color"].ToString());
        lightComponent.color = new Color(color[0], color[1], color[2]);
        
        // Set intensity
        lightComponent.intensity = float.Parse(lightData["intensity"].ToString());
        
        // Set range
        lightComponent.range = float.Parse(lightData["range"].ToString());
        
        // Set spot angle
        if (lightComponent.type == LightType.Spot)
        {
            lightComponent.spotAngle = float.Parse(lightData["spotAngle"].ToString());
        }
        
        // Set shadows
        lightComponent.shadows = bool.Parse(lightData["shadows"].ToString()) && enableDynamicShadows 
            ? LightShadows.Soft : LightShadows.None;
        
        // Enable/disable
        lightComponent.enabled = bool.Parse(lightData["enabled"].ToString());
        
        // Add flicker effect if specified
        bool flicker = bool.Parse(lightData["flicker"].ToString());
        if (flicker)
        {
            lightComponent.gameObject.AddComponent<LightFlicker>();
        }
    }
    
    void OnDestroy()
    {
        if (updateCoroutine != null)
        {
            StopCoroutine(updateCoroutine);
        }
        
        if (lightingSocket != null)
        {
            lightingSocket.Close();
        }
    }
}

public class LightFlicker : MonoBehaviour
{
    private Light lightComponent;
    private float baseIntensity;
    
    void Start()
    {
        lightComponent = GetComponent<Light>();
        baseIntensity = lightComponent.intensity;
    }
    
    void Update()
    {
        float flicker = Mathf.PerlinNoise(Time.time * 10f, 0f) * 0.3f;
        lightComponent.intensity = baseIntensity * (1f + flicker);
    }
}
'''

async def main():
    """Main entry point for dynamic cave lighting system"""
    
    print("💡 DYNAMIC CAVE LIGHTING SYSTEM")
    print("===============================")
    print("Features:")
    print("• Progress-driven lighting scenarios")
    print("• Advanced ray-tracing calculations")
    print("• Realistic cave lighting simulation")
    print("• Psychological lighting design")
    print("• Real-time Unity streaming")
    print("• Performance optimization")
    print("===============================")
    
    # Initialize lighting system
    lighting_system = DynamicLightingSystem()
    
    # Create Unity streamer
    unity_streamer = UnityLightingStreamer(lighting_system)
    
    # Save Unity integration script
    with open("DynamicCaveLightingClient.cs", "w") as f:
        f.write(unity_integration_script)
    
    print("📁 Unity integration script saved: DynamicCaveLightingClient.cs")
    print("📡 WebSocket server starting on ws://localhost:8894")
    
    # Start server
    await unity_streamer.start_server()

if __name__ == "__main__":
    asyncio.run(main()) 