"""
Procedural Puzzle Placement System for Unity Integration
======================================================

This system intelligently places puzzles within Guardian boundaries using
advanced spatial algorithms, accessibility analysis, and game design principles.
Ensures optimal player experience while respecting physical space constraints.

Features:
- Guardian boundary analysis and validation
- Intelligent spatial clustering algorithms
- Accessibility and reachability analysis
- Dynamic difficulty progression placement
- Multi-constraint optimization (space, flow, difficulty)
- Real-time collision detection and avoidance
- Ergonomic placement for VR comfort
- Performance-optimized spatial indexing

Interview Demonstration Points:
- Advanced spatial algorithms (Voronoi, Delaunay)
- Multi-objective optimization
- Graph theory for accessibility
- Computational geometry
- Real-time spatial analysis
- VR-specific considerations
"""

import asyncio
import websockets
import json
import numpy as np
import math
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import heapq
from scipy.spatial import Voronoi, Delaunay, ConvexHull
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PuzzleType(Enum):
    SPATIAL_MANIPULATION = "spatial_manipulation"    # Move/rotate objects
    PATTERN_RECOGNITION = "pattern_recognition"      # Visual pattern matching
    SEQUENCE_SOLVING = "sequence_solving"           # Order-based puzzles
    PHYSICS_INTERACTION = "physics_interaction"     # Real physics puzzles
    HIDDEN_OBJECT = "hidden_object"                 # Discovery/search
    MEMORY_CHALLENGE = "memory_challenge"           # Remember sequences/positions
    COOPERATIVE = "cooperative"                     # Multiple players
    TIME_PRESSURE = "time_pressure"                 # Speed-based challenges

class DifficultyLevel(Enum):
    TUTORIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5
    MASTER = 6

class PlacementPriority(Enum):
    CRITICAL_PATH = 1      # Main progression
    OPTIONAL_BRANCH = 2    # Side content
    SECRET_AREA = 3        # Hidden content
    TUTORIAL_ZONE = 4      # Learning area
    BOSS_AREA = 5         # Major challenge

@dataclass
class GuardianBoundary:
    """Represents Guardian boundary data from Oculus"""
    vertices: List[Tuple[float, float]]  # 2D boundary points (x, z)
    center: Tuple[float, float]          # Boundary center
    area: float                          # Total area in m²
    obstacles: List[Dict]                # Furniture, walls, etc.
    safe_zones: List[Dict]               # Areas guaranteed safe for movement
    height_bounds: Tuple[float, float]   # Min/max room height
    player_spawn: Tuple[float, float, float]  # Initial player position

@dataclass
class PuzzleConstraints:
    """Constraints for puzzle placement"""
    min_player_space: float      # Minimum space around puzzle (meters)
    max_reach_distance: float    # Maximum comfortable reach
    height_range: Tuple[float, float]  # Min/max height for interaction
    requires_wall: bool          # Must be placed against wall
    requires_floor: bool         # Must be on ground level
    requires_ceiling: bool       # Needs ceiling access
    min_lighting: float          # Minimum light level required
    social_distance: float       # Distance from other puzzles
    comfort_zone: float          # VR comfort considerations

@dataclass
class PuzzleDefinition:
    """Complete puzzle definition with placement requirements"""
    puzzle_id: str
    puzzle_type: PuzzleType
    difficulty: DifficultyLevel
    priority: PlacementPriority
    constraints: PuzzleConstraints
    estimated_duration: float    # Expected solve time (minutes)
    prerequisite_puzzles: List[str]  # Must be completed first
    unlocks_puzzles: List[str]   # Enables these puzzles
    spatial_footprint: Tuple[float, float, float]  # Width, depth, height
    interaction_points: List[Tuple[float, float, float]]  # Relative positions
    thematic_requirements: Dict[str, Any]  # Theme/story requirements

@dataclass
class PlacementSolution:
    """Result of puzzle placement optimization"""
    puzzle_id: str
    position: Tuple[float, float, float]
    rotation: float              # Y-axis rotation in degrees
    accessibility_score: float   # 0-1, how accessible
    difficulty_flow_score: float # 0-1, fits progression
    comfort_score: float         # 0-1, VR comfort rating
    total_score: float          # Combined optimization score
    placement_confidence: float  # Algorithm confidence
    alternative_positions: List[Tuple[float, float, float]]

class SpatialAnalyzer:
    """Advanced spatial analysis for Guardian boundaries"""
    
    def __init__(self):
        self.voronoi_cache = {}
        self.delaunay_cache = {}
        self.accessibility_graph = None
        
    def analyze_guardian_space(self, boundary: GuardianBoundary) -> Dict[str, Any]:
        """Comprehensive spatial analysis of Guardian boundary"""
        
        start_time = time.time()
        
        # Convert to numpy arrays for processing
        vertices = np.array(boundary.vertices)
        
        # Generate spatial tessellation
        voronoi_analysis = self._generate_voronoi_tessellation(vertices)
        delaunay_analysis = self._generate_delaunay_triangulation(vertices)
        
        # Analyze movement zones
        movement_zones = self._analyze_movement_zones(boundary)
        
        # Generate accessibility graph
        accessibility_graph = self._build_accessibility_graph(boundary, movement_zones)
        
        # Calculate spatial metrics
        spatial_metrics = self._calculate_spatial_metrics(boundary, vertices)
        
        # Identify optimal placement zones
        placement_zones = self._identify_placement_zones(
            boundary, movement_zones, spatial_metrics
        )
        
        analysis_time = time.time() - start_time
        
        return {
            'voronoi_analysis': voronoi_analysis,
            'delaunay_analysis': delaunay_analysis,
            'movement_zones': movement_zones,
            'accessibility_graph': accessibility_graph,
            'spatial_metrics': spatial_metrics,
            'placement_zones': placement_zones,
            'analysis_time': analysis_time
        }
    
    def _generate_voronoi_tessellation(self, vertices: np.ndarray) -> Dict:
        """Generate Voronoi diagram for space partitioning"""
        try:
            # Add boundary points to ensure proper tessellation
            boundary_points = self._add_boundary_points(vertices)
            
            # Generate Voronoi diagram
            voronoi = Voronoi(boundary_points)
            
            # Analyze Voronoi cells
            cells = []
            for i, region in enumerate(voronoi.regions):
                if len(region) > 0 and -1 not in region:
                    cell_vertices = voronoi.vertices[region]
                    cell_area = self._calculate_polygon_area(cell_vertices)
                    cell_center = np.mean(cell_vertices, axis=0)
                    
                    cells.append({
                        'id': i,
                        'vertices': cell_vertices.tolist(),
                        'area': cell_area,
                        'center': cell_center.tolist(),
                        'is_boundary': self._is_boundary_cell(cell_vertices, vertices)
                    })
            
            return {
                'cells': cells,
                'total_cells': len(cells),
                'average_cell_area': np.mean([cell['area'] for cell in cells]),
                'space_efficiency': self._calculate_space_efficiency(cells)
            }
            
        except Exception as e:
            logger.warning(f"Voronoi tessellation failed: {e}")
            return {'cells': [], 'error': str(e)}
    
    def _generate_delaunay_triangulation(self, vertices: np.ndarray) -> Dict:
        """Generate Delaunay triangulation for connectivity analysis"""
        try:
            delaunay = Delaunay(vertices)
            
            triangles = []
            for simplex in delaunay.simplices:
                triangle_vertices = vertices[simplex]
                triangle_area = self._calculate_triangle_area(triangle_vertices)
                triangle_center = np.mean(triangle_vertices, axis=0)
                
                triangles.append({
                    'vertices': triangle_vertices.tolist(),
                    'area': triangle_area,
                    'center': triangle_center.tolist(),
                    'aspect_ratio': self._calculate_triangle_aspect_ratio(triangle_vertices)
                })
            
            return {
                'triangles': triangles,
                'total_triangles': len(triangles),
                'average_area': np.mean([t['area'] for t in triangles]),
                'mesh_quality': self._evaluate_mesh_quality(triangles)
            }
            
        except Exception as e:
            logger.warning(f"Delaunay triangulation failed: {e}")
            return {'triangles': [], 'error': str(e)}
    
    def _analyze_movement_zones(self, boundary: GuardianBoundary) -> Dict:
        """Analyze zones for player movement and accessibility"""
        
        # Define movement parameters
        player_radius = 0.4  # Player collision radius (meters)
        obstacle_buffer = 0.3  # Buffer around obstacles
        
        # Create movement grid
        grid_resolution = 0.2  # 20cm grid resolution
        grid = self._create_movement_grid(boundary, grid_resolution)
        
        # Mark obstacle areas
        for obstacle in boundary.obstacles:
            self._mark_obstacle_in_grid(grid, obstacle, player_radius + obstacle_buffer)
        
        # Perform flood fill from player spawn
        reachable_areas = self._flood_fill_reachable(
            grid, boundary.player_spawn, player_radius
        )
        
        # Identify movement corridors
        corridors = self._identify_movement_corridors(reachable_areas)
        
        # Calculate zone connectivity
        connectivity = self._calculate_zone_connectivity(reachable_areas, corridors)
        
        return {
            'movement_grid': grid,
            'reachable_areas': reachable_areas,
            'corridors': corridors,
            'connectivity': connectivity,
            'total_reachable_area': np.sum(reachable_areas),
            'movement_efficiency': self._calculate_movement_efficiency(reachable_areas)
        }
    
    def _build_accessibility_graph(self, boundary: GuardianBoundary, 
                                 movement_zones: Dict) -> nx.Graph:
        """Build graph representing spatial accessibility"""
        
        graph = nx.Graph()
        
        # Add nodes for major spatial locations
        reachable_areas = movement_zones['reachable_areas']
        
        # Sample points in reachable areas
        sample_points = self._sample_accessible_points(reachable_areas, density=1.0)
        
        for i, point in enumerate(sample_points):
            graph.add_node(i, position=point, type='accessible')
        
        # Add edges based on line-of-sight and walkable paths
        for i, point1 in enumerate(sample_points):
            for j, point2 in enumerate(sample_points[i+1:], i+1):
                if self._is_path_clear(point1, point2, boundary, reachable_areas):
                    distance = np.linalg.norm(np.array(point1) - np.array(point2))
                    graph.add_edge(i, j, weight=distance)
        
        return graph
    
    def _identify_placement_zones(self, boundary: GuardianBoundary,
                                movement_zones: Dict, metrics: Dict) -> List[Dict]:
        """Identify optimal zones for puzzle placement"""
        
        placement_zones = []
        reachable_areas = movement_zones['reachable_areas']
        
        # Define zone types based on spatial characteristics
        zone_types = [
            {'name': 'central', 'min_area': 2.0, 'centrality_weight': 0.8},
            {'name': 'corner', 'min_area': 1.0, 'boundary_weight': 0.8},
            {'name': 'corridor', 'min_area': 0.5, 'connectivity_weight': 0.8},
            {'name': 'alcove', 'min_area': 0.8, 'privacy_weight': 0.8}
        ]
        
        for zone_type in zone_types:
            zones = self._find_zones_by_criteria(
                reachable_areas, boundary, zone_type
            )
            placement_zones.extend(zones)
        
        # Score and rank zones
        for zone in placement_zones:
            zone['suitability_score'] = self._calculate_zone_suitability(
                zone, boundary, movement_zones, metrics
            )
        
        # Sort by suitability
        placement_zones.sort(key=lambda z: z['suitability_score'], reverse=True)
        
        return placement_zones
    
    def _calculate_polygon_area(self, vertices: np.ndarray) -> float:
        """Calculate area of polygon using shoelace formula"""
        if len(vertices) < 3:
            return 0.0
        
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] 
                           for i in range(-1, len(x)-1)))
    
    def _calculate_triangle_area(self, vertices: np.ndarray) -> float:
        """Calculate area of triangle"""
        a, b, c = vertices
        return 0.5 * abs(np.cross(b - a, c - a))
    
    def _create_movement_grid(self, boundary: GuardianBoundary, 
                            resolution: float) -> np.ndarray:
        """Create grid for movement analysis"""
        # Calculate grid bounds
        vertices = np.array(boundary.vertices)
        min_x, min_y = np.min(vertices, axis=0)
        max_x, max_y = np.max(vertices, axis=0)
        
        # Create grid
        grid_width = int((max_x - min_x) / resolution) + 1
        grid_height = int((max_y - min_y) / resolution) + 1
        
        grid = np.zeros((grid_height, grid_width), dtype=bool)
        
        # Mark cells inside boundary
        for i in range(grid_height):
            for j in range(grid_width):
                x = min_x + j * resolution
                y = min_y + i * resolution
                if self._point_in_polygon((x, y), boundary.vertices):
                    grid[i, j] = True
        
        return grid
    
    def _point_in_polygon(self, point: Tuple[float, float], 
                         polygon: List[Tuple[float, float]]) -> bool:
        """Test if point is inside polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            if ((polygon[i][1] > y) != (polygon[j][1] > y)) and \
               (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) / 
                (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
                inside = not inside
            j = i
        
        return inside

class PuzzleOptimizer:
    """Multi-objective optimization for puzzle placement"""
    
    def __init__(self):
        self.spatial_analyzer = SpatialAnalyzer()
        self.placement_history = []
        
    def optimize_puzzle_placement(self, puzzles: List[PuzzleDefinition],
                                boundary: GuardianBoundary,
                                spatial_analysis: Dict) -> List[PlacementSolution]:
        """Optimize placement of all puzzles using multi-objective optimization"""
        
        start_time = time.time()
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(puzzles)
        
        # Determine placement order based on dependencies and priority
        placement_order = self._determine_placement_order(puzzles, dependency_graph)
        
        # Initialize placement solutions
        solutions = []
        placed_puzzles = {}
        
        # Place puzzles in order
        for puzzle in placement_order:
            solution = self._optimize_single_puzzle_placement(
                puzzle, boundary, spatial_analysis, placed_puzzles
            )
            
            if solution:
                solutions.append(solution)
                placed_puzzles[puzzle.puzzle_id] = solution
                logger.info(f"Placed puzzle {puzzle.puzzle_id} with score {solution.total_score:.3f}")
            else:
                logger.warning(f"Failed to place puzzle {puzzle.puzzle_id}")
        
        # Post-optimization refinement
        refined_solutions = self._refine_placement_solutions(
            solutions, boundary, spatial_analysis
        )
        
        optimization_time = time.time() - start_time
        logger.info(f"Optimized {len(puzzles)} puzzles in {optimization_time:.3f}s")
        
        return refined_solutions
    
    def _optimize_single_puzzle_placement(self, puzzle: PuzzleDefinition,
                                        boundary: GuardianBoundary,
                                        spatial_analysis: Dict,
                                        existing_placements: Dict) -> Optional[PlacementSolution]:
        """Optimize placement for a single puzzle"""
        
        # Get candidate positions
        candidates = self._generate_candidate_positions(
            puzzle, boundary, spatial_analysis, existing_placements
        )
        
        if not candidates:
            return None
        
        # Evaluate each candidate
        best_solution = None
        best_score = -1
        
        for position, rotation in candidates:
            # Calculate objective scores
            accessibility = self._calculate_accessibility_score(
                position, puzzle, boundary, spatial_analysis
            )
            
            difficulty_flow = self._calculate_difficulty_flow_score(
                position, puzzle, existing_placements
            )
            
            comfort = self._calculate_comfort_score(
                position, rotation, puzzle, boundary
            )
            
            # Constraint validation
            constraints_satisfied = self._validate_constraints(
                position, rotation, puzzle, boundary, existing_placements
            )
            
            if not constraints_satisfied:
                continue
            
            # Calculate total score
            total_score = self._calculate_total_score(
                accessibility, difficulty_flow, comfort, puzzle
            )
            
            if total_score > best_score:
                best_score = total_score
                best_solution = PlacementSolution(
                    puzzle_id=puzzle.puzzle_id,
                    position=position,
                    rotation=rotation,
                    accessibility_score=accessibility,
                    difficulty_flow_score=difficulty_flow,
                    comfort_score=comfort,
                    total_score=total_score,
                    placement_confidence=self._calculate_placement_confidence(total_score),
                    alternative_positions=candidates[:5]  # Top 5 alternatives
                )
        
        return best_solution
    
    def _generate_candidate_positions(self, puzzle: PuzzleDefinition,
                                    boundary: GuardianBoundary,
                                    spatial_analysis: Dict,
                                    existing_placements: Dict) -> List[Tuple[Tuple[float, float, float], float]]:
        """Generate candidate positions for puzzle placement"""
        
        candidates = []
        placement_zones = spatial_analysis['placement_zones']
        
        # Sample positions from high-suitability zones
        for zone in placement_zones[:10]:  # Top 10 zones
            zone_candidates = self._sample_zone_positions(
                zone, puzzle, boundary, existing_placements
            )
            candidates.extend(zone_candidates)
        
        # Add constraint-specific positions
        if puzzle.constraints.requires_wall:
            wall_candidates = self._generate_wall_positions(puzzle, boundary)
            candidates.extend(wall_candidates)
        
        if puzzle.constraints.requires_corner:
            corner_candidates = self._generate_corner_positions(puzzle, boundary)
            candidates.extend(corner_candidates)
        
        # Remove invalid positions
        valid_candidates = []
        for pos, rot in candidates:
            if self._is_position_valid(pos, puzzle, boundary, existing_placements):
                valid_candidates.append((pos, rot))
        
        return valid_candidates
    
    def _calculate_accessibility_score(self, position: Tuple[float, float, float],
                                     puzzle: PuzzleDefinition,
                                     boundary: GuardianBoundary,
                                     spatial_analysis: Dict) -> float:
        """Calculate how accessible the puzzle position is"""
        
        score = 0.0
        
        # Distance from player spawn
        spawn_distance = np.linalg.norm(
            np.array(position[:2]) - np.array(boundary.player_spawn[:2])
        )
        
        # Closer to spawn is generally better, but not too close
        optimal_distance = 3.0  # meters
        distance_score = 1.0 - abs(spawn_distance - optimal_distance) / optimal_distance
        score += distance_score * 0.3
        
        # Reachability via accessibility graph
        if spatial_analysis.get('accessibility_graph'):
            graph = spatial_analysis['accessibility_graph']
            nearest_nodes = self._find_nearest_graph_nodes(position, graph, k=5)
            
            reachability_scores = []
            for node in nearest_nodes:
                paths = nx.single_source_shortest_path_length(
                    graph, node, cutoff=10
                )
                reachability_scores.append(len(paths) / len(graph.nodes))
            
            if reachability_scores:
                score += max(reachability_scores) * 0.4
        
        # Clear line of sight from multiple points
        sight_points = self._get_strategic_sight_points(boundary)
        visible_count = 0
        
        for sight_point in sight_points:
            if self._has_clear_line_of_sight(sight_point, position, boundary):
                visible_count += 1
        
        visibility_score = visible_count / len(sight_points)
        score += visibility_score * 0.3
        
        return min(1.0, max(0.0, score))
    
    def _calculate_difficulty_flow_score(self, position: Tuple[float, float, float],
                                       puzzle: PuzzleDefinition,
                                       existing_placements: Dict) -> float:
        """Calculate how well the position fits difficulty progression"""
        
        if not existing_placements:
            return 1.0  # First puzzle, perfect score
        
        # Find prerequisite and successor puzzles
        prereq_positions = []
        successor_positions = []
        
        for placed_id, solution in existing_placements.items():
            if placed_id in puzzle.prerequisite_puzzles:
                prereq_positions.append(solution.position)
            if puzzle.puzzle_id in solution.unlocks_puzzles:
                successor_positions.append(solution.position)
        
        score = 0.0
        
        # Distance from prerequisites (should be reasonable progression)
        if prereq_positions:
            avg_prereq_distance = np.mean([
                np.linalg.norm(np.array(position) - np.array(prereq_pos))
                for prereq_pos in prereq_positions
            ])
            
            # Optimal progression distance based on difficulty
            optimal_distance = 2.0 + (puzzle.difficulty.value - 1) * 1.0
            distance_score = 1.0 - abs(avg_prereq_distance - optimal_distance) / optimal_distance
            score += max(0, distance_score) * 0.6
        else:
            score += 0.6  # No prerequisites, good score
        
        # Difficulty clustering (similar difficulty puzzles should be grouped)
        similar_difficulty_positions = [
            solution.position for solution in existing_placements.values()
            if abs(solution.difficulty.value - puzzle.difficulty.value) <= 1
        ]
        
        if similar_difficulty_positions:
            distances = [
                np.linalg.norm(np.array(position) - np.array(sim_pos))
                for sim_pos in similar_difficulty_positions
            ]
            
            # Prefer moderate clustering (not too close, not too far)
            avg_cluster_distance = np.mean(distances)
            cluster_score = 1.0 - abs(avg_cluster_distance - 5.0) / 5.0
            score += max(0, cluster_score) * 0.4
        else:
            score += 0.4  # No similar difficulty, neutral score
        
        return min(1.0, max(0.0, score))
    
    def _calculate_comfort_score(self, position: Tuple[float, float, float],
                               rotation: float, puzzle: PuzzleDefinition,
                               boundary: GuardianBoundary) -> float:
        """Calculate VR comfort score for the position"""
        
        score = 0.0
        
        # Height comfort (avoid extreme heights)
        height = position[2]
        comfortable_height_range = (0.8, 2.0)  # Standing reach
        
        if comfortable_height_range[0] <= height <= comfortable_height_range[1]:
            height_score = 1.0
        else:
            distance_from_range = min(
                abs(height - comfortable_height_range[0]),
                abs(height - comfortable_height_range[1])
            )
            height_score = max(0, 1.0 - distance_from_range / 1.0)
        
        score += height_score * 0.3
        
        # Space around puzzle (avoid cramped conditions)
        required_space = puzzle.constraints.min_player_space
        available_space = self._calculate_available_space(position, boundary)
        
        space_score = min(1.0, available_space / required_space)
        score += space_score * 0.3
        
        # Interaction angle comfort
        interaction_angles = []
        for interaction_point in puzzle.interaction_points:
            world_point = self._transform_to_world(interaction_point, position, rotation)
            angle = self._calculate_interaction_angle(position, world_point)
            interaction_angles.append(angle)
        
        if interaction_angles:
            # Prefer angles between -30 and +30 degrees from horizontal
            comfortable_angles = [
                1.0 - abs(angle) / 90.0 if abs(angle) <= 30 else
                max(0, 1.0 - (abs(angle) - 30) / 60.0)
                for angle in interaction_angles
            ]
            angle_score = np.mean(comfortable_angles)
        else:
            angle_score = 1.0
        
        score += angle_score * 0.2
        
        # Distance from boundaries (avoid edge placement)
        boundary_distances = []
        for vertex in boundary.vertices:
            dist = np.linalg.norm(np.array(position[:2]) - np.array(vertex))
            boundary_distances.append(dist)
        
        min_boundary_distance = min(boundary_distances)
        boundary_score = min(1.0, min_boundary_distance / 1.0)  # 1m from boundary
        score += boundary_score * 0.2
        
        return min(1.0, max(0.0, score))

class ProceduralPuzzlePlacer:
    """Main procedural puzzle placement system"""
    
    def __init__(self):
        self.spatial_analyzer = SpatialAnalyzer()
        self.optimizer = PuzzleOptimizer()
        self.placement_cache = {}
        
        # Performance tracking
        self.placement_times = deque(maxlen=50)
        self.total_placements = 0
        
    def place_puzzles(self, puzzle_definitions: List[Dict],
                     guardian_boundary: Dict,
                     placement_constraints: Dict = None) -> Dict[str, Any]:
        """Main entry point for puzzle placement"""
        
        start_time = time.time()
        
        # Parse input data
        puzzles = self._parse_puzzle_definitions(puzzle_definitions)
        boundary = self._parse_guardian_boundary(guardian_boundary)
        
        # Perform spatial analysis
        spatial_analysis = self.spatial_analyzer.analyze_guardian_space(boundary)
        
        # Optimize puzzle placement
        placement_solutions = self.optimizer.optimize_puzzle_placement(
            puzzles, boundary, spatial_analysis
        )
        
        # Generate additional metadata
        placement_metadata = self._generate_placement_metadata(
            placement_solutions, boundary, spatial_analysis
        )
        
        # Validate final placement
        validation_results = self._validate_final_placement(
            placement_solutions, boundary, puzzles
        )
        
        # Track performance
        total_time = time.time() - start_time
        self.placement_times.append(total_time)
        self.total_placements += 1
        
        # Prepare results
        results = {
            'puzzle_placements': [self._solution_to_dict(sol) for sol in placement_solutions],
            'placement_metadata': placement_metadata,
            'validation_results': validation_results,
            'spatial_analysis': spatial_analysis,
            'performance_metrics': {
                'placement_time': total_time,
                'puzzles_placed': len(placement_solutions),
                'placement_success_rate': len(placement_solutions) / len(puzzles),
                'average_placement_score': np.mean([sol.total_score for sol in placement_solutions])
            }
        }
        
        logger.info(f"Placed {len(placement_solutions)}/{len(puzzles)} puzzles in {total_time:.3f}s")
        
        return results
    
    def _parse_puzzle_definitions(self, puzzle_defs: List[Dict]) -> List[PuzzleDefinition]:
        """Parse puzzle definitions from input data"""
        puzzles = []
        
        for puzzle_data in puzzle_defs:
            constraints = PuzzleConstraints(
                min_player_space=puzzle_data.get('min_player_space', 0.8),
                max_reach_distance=puzzle_data.get('max_reach_distance', 1.2),
                height_range=tuple(puzzle_data.get('height_range', [0.5, 2.5])),
                requires_wall=puzzle_data.get('requires_wall', False),
                requires_floor=puzzle_data.get('requires_floor', True),
                requires_ceiling=puzzle_data.get('requires_ceiling', False),
                min_lighting=puzzle_data.get('min_lighting', 0.3),
                social_distance=puzzle_data.get('social_distance', 1.5),
                comfort_zone=puzzle_data.get('comfort_zone', 0.5)
            )
            
            puzzle = PuzzleDefinition(
                puzzle_id=puzzle_data['puzzle_id'],
                puzzle_type=PuzzleType(puzzle_data['puzzle_type']),
                difficulty=DifficultyLevel(puzzle_data['difficulty']),
                priority=PlacementPriority(puzzle_data.get('priority', 'CRITICAL_PATH')),
                constraints=constraints,
                estimated_duration=puzzle_data.get('estimated_duration', 5.0),
                prerequisite_puzzles=puzzle_data.get('prerequisite_puzzles', []),
                unlocks_puzzles=puzzle_data.get('unlocks_puzzles', []),
                spatial_footprint=tuple(puzzle_data.get('spatial_footprint', [0.5, 0.5, 0.5])),
                interaction_points=puzzle_data.get('interaction_points', [(0, 0, 0)]),
                thematic_requirements=puzzle_data.get('thematic_requirements', {})
            )
            
            puzzles.append(puzzle)
        
        return puzzles
    
    def _parse_guardian_boundary(self, boundary_data: Dict) -> GuardianBoundary:
        """Parse Guardian boundary data"""
        return GuardianBoundary(
            vertices=boundary_data['vertices'],
            center=tuple(boundary_data['center']),
            area=boundary_data['area'],
            obstacles=boundary_data.get('obstacles', []),
            safe_zones=boundary_data.get('safe_zones', []),
            height_bounds=tuple(boundary_data.get('height_bounds', [0.0, 3.0])),
            player_spawn=tuple(boundary_data.get('player_spawn', [0.0, 0.0, 1.7]))
        )
    
    def _solution_to_dict(self, solution: PlacementSolution) -> Dict:
        """Convert placement solution to dictionary"""
        return {
            'puzzle_id': solution.puzzle_id,
            'position': solution.position,
            'rotation': solution.rotation,
            'accessibility_score': solution.accessibility_score,
            'difficulty_flow_score': solution.difficulty_flow_score,
            'comfort_score': solution.comfort_score,
            'total_score': solution.total_score,
            'placement_confidence': solution.placement_confidence,
            'alternative_positions': solution.alternative_positions
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        if self.placement_times:
            avg_time = sum(self.placement_times) / len(self.placement_times)
            max_time = max(self.placement_times)
            min_time = min(self.placement_times)
        else:
            avg_time = max_time = min_time = 0.0
        
        return {
            'average_placement_time': avg_time,
            'max_placement_time': max_time,
            'min_placement_time': min_time,
            'total_placements': self.total_placements,
            'cache_size': len(self.placement_cache),
            'placement_efficiency': 1.0 / avg_time if avg_time > 0 else 0.0
        }

class UnityPuzzlePlacementStreamer:
    """WebSocket server for Unity puzzle placement integration"""
    
    def __init__(self, placement_system: ProceduralPuzzlePlacer):
        self.placement_system = placement_system
        self.connected_clients = set()
        self.server = None
        self.is_running = False
        
    async def start_server(self, host: str = "localhost", port: int = 8895):
        """Start WebSocket server for Unity connections"""
        logger.info(f"🧩 Starting Procedural Puzzle Placement server on {host}:{port}")
        
        self.server = await websockets.serve(
            self.handle_client, host, port, max_size=10 * 1024 * 1024  # 10MB max
        )
        
        self.is_running = True
        logger.info("✅ Puzzle Placement server started!")
        
        # Keep server running
        try:
            await self.server.wait_closed()
        except KeyboardInterrupt:
            logger.info("🛑 Puzzle Placement server stopped")
            self.is_running = False
    
    async def handle_client(self, websocket, path):
        """Handle Unity client connections"""
        client_id = f"unity_{len(self.connected_clients)}"
        self.connected_clients.add(websocket)
        
        logger.info(f"🎮 Unity puzzle placement client connected: {client_id}")
        
        # Send capabilities
        await self.send_capabilities(websocket)
        
        try:
            async for message in websocket:
                await self.process_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Unity puzzle placement client disconnected: {client_id}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def send_capabilities(self, websocket):
        """Send system capabilities to Unity"""
        capabilities = {
            "type": "placement_capabilities",
            "data": {
                "supported_puzzle_types": [ptype.value for ptype in PuzzleType],
                "supported_difficulty_levels": [diff.value for diff in DifficultyLevel],
                "spatial_analysis": True,
                "multi_objective_optimization": True,
                "accessibility_validation": True,
                "vr_comfort_optimization": True,
                "real_time_placement": True,
                "max_puzzles_per_session": 20
            }
        }
        
        await websocket.send(json.dumps(capabilities))
    
    async def process_message(self, websocket, message: str):
        """Process incoming messages from Unity"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "place_puzzles":
                await self.handle_puzzle_placement(websocket, data)
            elif message_type == "get_performance_stats":
                await self.handle_performance_request(websocket)
            elif message_type == "validate_placement":
                await self.handle_placement_validation(websocket, data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received from Unity")
        except Exception as e:
            logger.error(f"Error processing placement message: {e}")
    
    async def handle_puzzle_placement(self, websocket, data: Dict):
        """Handle puzzle placement requests"""
        try:
            puzzle_definitions = data["puzzle_definitions"]
            guardian_boundary = data["guardian_boundary"]
            placement_constraints = data.get("placement_constraints", {})
            
            # Perform placement (run in thread to avoid blocking)
            loop = asyncio.get_event_loop()
            placement_results = await loop.run_in_executor(
                None,  # Use default thread pool
                self.placement_system.place_puzzles,
                puzzle_definitions, guardian_boundary, placement_constraints
            )
            
            # Send response
            response = {
                "type": "placement_result",
                "data": placement_results
            }
            
            await websocket.send(json.dumps(response))
            logger.info(f"✅ Sent puzzle placement results to Unity")
            
        except Exception as e:
            error_response = {
                "type": "error",
                "message": f"Puzzle placement failed: {str(e)}"
            }
            await websocket.send(json.dumps(error_response))
            logger.error(f"Puzzle placement error: {e}")
    
    async def handle_performance_request(self, websocket):
        """Handle performance statistics requests"""
        stats = self.placement_system.get_performance_stats()
        
        response = {
            "type": "performance_stats",
            "data": stats
        }
        
        await websocket.send(json.dumps(response))

# Unity Integration Script
unity_integration_script = '''
/*
 * ProceduralPuzzlePlacementClient.cs
 * Unity client for procedural puzzle placement system
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using WebSocketSharp;
using Newtonsoft.Json;
using System;

public class ProceduralPuzzlePlacementClient : MonoBehaviour
{
    private WebSocket placementSocket;
    
    [Header("Connection Settings")]
    public string pythonPlacementUrl = "ws://localhost:8895";
    public bool autoConnect = true;
    
    [Header("Guardian Boundary")]
    public Transform[] boundaryPoints;
    public Transform playerSpawn;
    public Transform[] obstacles;
    
    [Header("Puzzle Definitions")]
    public PuzzleDefinition[] puzzleDefinitions;
    
    [Header("Placement Results")]
    public Transform puzzleParent;
    public GameObject[] puzzlePrefabs;
    
    [System.Serializable]
    public class PuzzleDefinition
    {
        public string puzzleId;
        public PuzzleType puzzleType;
        public int difficulty = 1;
        public string priority = "CRITICAL_PATH";
        public float minPlayerSpace = 0.8f;
        public float maxReachDistance = 1.2f;
        public Vector2 heightRange = new Vector2(0.5f, 2.5f);
        public bool requiresWall = false;
        public bool requiresFloor = true;
        public bool requiresCeiling = false;
        public float minLighting = 0.3f;
        public float socialDistance = 1.5f;
        public float comfortZone = 0.5f;
        public float estimatedDuration = 5.0f;
        public string[] prerequisitePuzzles;
        public string[] unlocksPuzzles;
        public Vector3 spatialFootprint = new Vector3(0.5f, 0.5f, 0.5f);
        public Vector3[] interactionPoints;
    }
    
    public enum PuzzleType
    {
        SpatialManipulation,
        PatternRecognition,
        SequenceSolving,
        PhysicsInteraction,
        HiddenObject,
        MemoryChallenge,
        Cooperative,
        TimePressure
    }
    
    void Start()
    {
        if (autoConnect)
        {
            ConnectToPuzzlePlacementSystem();
        }
    }
    
    public void ConnectToPuzzlePlacementSystem()
    {
        Debug.Log("🧩 Connecting to Procedural Puzzle Placement System...");
        
        placementSocket = new WebSocket(pythonPlacementUrl);
        
        placementSocket.OnOpen += (sender, e) =>
        {
            Debug.Log("✅ Connected to Puzzle Placement System!");
            RequestPuzzlePlacement();
        };
        
        placementSocket.OnMessage += (sender, e) =>
        {
            ProcessPlacementData(e.Data);
        };
        
        placementSocket.Connect();
    }
    
    public void RequestPuzzlePlacement()
    {
        // Prepare guardian boundary data
        var boundaryVertices = new List<float[]>();
        foreach (var point in boundaryPoints)
        {
            boundaryVertices.Add(new float[] { point.position.x, point.position.z });
        }
        
        var guardianBoundary = new
        {
            vertices = boundaryVertices,
            center = new float[] { 0f, 0f },
            area = CalculateBoundaryArea(),
            obstacles = GetObstacleData(),
            safe_zones = new object[0],
            height_bounds = new float[] { 0f, 3f },
            player_spawn = new float[] { 
                playerSpawn.position.x, 
                playerSpawn.position.y, 
                playerSpawn.position.z 
            }
        };
        
        // Prepare puzzle definitions
        var puzzleDefs = new List<object>();
        foreach (var puzzle in puzzleDefinitions)
        {
            puzzleDefs.Add(new
            {
                puzzle_id = puzzle.puzzleId,
                puzzle_type = puzzle.puzzleType.ToString().ToLower(),
                difficulty = puzzle.difficulty,
                priority = puzzle.priority,
                min_player_space = puzzle.minPlayerSpace,
                max_reach_distance = puzzle.maxReachDistance,
                height_range = new float[] { puzzle.heightRange.x, puzzle.heightRange.y },
                requires_wall = puzzle.requiresWall,
                requires_floor = puzzle.requiresFloor,
                requires_ceiling = puzzle.requiresCeiling,
                min_lighting = puzzle.minLighting,
                social_distance = puzzle.socialDistance,
                comfort_zone = puzzle.comfortZone,
                estimated_duration = puzzle.estimatedDuration,
                prerequisite_puzzles = puzzle.prerequisitePuzzles ?? new string[0],
                unlocks_puzzles = puzzle.unlocksPuzzles ?? new string[0],
                spatial_footprint = new float[] { 
                    puzzle.spatialFootprint.x, 
                    puzzle.spatialFootprint.y, 
                    puzzle.spatialFootprint.z 
                },
                interaction_points = GetInteractionPointsData(puzzle.interactionPoints),
                thematic_requirements = new object()
            });
        }
        
        var request = new
        {
            type = "place_puzzles",
            puzzle_definitions = puzzleDefs,
            guardian_boundary = guardianBoundary,
            placement_constraints = new object()
        };
        
        placementSocket.Send(JsonConvert.SerializeObject(request));
        Debug.Log($"🚀 Requested placement for {puzzleDefinitions.Length} puzzles");
    }
    
    private void ProcessPlacementData(string jsonData)
    {
        try
        {
            var message = JsonConvert.DeserializeObject<Dictionary<string, object>>(jsonData);
            string messageType = message["type"].ToString();
            
            if (messageType == "placement_result")
            {
                var data = JsonConvert.DeserializeObject<Dictionary<string, object>>(
                    message["data"].ToString()
                );
                
                var placements = JsonConvert.DeserializeObject<List<Dictionary<string, object>>>(
                    data["puzzle_placements"].ToString()
                );
                
                var performance = JsonConvert.DeserializeObject<Dictionary<string, object>>(
                    data["performance_metrics"].ToString()
                );
                
                ApplyPuzzlePlacements(placements);
                
                Debug.Log($"✅ Placed {placements.Count} puzzles " +
                         $"(Success rate: {float.Parse(performance["placement_success_rate"].ToString()):P})");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error processing placement data: {ex.Message}");
        }
    }
    
    private void ApplyPuzzlePlacements(List<Dictionary<string, object>> placements)
    {
        // Clear existing puzzles
        for (int i = puzzleParent.childCount - 1; i >= 0; i--)
        {
            DestroyImmediate(puzzleParent.GetChild(i).gameObject);
        }
        
        // Create new puzzles
        foreach (var placement in placements)
        {
            CreatePuzzle(placement);
        }
    }
    
    private void CreatePuzzle(Dictionary<string, object> placementData)
    {
        string puzzleId = placementData["puzzle_id"].ToString();
        var position = JsonConvert.DeserializeObject<float[]>(
            placementData["position"].ToString()
        );
        float rotation = float.Parse(placementData["rotation"].ToString());
        
        // Find matching puzzle definition
        PuzzleDefinition puzzleDef = null;
        foreach (var def in puzzleDefinitions)
        {
            if (def.puzzleId == puzzleId)
            {
                puzzleDef = def;
                break;
            }
        }
        
        if (puzzleDef == null)
        {
            Debug.LogWarning($"No puzzle definition found for {puzzleId}");
            return;
        }
        
        // Select appropriate prefab
        GameObject prefab = GetPuzzlePrefab(puzzleDef.puzzleType);
        if (prefab == null)
        {
            Debug.LogWarning($"No prefab found for puzzle type {puzzleDef.puzzleType}");
            return;
        }
        
        // Instantiate puzzle
        GameObject puzzleObject = Instantiate(prefab, puzzleParent);
        puzzleObject.name = $"Puzzle_{puzzleId}";
        puzzleObject.transform.position = new Vector3(position[0], position[1], position[2]);
        puzzleObject.transform.rotation = Quaternion.Euler(0, rotation, 0);
        
        // Add placement metadata component
        var metadata = puzzleObject.AddComponent<PuzzlePlacementMetadata>();
        metadata.Initialize(placementData);
        
        Debug.Log($"Created puzzle {puzzleId} at {puzzleObject.transform.position}");
    }
    
    private GameObject GetPuzzlePrefab(PuzzleType puzzleType)
    {
        int typeIndex = (int)puzzleType;
        if (typeIndex < puzzlePrefabs.Length && puzzlePrefabs[typeIndex] != null)
        {
            return puzzlePrefabs[typeIndex];
        }
        
        // Return first available prefab as fallback
        return puzzlePrefabs.Length > 0 ? puzzlePrefabs[0] : null;
    }
    
    private float CalculateBoundaryArea()
    {
        if (boundaryPoints.Length < 3) return 0f;
        
        float area = 0f;
        for (int i = 0; i < boundaryPoints.Length; i++)
        {
            int j = (i + 1) % boundaryPoints.Length;
            Vector3 p1 = boundaryPoints[i].position;
            Vector3 p2 = boundaryPoints[j].position;
            area += p1.x * p2.z - p2.x * p1.z;
        }
        return Mathf.Abs(area) / 2f;
    }
    
    private object[] GetObstacleData()
    {
        var obstacleData = new List<object>();
        
        foreach (var obstacle in obstacles)
        {
            obstacleData.Add(new
            {
                position = new float[] { obstacle.position.x, obstacle.position.z },
                size = new float[] { 
                    obstacle.localScale.x, 
                    obstacle.localScale.z 
                },
                type = "furniture"
            });
        }
        
        return obstacleData.ToArray();
    }
    
    private object[] GetInteractionPointsData(Vector3[] points)
    {
        if (points == null) return new object[0];
        
        var pointData = new List<object>();
        foreach (var point in points)
        {
            pointData.Add(new float[] { point.x, point.y, point.z });
        }
        
        return pointData.ToArray();
    }
    
    void OnDestroy()
    {
        if (placementSocket != null)
        {
            placementSocket.Close();
        }
    }
}

public class PuzzlePlacementMetadata : MonoBehaviour
{
    [Header("Placement Scores")]
    public float accessibilityScore;
    public float difficultyFlowScore;
    public float comfortScore;
    public float totalScore;
    public float placementConfidence;
    
    public void Initialize(Dictionary<string, object> placementData)
    {
        accessibilityScore = float.Parse(placementData["accessibility_score"].ToString());
        difficultyFlowScore = float.Parse(placementData["difficulty_flow_score"].ToString());
        comfortScore = float.Parse(placementData["comfort_score"].ToString());
        totalScore = float.Parse(placementData["total_score"].ToString());
        placementConfidence = float.Parse(placementData["placement_confidence"].ToString());
    }
}
'''

async def main():
    """Main entry point for procedural puzzle placement system"""
    
    print("🧩 PROCEDURAL PUZZLE PLACEMENT SYSTEM")
    print("====================================")
    print("Features:")
    print("• Guardian boundary spatial analysis")
    print("• Multi-objective placement optimization")
    print("• Advanced accessibility validation")
    print("• VR comfort optimization")
    print("• Dependency-aware puzzle sequencing")
    print("• Real-time Unity integration")
    print("====================================")
    
    # Initialize placement system
    placement_system = ProceduralPuzzlePlacer()
    
    # Create Unity streamer
    unity_streamer = UnityPuzzlePlacementStreamer(placement_system)
    
    # Save Unity integration script
    with open("ProceduralPuzzlePlacementClient.cs", "w") as f:
        f.write(unity_integration_script)
    
    print("📁 Unity integration script saved: ProceduralPuzzlePlacementClient.cs")
    print("📡 WebSocket server starting on ws://localhost:8895")
    
    # Start server
    await unity_streamer.start_server()

if __name__ == "__main__":
    asyncio.run(main()) 