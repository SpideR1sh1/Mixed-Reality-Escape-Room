# Mixed Reality Escape Room - Project Index

## 📁 Project Structure Overview

### 🎮 Unity Scripts (Scripts/)
#### Core Systems
- **GameManager.cs** - Central game state management, performance level control, and adaptive quality system
- **PerformanceMonitor.cs** - Real-time performance tracking (FPS, memory, CPU/GPU usage, thermal state)
- **PerformanceMonitorIntegration.cs** - Unity-Python backend communication bridge via WebSockets
- **OcclusionController.cs** - Mixed reality occlusion management system
- **IndividualOcclusionManager.cs** - Individual object occlusion handling
- **OcclusionState.cs** - Occlusion state definitions and management
- **DepthToMeshConverter.cs** - Depth data to 3D mesh conversion utilities

#### Environment & World Generation
- **CaveChunk.cs** - Procedural cave chunk generation and management
- **CaveWallGenerator.cs** - Dynamic cave wall creation and texturing
- **GuardianBoundaryManager.cs** - Mixed reality boundary and safety system management
- **ApplyWallTexture.cs** - Dynamic texture application to cave walls

#### Puzzle Systems
- **PuzzleManager.cs** - Central puzzle coordination and progression tracking
- **InteractiveElement.cs** - Base class for interactive puzzle elements

#### Utilities
- **ObjPlacer.cs** - Object placement and positioning utilities
- **ScreenshotSaver.cs** - Screenshot capture and saving functionality
- **SpawnBall.cs** - Ball spawning and physics interaction utilities
- **Timer.cs** - Game timing and countdown management

### 🐍 Python Backends (python_backends/)
#### Core Systems
- **massive_physics_engine.py** - Real-time physics simulation and collision detection
- **massive_vision_engine.py** - Computer vision and image processing algorithms
- **massive_narrative_engine.py** - Dynamic storytelling and narrative generation
- **distributed_world_generator.py** - Procedural world generation and distribution
- **performance_monitor_integration.py** - Unity-Python performance monitoring bridge

#### Interview Systems (interview_systems/)
- **master_interview_demo.py** - Main interview system demonstration
- **dynamic_cave_lighting.py** - Adaptive lighting system for cave environments
- **dynamic_cave_texture_generator.py** - Procedural texture generation for caves
- **procedural_puzzle_placement.py** - Algorithmic puzzle placement and optimization

#### Setup & Configuration
- **setup_massive_systems.py** - Automated setup and configuration for all backend systems
- **requirements.txt** - Python dependencies and package requirements
- **SETUP_GUIDE.md** - Comprehensive setup instructions

### 🎨 Assets
#### Materials
- **Cave Material.mat** - Cave environment material properties
- **New Material.mat** - Generic material template
- **Bounce.physicMaterial** - Physics material for bouncing objects

#### Prefabs
- **CaveMesh.prefab** - Procedurally generated cave mesh template
- **PlaneMeshCustom.prefab** - Custom plane mesh for level design
- **Ball.prefab** - Interactive ball object with physics
- **OcclusionToggler.prefab** - Occlusion system toggle controls

#### Shaders
- **CaveWallShader.shader** - Custom shader for cave wall rendering

### 📚 Documentation
- **README.md** - Project overview and getting started guide
- **Final Project Report.pdf** - Comprehensive project documentation and results
- **Mixed Reality Presentation.pdf** - Project presentation materials
- **LICENSE** - Project licensing information

## 🔧 System Architecture

### Unity Frontend
- **Performance Monitoring**: Real-time FPS, memory, and system resource tracking
- **Adaptive Quality**: Dynamic quality adjustment based on performance metrics
- **Mixed Reality Integration**: Occlusion management and boundary systems
- **Procedural Generation**: Dynamic cave and puzzle generation

### Python Backend
- **Physics Simulation**: High-performance physics calculations
- **Computer Vision**: Image processing and analysis
- **Narrative Generation**: Dynamic storytelling algorithms
- **Performance Optimization**: Cross-platform performance monitoring and optimization

### Communication Layer
- **WebSocket Integration**: Real-time bidirectional communication
- **REST API**: HTTP endpoints for data exchange
- **Performance Bridge**: Seamless Unity-Python performance data sharing

## 🚀 Key Features
1. **Real-time Performance Monitoring** across Unity and Python systems
2. **Adaptive Quality Management** with automatic optimization
3. **Procedural Content Generation** for infinite replayability
4. **Mixed Reality Integration** with occlusion and boundary management
5. **Cross-platform Performance Optimization** via Unity-Python bridge
6. **Dynamic Puzzle Generation** with algorithmic placement
7. **Intelligent Narrative Systems** for personalized experiences
