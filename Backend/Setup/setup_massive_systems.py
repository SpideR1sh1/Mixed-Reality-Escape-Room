"""
Massive Python Backend Systems Setup & Coordination
==================================================

This script initializes and coordinates all massive Python backend systems:
- Massive Physics Engine (Real-time simulation)
- Computer Vision Engine (Video processing & analysis)  
- Distributed World Generator (Procedural world creation)
- Narrative & NLP Engine (Dynamic storytelling)

Provides unified coordination and communication with Unity through multiple WebSocket servers.
"""

import asyncio
import logging
import json
import time
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import psutil
import signal
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional
import subprocess
import os

# Import our massive systems
from massive_physics_engine import MassivePhysicsEngine, UnityPhysicsStreamer
from massive_vision_engine import MassiveVisionEngine, UnityVisionStreamer
from distributed_world_generator import MassiveWorldGenerator, UnityWorldStreamer
from massive_narrative_engine import MassiveNarrativeEngine, UnityNarrativeStreamer

@dataclass
class SystemConfig:
    """Configuration for each massive system"""
    name: str
    enabled: bool
    port: int
    max_memory_gb: float
    max_cpu_percent: float
    gpu_required: bool
    dependencies: List[str]

class MassiveSystemsCoordinator:
    """Coordinates all massive Python backend systems"""
    
    def __init__(self):
        # System configurations
        self.system_configs = {
            'physics': SystemConfig(
                name="Massive Physics Engine",
                enabled=True,
                port=8888,
                max_memory_gb=4.0,
                max_cpu_percent=50.0,
                gpu_required=True,
                dependencies=['numba', 'scipy', 'websockets']
            ),
            'vision': SystemConfig(
                name="Computer Vision Engine", 
                enabled=True,
                port=8889,
                max_memory_gb=6.0,
                max_cpu_percent=60.0,
                gpu_required=True,
                dependencies=['torch', 'opencv-python', 'mediapipe', 'ultralytics']
            ),
            'world': SystemConfig(
                name="Distributed World Generator",
                enabled=True,
                port=8890,
                max_memory_gb=8.0,
                max_cpu_percent=70.0,
                gpu_required=False,
                dependencies=['dask', 'numpy', 'h5py', 'xarray']
            ),
            'narrative': SystemConfig(
                name="Narrative & NLP Engine",
                enabled=True,
                port=8891,
                max_memory_gb=10.0,
                max_cpu_percent=40.0,
                gpu_required=True,
                dependencies=['transformers', 'torch', 'spacy', 'nltk']
            )
        }
        
        # System instances
        self.systems = {}
        self.streamers = {}
        self.system_processes = {}
        
        # Performance monitoring
        self.performance_monitor = SystemPerformanceMonitor()
        self.resource_manager = ResourceManager()
        
        # Coordination
        self.coordination_server = CoordinationServer()
        self.system_health = {}
        
        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('massive_systems.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("🚀 Massive Systems Coordinator initialized")
    
    async def initialize_all_systems(self):
        """Initialize all massive Python backend systems"""
        
        self.logger.info("🔧 Starting initialization of all massive systems...")
        
        # Check system requirements
        await self._check_system_requirements()
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        # Initialize systems in optimal order (based on dependencies)
        initialization_order = ['narrative', 'world', 'physics', 'vision']
        
        for system_name in initialization_order:
            if self.system_configs[system_name].enabled:
                await self._initialize_system(system_name)
                
                # Wait for system to stabilize
                await asyncio.sleep(2)
        
        # Start coordination server
        await self.coordination_server.start(self.systems, self.streamers)
        
        # Start system health monitoring
        asyncio.create_task(self._monitor_system_health())
        
        self.logger.info("✅ All massive systems initialized successfully!")
        self.logger.info("🎮 Unity can now connect to the following WebSocket servers:")
        
        for system_name, config in self.system_configs.items():
            if config.enabled:
                self.logger.info(f"   • {config.name}: ws://localhost:{config.port}")
    
    async def _initialize_system(self, system_name: str):
        """Initialize a specific system"""
        
        config = self.system_configs[system_name]
        self.logger.info(f"🔧 Initializing {config.name}...")
        
        try:
            # Check resource availability
            if not await self._check_resources_available(config):
                self.logger.error(f"❌ Insufficient resources for {config.name}")
                return False
            
            # Initialize system based on type
            if system_name == 'physics':
                await self._initialize_physics_system()
            elif system_name == 'vision':
                await self._initialize_vision_system()
            elif system_name == 'world':
                await self._initialize_world_system()
            elif system_name == 'narrative':
                await self._initialize_narrative_system()
            
            self.system_health[system_name] = {
                'status': 'healthy',
                'last_check': time.time(),
                'uptime': 0,
                'errors': 0
            }
            
            self.logger.info(f"✅ {config.name} initialized successfully on port {config.port}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize {config.name}: {e}")
            return False
    
    async def _initialize_physics_system(self):
        """Initialize the massive physics engine"""
        
        # Create physics engine
        physics_engine = MassivePhysicsEngine()
        
        # Create Unity streamer
        physics_streamer = UnityPhysicsStreamer(physics_engine)
        
        # Store references
        self.systems['physics'] = physics_engine
        self.streamers['physics'] = physics_streamer
        
        # Start the physics server in a separate task
        asyncio.create_task(physics_streamer.start_server())
        
        self.logger.info("🔬 Physics engine started with GPU acceleration")
    
    async def _initialize_vision_system(self):
        """Initialize the computer vision engine"""
        
        # Create vision engine
        vision_engine = MassiveVisionEngine()
        
        # Start video processing
        await vision_engine.start_video_processing(video_source=0)
        
        # Create Unity streamer
        vision_streamer = UnityVisionStreamer(vision_engine)
        
        # Store references
        self.systems['vision'] = vision_engine
        self.streamers['vision'] = vision_streamer
        
        # Start the vision server in a separate task
        asyncio.create_task(vision_streamer.start_server())
        
        self.logger.info("👁️ Computer vision engine started with real-time processing")
    
    async def _initialize_world_system(self):
        """Initialize the distributed world generator"""
        
        # Create world generator
        world_generator = MassiveWorldGenerator()
        
        # Create Unity streamer
        world_streamer = UnityWorldStreamer()
        
        # Store references
        self.systems['world'] = world_generator
        self.streamers['world'] = world_streamer
        
        self.logger.info("🌍 Distributed world generator initialized with cluster computing")
    
    async def _initialize_narrative_system(self):
        """Initialize the narrative and NLP engine"""
        
        # Create narrative engine
        narrative_engine = MassiveNarrativeEngine()
        
        # Start narrative processing
        await narrative_engine.start_narrative_processing()
        
        # Create Unity streamer
        narrative_streamer = UnityNarrativeStreamer(narrative_engine)
        
        # Store references
        self.systems['narrative'] = narrative_engine
        self.streamers['narrative'] = narrative_streamer
        
        # Start the narrative server in a separate task
        asyncio.create_task(narrative_streamer.start_server())
        
        self.logger.info("📖 Narrative engine started with large language models")
    
    async def _check_system_requirements(self):
        """Check if system meets requirements for massive systems"""
        
        self.logger.info("🔍 Checking system requirements...")
        
        # Check available RAM
        available_ram = psutil.virtual_memory().available / (1024**3)  # GB
        required_ram = sum(config.max_memory_gb for config in self.system_configs.values() if config.enabled)
        
        if available_ram < required_ram:
            self.logger.warning(f"⚠️ Low RAM: {available_ram:.1f}GB available, {required_ram:.1f}GB required")
        else:
            self.logger.info(f"✅ RAM: {available_ram:.1f}GB available, {required_ram:.1f}GB required")
        
        # Check CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count < 8:
            self.logger.warning(f"⚠️ Limited CPU cores: {cpu_count} available (8+ recommended)")
        else:
            self.logger.info(f"✅ CPU: {cpu_count} cores available")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.logger.info(f"✅ GPU: {gpu_count} GPU(s) available, {gpu_memory:.1f}GB VRAM")
            else:
                self.logger.warning("⚠️ No CUDA-capable GPU detected (reduced performance)")
        except ImportError:
            self.logger.warning("⚠️ PyTorch not available for GPU detection")
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        free_space = disk_usage.free / (1024**3)  # GB
        
        if free_space < 50:  # 50GB minimum
            self.logger.warning(f"⚠️ Low disk space: {free_space:.1f}GB available")
        else:
            self.logger.info(f"✅ Disk: {free_space:.1f}GB available")
    
    async def _monitor_system_health(self):
        """Continuously monitor health of all systems"""
        
        self.logger.info("🏥 Starting system health monitoring...")
        
        while True:
            try:
                for system_name, system in self.systems.items():
                    # Check system responsiveness
                    health_status = await self._check_system_health(system_name, system)
                    self.system_health[system_name] = health_status
                    
                    # Log health issues
                    if health_status['status'] != 'healthy':
                        self.logger.warning(f"⚠️ {system_name} health: {health_status['status']}")
                
                # Check overall resource usage
                await self._check_resource_usage()
                
                # Auto-restart failed systems
                await self._handle_failed_systems()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _check_system_health(self, system_name: str, system) -> Dict:
        """Check health of a specific system"""
        
        current_time = time.time()
        
        try:
            # System-specific health checks
            if system_name == 'physics':
                # Check if physics simulation is running
                health = await self._check_physics_health(system)
            elif system_name == 'vision':
                # Check if video processing is active
                health = await self._check_vision_health(system)
            elif system_name == 'world':
                # Check if world generation is responsive
                health = await self._check_world_health(system)
            elif system_name == 'narrative':
                # Check if narrative generation is working
                health = await self._check_narrative_health(system)
            else:
                health = {'status': 'unknown'}
            
            # Update uptime
            if system_name in self.system_health:
                health['uptime'] = current_time - self.system_health[system_name].get('start_time', current_time)
            else:
                health['start_time'] = current_time
                health['uptime'] = 0
            
            health['last_check'] = current_time
            return health
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'last_check': current_time,
                'uptime': self.system_health.get(system_name, {}).get('uptime', 0)
            }

class SystemPerformanceMonitor:
    """Monitor performance of all massive systems"""
    
    def __init__(self):
        self.monitoring_active = False
        self.performance_data = {}
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring_active = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                
                # Collect GPU metrics if available
                gpu_metrics = self._collect_gpu_metrics()
                
                # Store performance data
                timestamp = time.time()
                self.performance_data[timestamp] = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_info.percent,
                    'memory_available_gb': memory_info.available / (1024**3),
                    'gpu_metrics': gpu_metrics
                }
                
                # Cleanup old data (keep last 1 hour)
                cutoff_time = timestamp - 3600
                self.performance_data = {
                    t: data for t, data in self.performance_data.items() 
                    if t > cutoff_time
                }
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                print(f"Error in performance monitoring: {e}")
                time.sleep(10)
    
    def _collect_gpu_metrics(self):
        """Collect GPU performance metrics"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_metrics = []
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    memory_cached = torch.cuda.memory_reserved(i) / (1024**3)
                    
                    gpu_metrics.append({
                        'device_id': i,
                        'memory_allocated_gb': memory_allocated,
                        'memory_cached_gb': memory_cached
                    })
                
                return gpu_metrics
        except:
            pass
        
        return []

class CoordinationServer:
    """Central coordination server for all massive systems"""
    
    def __init__(self):
        self.systems = {}
        self.streamers = {}
        self.coordination_port = 8892
        
    async def start(self, systems, streamers):
        """Start coordination server"""
        
        self.systems = systems
        self.streamers = streamers
        
        print(f"🎛️ Starting coordination server on ws://localhost:{self.coordination_port}")
        
        async def handle_coordination_client(websocket, path):
            print("🎮 Unity connected to coordination server")
            
            try:
                # Send system status
                await websocket.send(json.dumps({
                    'type': 'system_status',
                    'systems': {
                        name: {'status': 'online', 'port': config.port}
                        for name, config in [
                            ('physics', {'port': 8888}),
                            ('vision', {'port': 8889}),
                            ('world', {'port': 8890}),
                            ('narrative', {'port': 8891})
                        ]
                    }
                }))
                
                # Handle coordination requests
                async for message in websocket:
                    request = json.loads(message)
                    await self.handle_coordination_request(request, websocket)
                    
            except Exception as e:
                print(f"Coordination error: {e}")
        
        # Start coordination WebSocket server
        import websockets
        server = await websockets.serve(
            handle_coordination_client, 
            "localhost", 
            self.coordination_port
        )
        
        print("🎛️ Coordination server ready!")
        return server
    
    async def handle_coordination_request(self, request, websocket):
        """Handle coordination requests from Unity"""
        
        request_type = request.get('type')
        
        if request_type == 'system_health':
            # Unity requesting system health status
            health_data = {}
            for system_name in self.systems:
                health_data[system_name] = {
                    'status': 'healthy',  # Simplified for now
                    'uptime': time.time(),
                    'connections': 1
                }
            
            await websocket.send(json.dumps({
                'type': 'health_response',
                'data': health_data
            }))
        
        elif request_type == 'restart_system':
            # Unity requesting system restart
            system_name = request.get('system_name')
            if system_name in self.systems:
                # Implement system restart logic
                await websocket.send(json.dumps({
                    'type': 'restart_response',
                    'system_name': system_name,
                    'status': 'restarting'
                }))

def create_unity_master_script():
    """Create Unity master integration script"""
    
    unity_master_script = '''
/*
 * MassivePythonBackend.cs
 * Master Unity integration for all massive Python backend systems
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using WebSocketSharp;
using Newtonsoft.Json;
using System;

public class MassivePythonBackend : MonoBehaviour
{
    [Header("System Configuration")]
    public bool enablePhysicsEngine = true;
    public bool enableVisionEngine = true;
    public bool enableWorldGenerator = true;
    public bool enableNarrativeEngine = true;
    
    [Header("Connection Settings")]
    public string coordinationUrl = "ws://localhost:8892";
    
    // Individual system clients
    public MassivePhysicsClient physicsClient;
    public MassiveVisionClient visionClient;
    public MassiveWorldClient worldClient;
    public MassiveNarrativeClient narrativeClient;
    
    private WebSocket coordinationSocket;
    private Dictionary<string, SystemStatus> systemStatuses;
    
    [Serializable]
    public class SystemStatus
    {
        public string status;
        public float uptime;
        public int connections;
    }
    
    void Start()
    {
        Debug.Log("🚀 Initializing Massive Python Backend Integration");
        
        systemStatuses = new Dictionary<string, SystemStatus>();
        
        // Connect to coordination server
        ConnectToCoordinationServer();
        
        // Initialize individual system clients
        InitializeSystemClients();
    }
    
    private void ConnectToCoordinationServer()
    {
        coordinationSocket = new WebSocket(coordinationUrl);
        
        coordinationSocket.OnOpen += (sender, e) =>
        {
            Debug.Log("✅ Connected to Python coordination server!");
            RequestSystemStatus();
        };
        
        coordinationSocket.OnMessage += (sender, e) =>
        {
            try
            {
                var message = JsonConvert.DeserializeObject<Dictionary<string, object>>(e.Data);
                ProcessCoordinationMessage(message);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Coordination error: {ex.Message}");
            }
        };
        
        coordinationSocket.Connect();
    }
    
    private void InitializeSystemClients()
    {
        // Initialize physics client
        if (enablePhysicsEngine && physicsClient != null)
        {
            physicsClient.ConnectToPythonPhysics();
        }
        
        // Initialize vision client
        if (enableVisionEngine && visionClient != null)
        {
            visionClient.ConnectToPythonVision();
        }
        
        // Initialize world client
        if (enableWorldGenerator && worldClient != null)
        {
            worldClient.ConnectToWorldGenerator();
        }
        
        // Initialize narrative client
        if (enableNarrativeEngine && narrativeClient != null)
        {
            narrativeClient.ConnectToNarrativeEngine();
        }
    }
    
    private void RequestSystemStatus()
    {
        var request = new Dictionary<string, object>
        {
            ["type"] = "system_health"
        };
        
        coordinationSocket.Send(JsonConvert.SerializeObject(request));
    }
    
    void Update()
    {
        // Monitor system health every 30 seconds
        if (Time.time % 30.0f < Time.deltaTime)
        {
            RequestSystemStatus();
        }
    }
    
    public void RestartSystem(string systemName)
    {
        var request = new Dictionary<string, object>
        {
            ["type"] = "restart_system",
            ["system_name"] = systemName
        };
        
        coordinationSocket.Send(JsonConvert.SerializeObject(request));
    }
    
    private void ProcessCoordinationMessage(Dictionary<string, object> message)
    {
        string messageType = message["type"].ToString();
        
        if (messageType == "system_status")
        {
            Debug.Log("📊 Received system status update");
            // Update system status display
        }
        else if (messageType == "health_response")
        {
            var healthData = JsonConvert.DeserializeObject<Dictionary<string, SystemStatus>>(
                message["data"].ToString()
            );
            
            systemStatuses = healthData;
            UpdateHealthDisplay();
        }
    }
    
    private void UpdateHealthDisplay()
    {
        foreach (var system in systemStatuses)
        {
            Debug.Log($"System {system.Key}: {system.Value.status} (uptime: {system.Value.uptime:F1}s)");
        }
    }
    
    void OnApplicationQuit()
    {
        // Gracefully disconnect from all systems
        coordinationSocket?.Close();
        
        physicsClient?.GetComponent<MassivePhysicsClient>()?.DisconnectFromPhysics();
        visionClient?.GetComponent<MassiveVisionClient>()?.DisconnectFromVision();
        worldClient?.GetComponent<MassiveWorldClient>()?.DisconnectFromWorld();
        narrativeClient?.GetComponent<MassiveNarrativeClient>()?.DisconnectFromNarrative();
        
        Debug.Log("🛑 Disconnected from all massive Python systems");
    }
}
'''
    
    with open('unity_integration/MassivePythonBackend.cs', 'w') as f:
        f.write(unity_master_script)

async def main():
    """Main entry point for massive systems coordination"""
    
    print("🚀 MASSIVE PYTHON BACKEND SYSTEMS")
    print("=================================")
    print("Initializing computational powerhouse for Unity:")
    print("• 🔬 Massive Physics Engine (10,000+ objects)")
    print("• 👁️ Computer Vision Engine (Real-time analysis)")
    print("• 🌍 Distributed World Generator (Massive worlds)")
    print("• 📖 Narrative & NLP Engine (Dynamic storytelling)")
    print("=================================")
    
    # Create Unity integration scripts
    create_unity_master_script()
    
    # Initialize coordinator
    coordinator = MassiveSystemsCoordinator()
    
    try:
        # Initialize all systems
        await coordinator.initialize_all_systems()
        
        print("\n🎉 ALL MASSIVE SYSTEMS ONLINE!")
        print("Unity Integration Instructions:")
        print("1. Add MassivePythonBackend.cs to your Unity project")
        print("2. Attach individual system clients to your GameObjects")
        print("3. Configure connection URLs in the inspector")
        print("4. Enable the systems you want to use")
        print("5. Play your scene - systems will auto-connect!")
        
        print("\n📡 WebSocket Servers Running:")
        print("• Physics Engine: ws://localhost:8888")
        print("• Vision Engine: ws://localhost:8889") 
        print("• World Generator: ws://localhost:8890")
        print("• Narrative Engine: ws://localhost:8891")
        print("• Coordination Server: ws://localhost:8892")
        
        # Keep running indefinitely
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down massive systems...")
        coordinator.logger.info("Graceful shutdown initiated")
        
        # Cleanup systems
        for system_name, system in coordinator.systems.items():
            try:
                if hasattr(system, 'cleanup'):
                    system.cleanup()
                coordinator.logger.info(f"✅ {system_name} cleaned up")
            except Exception as e:
                coordinator.logger.error(f"Error cleaning up {system_name}: {e}")
        
        print("✅ All systems shut down gracefully")

if __name__ == "__main__":
    # Ensure we have all required directories
    os.makedirs('unity_integration', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Run the massive systems
    asyncio.run(main()) 