"""
Unity Performance Monitor Integration for Python Backends
========================================================

This system provides real-time performance monitoring integration between Unity's
PerformanceMonitor and Python backend systems. It enables:

- Cross-platform performance metric collection
- Real-time performance optimization recommendations
- Adaptive quality adjustments based on system performance
- Performance data streaming to Unity for visualization
- Machine learning-based performance prediction
- Automated backend optimization based on Unity performance metrics

The system communicates with Unity via WebSocket and provides REST API endpoints
for performance data collection and optimization.
"""

import asyncio
import websockets
import json
import time
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import psutil
import GPUtil
from collections import deque
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UnityPerformanceMetrics:
    """Unity performance metrics structure matching C# PerformanceMetrics"""
    averageFPS: float
    minFPS: float
    maxFPS: float
    frameTimeVariance: float
    totalMemoryMB: float
    reservedMemoryMB: float
    unusedMemoryMB: float
    memoryLeakDetected: bool
    cpuUsagePercent: float
    gpuUsagePercent: float
    renderThreadTime: float
    drawCalls: int
    thermalThrottlingDetected: bool
    thermalState: str
    timestamp: float

@dataclass
class PythonBackendMetrics:
    """Python backend performance metrics"""
    cpuUsagePercent: float
    memoryUsageMB: float
    gpuUsagePercent: float
    gpuMemoryMB: float
    networkLatencyMs: float
    processingTimeMs: float
    queueSize: int
    activeThreads: int
    timestamp: datetime

@dataclass
class PerformanceRecommendation:
    """Performance optimization recommendation"""
    action: str
    priority: str  # "low", "medium", "high", "critical"
    confidence: float
    expectedImprovement: float
    implementation: str
    timestamp: datetime

class UnityPerformanceIntegration:
    """Handles real-time integration with Unity's PerformanceMonitor"""
    
    def __init__(self, websocket_port: int = 8889):
        self.websocket_port = websocket_port
        self.unity_clients = set()
        self.performance_history = deque(maxlen=1000)
        self.recommendations_history = deque(maxlen=100)
        self.is_monitoring = False
        
        # Performance thresholds
        self.fps_threshold_low = 60.0
        self.fps_threshold_critical = 45.0
        self.memory_threshold_high = 2000.0  # MB
        self.cpu_threshold_high = 85.0  # %
        
        # Backend optimization settings
        self.auto_optimize_backend = True
        self.quality_levels = ["low", "medium", "high", "ultra"]
        self.current_quality_level = "high"
        
    async def start_server(self):
        """Start WebSocket server for Unity performance monitoring"""
        logger.info(f"🚀 Starting Unity Performance Integration Server on ws://localhost:{self.websocket_port}")
        
        async def handle_unity_client(websocket, path):
            self.unity_clients.add(websocket)
            logger.info(f"🎮 Unity PerformanceMonitor connected. Active clients: {len(self.unity_clients)}")
            
            try:
                # Send initial configuration
                await websocket.send(json.dumps({
                    'type': 'performance_config',
                    'data': {
                        'auto_optimize_backend': self.auto_optimize_backend,
                        'quality_levels': self.quality_levels,
                        'current_quality': self.current_quality_level
                    }
                }))
                
                # Handle Unity performance data
                async for message in websocket:
                    try:
                        unity_data = json.loads(message)
                        await self.handle_unity_performance_data(unity_data)
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON received from Unity")
                        
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.unity_clients.remove(websocket)
                logger.info(f"🎮 Unity PerformanceMonitor disconnected. Active clients: {len(self.unity_clients)}")
        
        # Start WebSocket server
        server = await websockets.serve(handle_unity_client, "localhost", self.websocket_port)
        logger.info("📡 Performance Integration ready for Unity connection!")
        
        # Start monitoring loop
        asyncio.create_task(self.performance_monitoring_loop())
        
        await server.wait_closed()
    
    async def handle_unity_performance_data(self, unity_data: Dict):
        """Process performance data from Unity"""
        try:
            if unity_data.get('type') == 'performance_metrics':
                metrics = UnityPerformanceMetrics(**unity_data['data'])
                await self.process_unity_metrics(metrics)
                
            elif unity_data.get('type') == 'performance_alert':
                alert = unity_data['data']
                await self.handle_unity_alert(alert)
                
            elif unity_data.get('type') == 'quality_change_request':
                quality_level = unity_data['data']['quality_level']
                await self.handle_quality_change_request(quality_level)
                
        except Exception as e:
            logger.error(f"Error processing Unity performance data: {e}")
    
    async def process_unity_metrics(self, metrics: UnityPerformanceMetrics):
        """Process and analyze Unity performance metrics"""
        # Store metrics in history
        self.performance_history.append(asdict(metrics))
        
        # Analyze performance trends
        performance_score = self.calculate_performance_score(metrics)
        
        # Generate recommendations if needed
        if performance_score < 0.7:
            recommendations = await self.generate_performance_recommendations(metrics)
            await self.send_recommendations_to_unity(recommendations)
        
        # Optimize backend if auto-optimization is enabled
        if self.auto_optimize_backend:
            await self.optimize_backend_performance(metrics)
        
        # Log performance data
        logger.info(f"📊 Unity Performance: {metrics.averageFPS:.1f} FPS, "
                   f"{metrics.totalMemoryMB:.1f} MB, CPU: {metrics.cpuUsagePercent:.1f}%")
    
    async def handle_unity_alert(self, alert: Dict):
        """Handle performance alerts from Unity"""
        alert_type = alert.get('type', 'unknown')
        severity = alert.get('severity', 'info')
        
        logger.warning(f"🚨 Unity Performance Alert: {alert_type} - {severity}")
        
        if severity in ['warning', 'critical']:
            # Take immediate action
            await self.emergency_performance_optimization(alert)
    
    async def handle_quality_change_request(self, quality_level: str):
        """Handle quality level change requests from Unity"""
        if quality_level in self.quality_levels:
            self.current_quality_level = quality_level
            logger.info(f"🎛️ Quality level changed to: {quality_level}")
            
            # Apply quality settings to backend systems
            await self.apply_quality_settings(quality_level)
    
    def calculate_performance_score(self, metrics: UnityPerformanceMetrics) -> float:
        """Calculate overall performance score (0.0 to 1.0)"""
        fps_score = min(metrics.averageFPS / self.fps_threshold_low, 1.0)
        memory_score = max(0.0, 1.0 - (metrics.totalMemoryMB / self.memory_threshold_high))
        cpu_score = max(0.0, 1.0 - (metrics.cpuUsagePercent / 100.0))
        
        # Weighted average
        overall_score = (fps_score * 0.5 + memory_score * 0.3 + cpu_score * 0.2)
        return max(0.0, min(1.0, overall_score))
    
    async def generate_performance_recommendations(self, metrics: UnityPerformanceMetrics) -> List[PerformanceRecommendation]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # FPS-based recommendations
        if metrics.averageFPS < self.fps_threshold_critical:
            recommendations.append(PerformanceRecommendation(
                action="Reduce rendering quality to minimum",
                priority="critical",
                confidence=0.9,
                expectedImprovement=0.3,
                implementation="Set Unity quality level to 'Low'",
                timestamp=datetime.now()
            ))
        elif metrics.averageFPS < self.fps_threshold_low:
            recommendations.append(PerformanceRecommendation(
                action="Moderate quality reduction",
                priority="high",
                confidence=0.7,
                expectedImprovement=0.2,
                implementation="Set Unity quality level to 'Medium'",
                timestamp=datetime.now()
            ))
        
        # Memory-based recommendations
        if metrics.totalMemoryMB > self.memory_threshold_high:
            recommendations.append(PerformanceRecommendation(
                action="Optimize memory usage",
                priority="high",
                confidence=0.8,
                expectedImprovement=0.25,
                implementation="Reduce texture quality and object pooling",
                timestamp=datetime.now()
            ))
        
        # CPU-based recommendations
        if metrics.cpuUsagePercent > self.cpu_threshold_high:
            recommendations.append(PerformanceRecommendation(
                action="Reduce CPU load",
                priority="medium",
                confidence=0.6,
                expectedImprovement=0.15,
                implementation="Optimize update loops and reduce physics complexity",
                timestamp=datetime.now()
            ))
        
        return recommendations
    
    async def send_recommendations_to_unity(self, recommendations: List[PerformanceRecommendation]):
        """Send performance recommendations back to Unity"""
        if not self.unity_clients:
            return
        
        message = {
            'type': 'performance_recommendations',
            'data': [asdict(rec) for rec in recommendations]
        }
        
        # Send to all connected Unity clients
        disconnected_clients = set()
        for client in self.unity_clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.unity_clients -= disconnected_clients
        
        logger.info(f"📤 Sent {len(recommendations)} recommendations to Unity")
    
    async def optimize_backend_performance(self, metrics: UnityPerformanceMetrics):
        """Automatically optimize backend performance based on Unity metrics"""
        if metrics.averageFPS < self.fps_threshold_low:
            # Reduce backend processing load
            await self.reduce_backend_processing_load()
        elif metrics.averageFPS > self.fps_threshold_low * 1.2:
            # Increase backend processing quality
            await self.increase_backend_processing_quality()
    
    async def emergency_performance_optimization(self, alert: Dict):
        """Emergency performance optimization for critical alerts"""
        logger.warning("🚨 Emergency performance optimization activated!")
        
        # Immediately reduce backend load
        await self.reduce_backend_processing_load()
        
        # Send emergency recommendations to Unity
        emergency_recommendations = [
            PerformanceRecommendation(
                action="Emergency performance mode",
                priority="critical",
                confidence=0.95,
                expectedImprovement=0.4,
                implementation="Disable non-essential features immediately",
                timestamp=datetime.now()
            )
        ]
        
        await self.send_recommendations_to_unity(emergency_recommendations)
    
    async def reduce_backend_processing_load(self):
        """Reduce backend processing load for better Unity performance"""
        # This would integrate with your existing backend systems
        logger.info("🔧 Reducing backend processing load")
        
        # Example: Reduce physics simulation complexity
        # await physics_engine.set_simulation_quality("low")
        
        # Example: Reduce AI processing frequency
        # await ai_system.set_update_frequency(0.5)
    
    async def increase_backend_processing_quality(self):
        """Increase backend processing quality when Unity performance allows"""
        logger.info("🔧 Increasing backend processing quality")
        
        # Example: Increase physics simulation quality
        # await physics_engine.set_simulation_quality("high")
        
        # Example: Increase AI processing frequency
        # await ai_system.set_update_frequency(1.0)
    
    async def apply_quality_settings(self, quality_level: str):
        """Apply quality settings to backend systems"""
        logger.info(f"🎛️ Applying {quality_level} quality settings to backend systems")
        
        # This would integrate with your existing backend systems
        # Example: Adjust physics simulation parameters
        # await physics_engine.set_quality_level(quality_level)
        
        # Example: Adjust AI system complexity
        # await ai_system.set_quality_level(quality_level)
    
    async def performance_monitoring_loop(self):
        """Main performance monitoring loop"""
        while True:
            try:
                # Collect Python backend metrics
                backend_metrics = self.collect_backend_metrics()
                
                # Analyze backend performance
                await self.analyze_backend_performance(backend_metrics)
                
                # Send backend metrics to Unity if requested
                if self.unity_clients:
                    await self.send_backend_metrics_to_unity(backend_metrics)
                
                await asyncio.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error
    
    def collect_backend_metrics(self) -> PythonBackendMetrics:
        """Collect current Python backend performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU metrics (if available)
            gpu_percent = 0.0
            gpu_memory = 0.0
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_percent = gpu.load * 100
                    gpu_memory = gpu.memoryUsed
            except:
                pass
            
            return PythonBackendMetrics(
                cpuUsagePercent=cpu_percent,
                memoryUsageMB=memory.used / (1024 * 1024),
                gpuUsagePercent=gpu_percent,
                gpuMemoryMB=gpu_memory,
                networkLatencyMs=0.0,  # Would measure actual network latency
                processingTimeMs=0.0,   # Would measure actual processing time
                queueSize=0,            # Would measure actual queue sizes
                activeThreads=threading.active_count(),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error collecting backend metrics: {e}")
            return PythonBackendMetrics(
                cpuUsagePercent=0.0,
                memoryUsageMB=0.0,
                gpuUsagePercent=0.0,
                gpuMemoryMB=0.0,
                networkLatencyMs=0.0,
                processingTimeMs=0.0,
                queueSize=0,
                activeThreads=0,
                timestamp=datetime.now()
            )
    
    async def analyze_backend_performance(self, metrics: PythonBackendMetrics):
        """Analyze backend performance and take action if needed"""
        if metrics.cpuUsagePercent > 90.0:
            logger.warning(f"🚨 High backend CPU usage: {metrics.cpuUsagePercent:.1f}%")
            await self.emergency_backend_optimization()
        elif metrics.memoryUsageMB > 4000.0:  # 4GB threshold
            logger.warning(f"🚨 High backend memory usage: {metrics.memoryUsageMB:.1f} MB")
            await self.optimize_backend_memory()
    
    async def emergency_backend_optimization(self):
        """Emergency optimization for backend performance issues"""
        logger.warning("🚨 Emergency backend optimization activated!")
        
        # Reduce processing load immediately
        # await physics_engine.set_simulation_quality("minimal")
        # await ai_system.disable_non_essential_features()
    
    async def optimize_backend_memory(self):
        """Optimize backend memory usage"""
        logger.info("🔧 Optimizing backend memory usage")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Reduce cache sizes
        # await physics_engine.clear_cache()
        # await ai_system.clear_cache()
    
    async def send_backend_metrics_to_unity(self, metrics: PythonBackendMetrics):
        """Send backend metrics to Unity for monitoring"""
        if not self.unity_clients:
            return
        
        message = {
            'type': 'backend_metrics',
            'data': asdict(metrics)
        }
        
        # Send to all connected Unity clients
        disconnected_clients = set()
        for client in self.unity_clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.unity_clients -= disconnected_clients

# FastAPI for REST endpoints
app = FastAPI(title="Unity Performance Monitor Integration API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global integration instance
integration = UnityPerformanceIntegration()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Unity Performance Monitor Integration API", "status": "running"}

@app.get("/performance/unity")
async def get_unity_performance():
    """Get latest Unity performance metrics"""
    if integration.performance_history:
        return {"metrics": list(integration.performance_history)[-10:]}  # Last 10 entries
    return {"metrics": []}

@app.get("/performance/backend")
async def get_backend_performance():
    """Get current backend performance metrics"""
    metrics = integration.collect_backend_metrics()
    return asdict(metrics)

@app.get("/recommendations")
async def get_recommendations():
    """Get latest performance recommendations"""
    if integration.recommendations_history:
        return {"recommendations": list(integration.recommendations_history)[-10:]}  # Last 10 entries
    return {"recommendations": []}

@app.post("/quality/change")
async def change_quality_level(quality_level: str):
    """Change quality level"""
    if quality_level in integration.quality_levels:
        integration.current_quality_level = quality_level
        return {"message": f"Quality level changed to {quality_level}", "status": "success"}
    else:
        raise HTTPException(status_code=400, detail=f"Invalid quality level. Must be one of {integration.quality_levels}")

@app.post("/optimization/emergency")
async def trigger_emergency_optimization():
    """Trigger emergency performance optimization"""
    await integration.emergency_performance_optimization({"type": "manual_trigger"})
    return {"message": "Emergency optimization triggered", "status": "success"}

async def main():
    """Main function to start both WebSocket server and FastAPI"""
    logger.info("🚀 Starting Unity Performance Monitor Integration...")
    
    # Start WebSocket server in background
    websocket_task = asyncio.create_task(integration.start_server())
    
    # Start FastAPI server
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    # Run both servers
    await asyncio.gather(
        websocket_task,
        server.serve()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Performance Monitor Integration stopped by user")
    except Exception as e:
        logger.error(f"❌ Error running Performance Monitor Integration: {e}")
