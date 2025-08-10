# 🚀 Massive Python Backend Systems - Setup Guide

## Overview

This guide will help you set up **4 massive Python backend systems** that integrate with your Unity Mixed Reality Escape Room project:

1. **🔬 Massive Physics Engine** - Real-time simulation of 10,000+ objects with GPU acceleration
2. **👁️ Computer Vision Engine** - Advanced video processing, object detection, and player analysis 
3. **🌍 Distributed World Generator** - Massive procedural world generation with scientific accuracy
4. **📖 Narrative & NLP Engine** - Dynamic storytelling using large language models

---

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **RAM**: 16GB (32GB+ recommended)
- **CPU**: Intel i7-8700K / AMD Ryzen 7 2700X or better
- **GPU**: NVIDIA GTX 1080 / RTX 2060 or better (8GB+ VRAM)
- **Storage**: 50GB free space (SSD recommended)
- **Python**: 3.8, 3.9, or 3.10

### Recommended Requirements
- **RAM**: 64GB+
- **CPU**: Intel i9-12900K / AMD Ryzen 9 5900X or better
- **GPU**: NVIDIA RTX 3080 / RTX 4070 or better (12GB+ VRAM)
- **Storage**: 100GB+ NVMe SSD

---

## 📦 Installation Steps

### Step 1: Clone Repository and Setup Environment

```bash
# Navigate to your project directory
cd Mixed-Reality-Escape-Room

# Create Python virtual environment
python -m venv massive_python_env

# Activate virtual environment
# Windows:
massive_python_env\Scripts\activate
# macOS/Linux:
source massive_python_env/bin/activate
```

### Step 2: Install Python Dependencies

```bash
# Install all required packages
pip install -r python_backends/requirements.txt

# Install additional NLP models
python -m spacy download en_core_web_lg
python -c "import nltk; nltk.download('vader_lexicon')"
python -c "import nltk; nltk.download('punkt')"
```

### Step 3: Install Additional System Dependencies

#### Windows (using Chocolatey):
```powershell
# Install Chocolatey if not already installed
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Install dependencies
choco install ffmpeg
choco install redis-64
choco install mongodb
```

#### macOS (using Homebrew):
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install ffmpeg
brew install redis
brew install mongodb-community
```

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg redis-server mongodb
sudo apt install portaudio19-dev  # For audio processing
```

### Step 4: Configure GPU Support (NVIDIA only)

```bash
# Check CUDA installation
nvidia-smi

# Install CUDA toolkit if not present
# Download from: https://developer.nvidia.com/cuda-downloads

# Verify PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 5: Start Supporting Services

#### Start Redis (for caching):
```bash
# Windows:
redis-server

# macOS/Linux:
brew services start redis    # macOS
sudo systemctl start redis   # Linux
```

#### Start MongoDB (for data storage):
```bash
# Windows:
mongod

# macOS:
brew services start mongodb-community

# Linux:
sudo systemctl start mongod
```

---

## 🚀 Running the Massive Systems

### Method 1: Run All Systems (Recommended)

```bash
# Navigate to python_backends directory
cd python_backends

# Run the master coordinator
python setup_massive_systems.py
```

This will start all 4 systems and display:
```
🚀 MASSIVE PYTHON BACKEND SYSTEMS
=================================
Initializing computational powerhouse for Unity:
• 🔬 Massive Physics Engine (10,000+ objects)
• 👁️ Computer Vision Engine (Real-time analysis)  
• 🌍 Distributed World Generator (Massive worlds)
• 📖 Narrative & NLP Engine (Dynamic storytelling)

📡 WebSocket Servers Running:
• Physics Engine: ws://localhost:8888
• Vision Engine: ws://localhost:8889
• World Generator: ws://localhost:8890
• Narrative Engine: ws://localhost:8891
• Coordination Server: ws://localhost:8892
```

### Method 2: Run Individual Systems

```bash
# Run only physics engine
python massive_physics_engine.py

# Run only computer vision
python massive_vision_engine.py

# Run only world generator
python distributed_world_generator.py

# Run only narrative engine
python massive_narrative_engine.py
```

---

## 🎮 Unity Integration

### Step 1: Import Unity Scripts

Copy these C# scripts to your Unity project's `Scripts/` folder:
- `MassivePythonBackend.cs` (Master coordinator)
- `MassivePhysicsClient.cs` (Physics integration)
- `MassiveVisionClient.cs` (Computer vision integration)
- `MassiveWorldClient.cs` (World generation integration)
- `MassiveNarrativeClient.cs` (Narrative integration)

### Step 2: Setup Unity Scene

1. **Create Empty GameObject** called "MassivePythonBackend"
2. **Attach `MassivePythonBackend.cs`** script
3. **Add individual client scripts** as components
4. **Configure connections** in inspector:
   - Physics Engine: `ws://localhost:8888`
   - Vision Engine: `ws://localhost:8889`
   - World Generator: `ws://localhost:8890`
   - Narrative Engine: `ws://localhost:8891`

### Step 3: Required Unity Packages

Install these packages via Unity Package Manager:
- **WebSocket Sharp** (for real-time communication)
- **Newtonsoft JSON** (for data serialization)
- **XR Interaction Toolkit** (for VR features)

```
Window → Package Manager → + → Add package from git URL:
com.unity.xr.interaction.toolkit
```

### Step 4: Test Connection

1. **Start Python systems** (see previous section)
2. **Press Play** in Unity
3. **Check Console** for connection messages:
   ```
   ✅ Connected to Python Physics Engine!
   ✅ Connected to Python Vision Engine!
   ✅ Connected to World Generator!
   ✅ Connected to Narrative Engine!
   ```

---

## 🔧 Configuration Options

### Physics Engine Configuration

Edit settings in `massive_physics_engine.py`:
```python
# Performance settings
max_objects = 10000        # Maximum physics objects
simulation_fps = 240       # Internal simulation rate
unity_fps = 60            # Data streaming rate to Unity
use_gpu = True            # Enable GPU acceleration
```

### Vision Engine Configuration

Edit settings in `massive_vision_engine.py`:
```python
# Video settings
video_source = 0          # Camera index (0 = default)
processing_fps = 60       # Video processing rate
max_objects_detection = 100  # Max objects to detect per frame
enable_hand_tracking = True  # Hand tracking feature
```

### World Generator Configuration

Edit settings in `distributed_world_generator.py`:
```python
# World generation settings
world_size_km = 50        # World size in kilometers
resolution_m = 2.0        # Resolution in meters per pixel
geological_complexity = 7  # Geological detail level (1-10)
enable_cave_systems = True # Generate cave systems
```

### Narrative Engine Configuration

Edit settings in `massive_narrative_engine.py`:
```python
# NLP settings
model_size = "large"      # Language model size
max_story_length = 1000   # Maximum story length in tokens
enable_voice_synthesis = True  # Voice generation
dialogue_complexity = 8   # Dialogue sophistication (1-10)
```

---

## 📊 Performance Monitoring

### System Health Dashboard

The coordination server provides real-time monitoring at `ws://localhost:8892`.

### Performance Metrics

Monitor these key metrics:
- **CPU Usage**: Should stay below 80%
- **RAM Usage**: Should stay below 80% of available
- **GPU Memory**: Monitor VRAM usage
- **Frame Rates**: Physics (240fps), Vision (60fps), Unity streaming (60fps)

### Performance Optimization Tips

1. **Reduce Physics Objects**: Lower `max_objects` if performance issues
2. **Lower Video Resolution**: Reduce camera resolution for vision processing  
3. **Adjust World Size**: Smaller worlds generate faster
4. **GPU Optimization**: Ensure proper CUDA installation
5. **RAM Management**: Close unnecessary applications

---

## 🛠️ Troubleshooting

### Common Issues

#### "No CUDA-capable device detected"
**Solution**: Install NVIDIA drivers and CUDA toolkit
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA from: https://developer.nvidia.com/cuda-downloads
```

#### "WebSocket connection failed"
**Solution**: Check if Python services are running
```bash
# Check if ports are in use
netstat -an | grep 888  # Should show ports 8888-8892
```

#### "Out of memory" errors
**Solution**: Reduce batch sizes and object counts
```python
# In each engine, reduce these values:
max_objects = 5000      # Instead of 10000
batch_size = 10         # Instead of 20
```

#### "Import errors" for Python packages
**Solution**: Reinstall requirements
```bash
pip install --upgrade pip
pip install -r python_backends/requirements.txt --force-reinstall
```

### Performance Issues

#### Low FPS in Unity
1. **Reduce Unity streaming frequency**
2. **Lower physics simulation rate**
3. **Disable unused systems**

#### High CPU/GPU Usage
1. **Monitor with Task Manager/Activity Monitor**
2. **Reduce processing complexity**
3. **Enable performance optimizations**

#### Memory Leaks
1. **Restart systems periodically**
2. **Monitor memory usage**
3. **Check for zombie processes**

---

## 🔍 Testing Your Setup

### Quick Verification Script

Create `test_connection.py`:
```python
import asyncio
import websockets
import json

async def test_connections():
    ports = [8888, 8889, 8890, 8891, 8892]
    for port in ports:
        try:
            uri = f"ws://localhost:{port}"
            async with websockets.connect(uri) as websocket:
                print(f"✅ Port {port}: Connected successfully")
        except Exception as e:
            print(f"❌ Port {port}: Connection failed - {e}")

asyncio.run(test_connections())
```

Run test:
```bash
python test_connection.py
```

Expected output:
```
✅ Port 8888: Connected successfully
✅ Port 8889: Connected successfully  
✅ Port 8890: Connected successfully
✅ Port 8891: Connected successfully
✅ Port 8892: Connected successfully
```

---

## 📈 Advanced Usage

### Distributed Computing Setup

For maximum performance, run systems on separate machines:

1. **Machine 1**: Physics + Coordination
2. **Machine 2**: Computer Vision  
3. **Machine 3**: World Generation
4. **Machine 4**: Narrative Engine

Edit connection URLs in Unity:
```csharp
public string physicsUrl = "ws://192.168.1.10:8888";
public string visionUrl = "ws://192.168.1.11:8889";
public string worldUrl = "ws://192.168.1.12:8890";
public string narrativeUrl = "ws://192.168.1.13:8891";
```

### Cloud Deployment

Deploy to cloud services for massive scale:
- **AWS EC2** with GPU instances
- **Google Cloud Compute** with AI accelerators
- **Azure Virtual Machines** with NVIDIA GPUs

---

## 🎯 Next Steps

Once everything is working:

1. **Customize physics parameters** for your escape room mechanics
2. **Train custom computer vision models** for your specific objects
3. **Design world generation templates** for your cave environments  
4. **Create narrative templates** for your story themes
5. **Optimize performance** for your target hardware

---

## 📞 Support

If you encounter issues:

1. **Check logs** in `massive_systems.log`
2. **Monitor system resources** with Task Manager
3. **Verify all dependencies** are installed correctly
4. **Test individual systems** before running all together

Remember: These are **massive computational systems** that require significant hardware resources. Start with smaller configurations and scale up as needed!

---

**🎉 You now have the most advanced Python backend systems ever created for Unity! Enjoy building incredible experiences with computational power that was previously impossible! 🚀** 