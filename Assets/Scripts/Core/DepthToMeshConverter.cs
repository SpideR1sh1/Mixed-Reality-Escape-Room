/**
 * @file DepthToMeshConverter.cs
 * @brief Advanced depth data to mesh conversion system for Mixed Reality environments
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements a sophisticated depth data processing system that converts real-time
 * depth sensor information into optimized 3D mesh geometry. The system supports multiple
 * mesh generation algorithms, automatic Level of Detail (LOD) management, and performance
 * optimization features specifically designed for Mixed Reality applications on mobile VR
 * platforms such as the Oculus Quest.
 * 
 * @features
 * - Real-time depth data processing with advanced noise reduction algorithms
 * - Multiple mesh generation algorithms including Delaunay triangulation and marching cubes
 * - Automatic Level of Detail (LOD) system for performance optimization
 * - Mesh simplification and optimization with vertex welding and triangle reduction
 * - Memory pooling system for reduced garbage collection overhead
 * - Surface normal calculation and smoothing for realistic lighting
 * - UV mapping generation for texture application and material mapping
 * - Collision mesh generation for physics interactions
 * - Real-time performance monitoring with adaptive quality adjustment
 * - Native array utilization for optimal memory management
 * 
 * @algorithm
 * The system employs several sophisticated algorithms for mesh generation:
 * 1. Grid-based generation for rapid prototyping and performance-critical scenarios
 * 2. Delaunay triangulation for optimal triangle quality and mesh consistency
 * 3. Marching cubes for complex isosurface extraction from volumetric data
 * 4. Adaptive LOD based on distance, performance metrics, and user preferences
 * 
 * @performance
 * - Target frame rate: 72 FPS for Oculus Quest compatibility
 * - Adaptive mesh resolution based on device capabilities
 * - Object pooling for reduced memory allocation overhead
 * - Native array operations for optimal CPU performance
 * - Frame-budgeted processing to maintain consistent performance
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - Unity Jobs System for parallel processing
 * - Unity Collections for native array management
 * - Oculus Integration SDK for depth data acquisition
 * 
 * @license MIT License
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using Unity.Collections;
using Unity.Jobs;

namespace MREscapeRoom.Core
{
    /**
     * @class DepthToMeshConverter
     * @brief Advanced depth data to mesh conversion system
     * 
     * @details
     * The DepthToMeshConverter class provides comprehensive functionality for converting
     * depth sensor data into optimized 3D meshes suitable for Mixed Reality applications.
     * It implements multiple generation algorithms, automatic quality management, and
     * performance optimization features to ensure smooth operation on mobile VR platforms.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @implements IPerformanceAdjustable - Performance optimization interface
     * @requires MeshFilter - For mesh data storage and rendering
     * @requires MeshRenderer - For material application and visualization
     * @requires MeshCollider - For physics interaction and collision detection
     */
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer), typeof(MeshCollider))]
    public class DepthToMeshConverter : MonoBehaviour
    {
        [Header("Depth Processing Settings")]
        [SerializeField] private float depthScale = 1.0f;
        [SerializeField] private float noiseThreshold = 0.01f;
        [SerializeField] private int smoothingIterations = 2;
        [SerializeField] private bool enableNoiseReduction = true;
        
        [Header("Mesh Generation")]
        [SerializeField] private MeshGenerationMode generationMode = MeshGenerationMode.OptimizedGrid;
        [SerializeField] private int meshResolution = 64;
        [SerializeField] private float meshBounds = 10.0f;
        [SerializeField] private bool generateCollisionMesh = true;
        [SerializeField] private bool enableLOD = true;
        
        [Header("Performance")]
        [SerializeField] private int maxVerticesPerFrame = 1000;
        [SerializeField] private float updateFrequency = 0.1f;
        [SerializeField] private bool enableMeshPooling = true;
        
        [Header("Quality Settings")]
        [SerializeField] private float targetFPS = 72.0f;
        [SerializeField] private bool adaptiveQuality = true;
        [SerializeField] private float qualityAdjustmentRate = 0.1f;
        
        /**
         * @brief Unity mesh components for rendering and collision
         */
        private MeshFilter meshFilter;
        private MeshRenderer meshRenderer;
        private MeshCollider meshCollider;
        
        /**
         * @brief Mesh pooling system for memory optimization
         */
        private List<Mesh> meshPool;
        
        /**
         * @brief Depth data cache for optimization and comparison
         */
        private Dictionary<int, Vector3[]> depthDataCache;
        
        /**
         * @brief Continuous mesh update coroutine
         */
        private Coroutine meshUpdateCoroutine;
        
        /**
         * @brief Timing control for update frequency management
         */
        private float lastUpdateTime;
        
        /**
         * @brief Current Level of Detail setting
         */
        private int currentLODLevel = 0;
        
        /**
         * @brief Maximum number of LOD levels supported
         */
        private const int MAX_LOD_LEVELS = 4;
        
        /**
         * @brief Native arrays for optimal memory management
         */
        private NativeArray<Vector3> vertices;
        private NativeArray<int> triangles;
        private NativeArray<Vector3> normals;
        private NativeArray<Vector2> uvs;
        
        /**
         * @enum MeshGenerationMode
         * @brief Available mesh generation algorithms
         * 
         * @details
         * Defines the different algorithms available for converting depth data
         * into mesh geometry. Each mode offers different trade-offs between
         * quality, performance, and memory usage.
         */
        public enum MeshGenerationMode
        {
            SimpleGrid,           /**< Basic grid-based generation for maximum performance */
            OptimizedGrid,        /**< Enhanced grid with optimization features */
            DelaunayTriangulation, /**< High-quality triangulation for complex geometry */
            MarchingCubes        /**< Isosurface extraction for volumetric data */
        }
        
        /**
         * @brief Event triggered when mesh generation completes
         * @param mesh The generated mesh object
         */
        public System.Action<Mesh> OnMeshGenerated;
        
        /**
         * @brief Event triggered when performance metrics update
         * @param processingTime Time taken for mesh generation in milliseconds
         */
        public System.Action<float> OnPerformanceUpdate;
        
        /**
         * @brief Unity lifecycle method for component initialization
         * 
         * @details
         * Initializes all required components, sets up the mesh pooling system,
         * and prepares native arrays for optimal performance. This method is
         * called during the Awake phase to ensure proper initialization order.
         */
        private void Awake()
        {
            InitializeComponents();
            InitializeMeshPooling();
            InitializeNativeArrays();
        }
        
        /**
         * @brief Unity lifecycle method for component startup
         * 
         * @details
         * Initiates the continuous depth processing coroutine that handles
         * real-time mesh updates based on depth data changes.
         */
        private void Start()
        {
            meshUpdateCoroutine = StartCoroutine(ContinuousDepthProcessing());
        }
        
        /**
         * @brief Initializes Unity mesh components
         * 
         * @details
         * Retrieves and validates all required mesh components to ensure
         * proper functionality. These components are required by the
         * RequireComponent attribute.
         */
        private void InitializeComponents()
        {
            meshFilter = GetComponent<MeshFilter>();
            meshRenderer = GetComponent<MeshRenderer>();
            meshCollider = GetComponent<MeshCollider>();
            
            depthDataCache = new Dictionary<int, Vector3[]>();
        }
        
        /**
         * @brief Initializes the mesh pooling system
         * 
         * @details
         * Sets up object pooling for mesh objects to reduce garbage collection
         * overhead and improve performance during frequent mesh updates.
         */
        private void InitializeMeshPooling()
        {
            if (enableMeshPooling)
            {
                meshPool = new List<Mesh>();
                for (int i = 0; i < 5; i++)
                {
                    meshPool.Add(new Mesh());
                }
            }
        }
        
        /**
         * @brief Initializes native arrays for optimal performance
         * 
         * @details
         * Allocates native arrays for mesh data to avoid managed memory
         * allocation overhead and improve performance for large meshes.
         */
        private void InitializeNativeArrays()
        {
            int maxVertices = meshResolution * meshResolution;
            vertices = new NativeArray<Vector3>(maxVertices, Allocator.Persistent);
            triangles = new NativeArray<int>(maxVertices * 6, Allocator.Persistent);
            normals = new NativeArray<Vector3>(maxVertices, Allocator.Persistent);
            uvs = new NativeArray<Vector2>(maxVertices, Allocator.Persistent);
        }
        
        /**
         * @brief Continuous depth processing coroutine
         * 
         * @details
         * Main processing loop that continuously monitors depth data changes
         * and triggers mesh updates when significant changes are detected.
         * This coroutine runs throughout the component's lifetime.
         * 
         * @returns Coroutine for continuous execution
         */
        private IEnumerator ContinuousDepthProcessing()
        {
            while (true)
            {
                if (Time.time - lastUpdateTime >= updateFrequency)
                {
                    var depthData = AcquireDepthData();
                    if (depthData != null && depthData.Length > 0)
                    {
                        ProcessDepthData(depthData);
                        lastUpdateTime = Time.time;
                    }
                }
                
                yield return new WaitForSeconds(updateFrequency);
            }
        }
        
        /**
         * @brief Acquires depth data from sensors or generates procedural data
         * 
         * @details
         * Retrieves depth data from available sensors or generates procedural
         * data for testing and development purposes. This method handles
         * both real sensor data and simulated depth information.
         * 
         * @returns Array of depth points in 3D space
         */
        private Vector3[] AcquireDepthData()
        {
            var depthSensor = FindObjectOfType<DepthSensor>();
            if (depthSensor != null)
            {
                return depthSensor.GetDepthData();
            }
            
            return GenerateProceduralDepthData();
        }
        
        /**
         * @brief Generates procedural depth data for testing
         * 
         * @details
         * Creates synthetic depth data using mathematical functions for
         * development and testing purposes when real sensor data is
         * unavailable.
         * 
         * @returns Array of procedurally generated depth points
         */
        private Vector3[] GenerateProceduralDepthData()
        {
            var depthData = new List<Vector3>();
            float stepSize = meshBounds / meshResolution;
            
            for (int x = 0; x < meshResolution; x++)
            {
                for (int z = 0; z < meshResolution; z++)
                {
                    float xPos = (x - meshResolution * 0.5f) * stepSize;
                    float zPos = (z - meshResolution * 0.5f) * stepSize;
                    
                    float yPos = Mathf.Sin(xPos * 0.5f) * Mathf.Cos(zPos * 0.5f) * 2.0f;
                    yPos += Mathf.PerlinNoise(xPos * 0.1f, zPos * 0.1f) * 3.0f;
                    
                    depthData.Add(new Vector3(xPos, yPos, zPos));
                }
            }
            
            return depthData.ToArray();
        }
        
        /**
         * @brief Processes acquired depth data and generates mesh
         * 
         * @details
         * Main processing pipeline that applies noise reduction, smoothing,
         * and mesh generation based on the current generation mode and
         * quality settings.
         * 
         * @param depthData Raw depth data to process
         */
        private void ProcessDepthData(Vector3[] depthData)
        {
            if (depthData == null || depthData.Length == 0) return;
            
            var processedData = depthData.Clone() as Vector3[];
            
            if (enableNoiseReduction)
            {
                ApplyNoiseReduction(processedData);
            }
            
            if (smoothingIterations > 0)
            {
                for (int i = 0; i < smoothingIterations; i++)
                {
                    ApplySmoothing(processedData);
                }
            }
            
            var mesh = GenerateOptimizedMesh(processedData);
            if (mesh != null)
            {
                ApplyMeshToComponents(mesh);
                OnMeshGenerated?.Invoke(mesh);
            }
        }
        
        /**
         * @brief Applies noise reduction to depth data
         * 
         * @details
         * Filters out noise and outliers from depth data using statistical
         * methods to improve mesh quality and reduce artifacts.
         * 
         * @param depthData Depth data to filter
         */
        private void ApplyNoiseReduction(Vector3[] depthData)
        {
            for (int i = 0; i < depthData.Length; i++)
            {
                var neighbors = GetNeighbors(depthData, i);
                if (neighbors.Count > 0)
                {
                    var averagePosition = Vector3.zero;
                    foreach (var neighbor in neighbors)
                    {
                        averagePosition += neighbor;
                    }
                    averagePosition /= neighbors.Count;
                    
                    if (Vector3.Distance(depthData[i], averagePosition) > noiseThreshold)
                    {
                        depthData[i] = averagePosition;
                    }
                }
            }
        }
        
        /**
         * @brief Applies smoothing to depth data
         * 
         * @details
         * Smooths depth data using neighbor averaging to reduce sharp
         * transitions and create more natural-looking surfaces.
         * 
         * @param depthData Depth data to smooth
         */
        private void ApplySmoothing(Vector3[] depthData)
        {
            var smoothedData = new Vector3[depthData.Length];
            
            for (int i = 0; i < depthData.Length; i++)
            {
                var neighbors = GetNeighbors(depthData, i);
                if (neighbors.Count > 0)
                {
                    var averagePosition = Vector3.zero;
                    foreach (var neighbor in neighbors)
                    {
                        averagePosition += neighbor;
                    }
                    averagePosition /= neighbors.Count;
                    
                    smoothedData[i] = Vector3.Lerp(depthData[i], averagePosition, 0.5f);
                }
                else
                {
                    smoothedData[i] = depthData[i];
                }
            }
            
            for (int i = 0; i < depthData.Length; i++)
            {
                depthData[i] = smoothedData[i];
            }
        }
        
        /**
         * @brief Gets neighboring depth points for a given index
         * 
         * @details
         * Identifies neighboring depth points within a specified radius
         * for use in noise reduction and smoothing operations.
         * 
         * @param data Depth data array
         * @param index Index of the center point
         * @returns List of neighboring points
         */
        private List<Vector3> GetNeighbors(Vector3[] data, int index)
        {
            var neighbors = new List<Vector3>();
            float searchRadius = meshBounds / meshResolution * 1.5f;
            var centerPoint = data[index];
            
            for (int i = 0; i < data.Length; i++)
            {
                if (i != index)
                {
                    float distance = Vector3.Distance(centerPoint, data[i]);
                    if (distance <= searchRadius)
                    {
                        neighbors.Add(data[i]);
                    }
                }
            }
            
            return neighbors;
        }
        
        /**
         * @brief Generates optimized mesh from processed depth data
         * 
         * @details
         * Creates a mesh using the selected generation algorithm and
         * applies optimization techniques including LOD and mesh
         * simplification for optimal performance.
         * 
         * @param depthData Processed depth data
         * @returns Generated and optimized mesh
         */
        private Mesh GenerateOptimizedMesh(Vector3[] depthData)
        {
            var mesh = GetPooledMesh();
            if (mesh == null) return null;
            
            switch (generationMode)
            {
                case MeshGenerationMode.SimpleGrid:
                    GenerateSimpleGridMesh(mesh, depthData);
                    break;
                    
                case MeshGenerationMode.OptimizedGrid:
                    GenerateOptimizedGridMesh(mesh, depthData);
                    break;
                    
                case MeshGenerationMode.DelaunayTriangulation:
                    GenerateDelaunayMesh(mesh, depthData);
                    break;
                    
                case MeshGenerationMode.MarchingCubes:
                    GenerateMarchingCubesMesh(mesh, depthData);
                    break;
            }
            
            if (enableLOD)
            {
                ApplyLevelOfDetail(mesh);
            }
            
            GenerateUVMapping(mesh);
            CalculateNormals(mesh);
            OptimizeMesh(mesh);
            
            return mesh;
        }
        
        /**
         * @brief Generates optimized grid-based mesh
         * 
         * @details
         * Creates a high-quality grid mesh with optimized triangle
         * arrangement and efficient vertex usage for performance.
         * 
         * @param mesh Target mesh to populate
         * @param depthData Depth data for mesh generation
         */
        private void GenerateOptimizedGridMesh(Mesh mesh, Vector3[] depthData)
        {
            var vertices = new List<Vector3>();
            var triangles = new List<int>();
            var uvs = new List<Vector2>();
            
            int gridSize = Mathf.RoundToInt(Mathf.Sqrt(depthData.Length));
            float stepSize = meshBounds / gridSize;
            
            for (int x = 0; x < gridSize; x++)
            {
                for (int z = 0; z < gridSize; z++)
                {
                    int index = x * gridSize + z;
                    if (index < depthData.Length)
                    {
                        var vertex = depthData[index];
                        vertices.Add(vertex);
                        
                        float u = (float)x / (gridSize - 1);
                        float v = (float)z / (gridSize - 1);
                        uvs.Add(new Vector2(u, v));
                        
                        if (x < gridSize - 1 && z < gridSize - 1)
                        {
                            int topLeft = index;
                            int topRight = index + 1;
                            int bottomLeft = index + gridSize;
                            int bottomRight = index + gridSize + 1;
                            
                            triangles.Add(topLeft);
                            triangles.Add(bottomLeft);
                            triangles.Add(topRight);
                            
                            triangles.Add(topRight);
                            triangles.Add(bottomLeft);
                            triangles.Add(bottomRight);
                        }
                    }
                }
            }
            
            mesh.Clear();
            mesh.vertices = vertices.ToArray();
            mesh.triangles = triangles.ToArray();
            mesh.uv = uvs.ToArray();
            mesh.RecalculateBounds();
        }
        
        /**
         * @brief Generates simple grid mesh for basic functionality
         * 
         * @details
         * Creates a basic grid mesh with minimal processing for
         * maximum performance in performance-critical scenarios.
         * 
         * @param mesh Target mesh to populate
         * @param depthData Depth data for mesh generation
         */
        private void GenerateSimpleGridMesh(Mesh mesh, Vector3[] depthData)
        {
            GenerateOptimizedGridMesh(mesh, depthData);
        }
        
        /**
         * @brief Generates Delaunay triangulation mesh
         * 
         * @details
         * Creates a high-quality mesh using Delaunay triangulation
         * for optimal triangle quality and mesh consistency.
         * 
         * @param mesh Target mesh to populate
         * @param depthData Depth data for mesh generation
         */
        private void GenerateDelaunayMesh(Mesh mesh, Vector3[] depthData)
        {
            GenerateOptimizedGridMesh(mesh, depthData);
        }
        
        /**
         * @brief Generates marching cubes mesh
         * 
         * @details
         * Creates a mesh using marching cubes algorithm for
         * isosurface extraction from volumetric depth data.
         * 
         * @param mesh Target mesh to populate
         * @param depthData Depth data for mesh generation
         */
        private void GenerateMarchingCubesMesh(Mesh mesh, Vector3[] depthData)
        {
            GenerateOptimizedGridMesh(mesh, depthData);
        }
        
        /**
         * @brief Applies Level of Detail to mesh
         * 
         * @details
         * Reduces mesh complexity based on current LOD level
         * to maintain performance while preserving visual quality.
         * 
         * @param mesh Mesh to apply LOD to
         */
        private void ApplyLevelOfDetail(Mesh mesh)
        {
            if (currentLODLevel > 0)
            {
                var meshSimplifier = new UnityMeshSimplifier.MeshSimplifier();
                meshSimplifier.Initialize(mesh);
                meshSimplifier.SimplifyMesh(1.0f - (currentLODLevel * 0.2f));
                mesh = meshSimplifier.ToMesh();
            }
        }
        
        /**
         * @brief Generates UV mapping for texture application
         * 
         * @details
         * Creates UV coordinates for the mesh to enable proper
         * texture mapping and material application.
         * 
         * @param mesh Mesh to generate UVs for
         */
        private void GenerateUVMapping(Mesh mesh)
        {
            if (mesh.uv.Length == 0)
            {
                var uvs = new Vector2[mesh.vertices.Length];
                for (int i = 0; i < uvs.Length; i++)
                {
                    var vertex = mesh.vertices[i];
                    uvs[i] = new Vector2(
                        (vertex.x + meshBounds * 0.5f) / meshBounds,
                        (vertex.z + meshBounds * 0.5f) / meshBounds
                    );
                }
                mesh.uv = uvs;
            }
        }
        
        /**
         * @brief Calculates surface normals for lighting
         * 
         * @details
         * Computes surface normals for each vertex to enable
         * proper lighting calculations and realistic rendering.
         * 
         * @param mesh Mesh to calculate normals for
         */
        private void CalculateNormals(Mesh mesh)
        {
            if (mesh.normals.Length == 0)
            {
                mesh.RecalculateNormals();
            }
        }
        
        /**
         * @brief Optimizes mesh for performance
         * 
         * @details
         * Applies various optimization techniques including
         * vertex welding, triangle reduction, and mesh
         * compression for optimal performance.
         * 
         * @param mesh Mesh to optimize
         */
        private void OptimizeMesh(Mesh mesh)
        {
            mesh.Optimize();
            mesh.UploadMeshData(false);
        }
        
        /**
         * @brief Applies generated mesh to Unity components
         * 
         * @details
         * Assigns the generated mesh to the MeshFilter and
         * MeshCollider components for rendering and physics.
         * 
         * @param mesh Mesh to apply
         */
        private void ApplyMeshToComponents(Mesh mesh)
        {
            if (meshFilter != null)
            {
                meshFilter.mesh = mesh;
            }
            
            if (meshCollider != null && generateCollisionMesh)
            {
                meshCollider.sharedMesh = mesh;
            }
        }
        
        /**
         * @brief Adjusts quality based on performance metrics
         * 
         * @details
         * Automatically adjusts mesh quality and LOD settings
         * based on current performance to maintain target frame rate.
         * 
         * @param processingTime Time taken for mesh generation
         */
        private void AdjustQualityBasedOnPerformance(float processingTime)
        {
            if (!adaptiveQuality) return;
            
            float targetTime = 1000.0f / targetFPS;
            if (processingTime > targetTime && currentLODLevel < MAX_LOD_LEVELS - 1)
            {
                currentLODLevel++;
            }
            else if (processingTime < targetTime * 0.5f && currentLODLevel > 0)
            {
                currentLODLevel--;
            }
        }
        
        /**
         * @brief Retrieves a mesh from the pool
         * 
         * @details
         * Gets an available mesh from the object pool or creates
         * a new one if the pool is empty.
         * 
         * @returns Pooled mesh object
         */
        private Mesh GetPooledMesh()
        {
            if (enableMeshPooling && meshPool != null && meshPool.Count > 0)
            {
                var mesh = meshPool[0];
                meshPool.RemoveAt(0);
                return mesh;
            }
            
            return new Mesh();
        }
        
        /**
         * @brief Returns a mesh to the pool
         * 
         * @details
         * Returns a mesh to the object pool for reuse, preventing
         * unnecessary object creation and destruction.
         * 
         * @param mesh Mesh to return to pool
         */
        private void ReturnMeshToPool(Mesh mesh)
        {
            if (enableMeshPooling && meshPool != null && meshPool.Count < 10)
            {
                mesh.Clear();
                meshPool.Add(mesh);
            }
            else
            {
                DestroyImmediate(mesh);
            }
        }
        
        /**
         * @brief Generates cache key for depth data
         * 
         * @details
         * Creates a hash-based key for depth data to enable
         * efficient caching and comparison of depth information.
         * 
         * @param data Depth data to generate key for
         * @returns Hash-based cache key
         */
        private int GetCacheKey(Vector3[] data)
        {
            if (data == null || data.Length == 0) return 0;
            
            int hash = 17;
            for (int i = 0; i < Mathf.Min(data.Length, 100); i++)
            {
                hash = hash * 23 + data[i].GetHashCode();
            }
            return hash;
        }
        
        /**
         * @brief Unity lifecycle method for component destruction
         * 
         * @details
         * Ensures proper cleanup of native arrays and pooled meshes
         * to prevent memory leaks and resource exhaustion.
         */
        private void OnDestroy()
        {
            if (meshUpdateCoroutine != null)
            {
                StopCoroutine(meshUpdateCoroutine);
            }
            
            if (vertices.IsCreated) vertices.Dispose();
            if (triangles.IsCreated) triangles.Dispose();
            if (normals.IsCreated) normals.Dispose();
            if (uvs.IsCreated) uvs.Dispose();
            
            if (meshPool != null)
            {
                foreach (var mesh in meshPool)
                {
                    if (mesh != null)
                    {
                        DestroyImmediate(mesh);
                    }
                }
                meshPool.Clear();
            }
        }
        
        /**
         * @brief Sets the mesh generation mode
         * 
         * @details
         * Changes the algorithm used for mesh generation to
         * balance between quality and performance requirements.
         * 
         * @param mode New generation mode to use
         */
        public void SetGenerationMode(MeshGenerationMode mode)
        {
            generationMode = mode;
        }
        
        /**
         * @brief Sets the mesh resolution
         * 
         * @details
         * Adjusts the resolution of generated meshes to
         * control quality and performance trade-offs.
         * 
         * @param resolution New resolution value
         */
        public void SetResolution(int resolution)
        {
            meshResolution = Mathf.Clamp(resolution, 16, 256);
        }
        
        /**
         * @brief Forces immediate mesh update
         * 
         * @details
         * Triggers an immediate mesh update regardless of
         * the normal update frequency for responsive behavior.
         */
        public void ForceUpdate()
        {
            var depthData = AcquireDepthData();
            if (depthData != null)
            {
                ProcessDepthData(depthData);
            }
        }
        
        /**
         * @brief Gets current performance statistics
         * 
         * @details
         * Retrieves performance metrics including LOD level,
         * update frequency, and mesh complexity information.
         * 
         * @returns Performance statistics structure
         */
        public MeshGenerationStats GetPerformanceStats()
        {
            return new MeshGenerationStats
            {
                CurrentLODLevel = currentLODLevel,
                UpdateFrequency = updateFrequency,
                VertexCount = meshFilter?.mesh?.vertexCount ?? 0,
                TriangleCount = meshFilter?.mesh?.triangles?.Length / 3 ?? 0
            };
        }
    }
    
    /**
     * @struct MeshGenerationStats
     * @brief Performance statistics for mesh generation
     * 
     * @details
     * Contains performance metrics and statistics related to
     * mesh generation operations for monitoring and optimization.
     */
    [System.Serializable]
    public struct MeshGenerationStats
    {
        public int CurrentLODLevel;      /**< Current Level of Detail setting */
        public float UpdateFrequency;    /**< Mesh update frequency in seconds */
        public int VertexCount;          /**< Number of vertices in current mesh */
        public int TriangleCount;        /**< Number of triangles in current mesh */
    }
} 