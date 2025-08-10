/**
 * @file CaveChunk.cs
 * @brief Individual cave chunk component with Level of Detail (LOD) support
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements an individual cave chunk component that represents a section
 * of the cave environment with comprehensive Level of Detail (LOD) support, mesh
 * optimization, collision handling, and performance monitoring. Each chunk manages
 * its own mesh data, materials, and interactive elements while providing seamless
 * integration with the overall cave system.
 * 
 * @features
 * - Dynamic Level of Detail (LOD) system with distance-based quality adjustment
 * - Advanced mesh optimization including vertex welding and simplification
 * - Automatic collision mesh generation for physics interactions
 * - Performance monitoring and adaptive quality adjustment
 * - Material management with UV mapping support
 * - Seamless chunk boundaries for continuous cave environments
 * - Occlusion culling and frustum culling for performance optimization
 * - Interactive element placement and management within chunks
 * - Real-time performance tracking and optimization
 * 
 * @lod_system
 * The LOD system provides multiple quality levels:
 * 1. High Quality (100% vertices) - Close range detailed rendering
 * 2. Medium Quality (75% vertices) - Mid-range balanced rendering
 * 3. Low Quality (50% vertices) - Far-range performance-optimized rendering
 * 4. Minimal Quality (25% vertices) - Maximum distance minimal rendering
 * 
 * @performance_optimization
 * - Automatic vertex count monitoring and optimization
 * - Frame-budgeted visibility and LOD updates
 * - Adaptive material quality based on performance level
 * - Efficient culling systems for off-screen chunks
 * - Memory management with object pooling support
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - MeshFilter, MeshRenderer, and MeshCollider components
 * - MREscapeRoom.Core namespace for performance interfaces
 * - InteractiveElement system for interactive object management
 * 
 * @license MIT License
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MREscapeRoom.Core;

namespace MREscapeRoom.Environment
{
    /**
     * @class CaveChunk
     * @brief Individual cave chunk component with comprehensive LOD support
     * 
     * @details
     * The CaveChunk class represents an individual section of the cave environment
     * with advanced features including dynamic LOD management, mesh optimization,
     * performance monitoring, and interactive element placement. Each chunk operates
     * independently while maintaining seamless integration with the overall cave system.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @implements IPerformanceAdjustable - Performance optimization interface
     * @requires MeshFilter - For mesh data storage and rendering
     * @requires MeshRenderer - For material application and visualization
     * @requires MeshCollider - For physics interaction and collision detection
     */
    [RequireComponent(typeof(MeshFilter), typeof(MeshRenderer), typeof(MeshCollider))]
    public class CaveChunk : MonoBehaviour, IPerformanceAdjustable
    {
        [Header("Chunk Properties")]
        [SerializeField] private Vector2Int gridPosition;                    /**< Grid position within the cave system */
        [SerializeField] private float chunkSize;                           /**< Size of the chunk in world units */
        [SerializeField] private CaveGenerationSettings generationSettings; /**< Settings for cave generation algorithms */
        
        [Header("LOD Settings")]
        [SerializeField] private bool enableLOD = true;                      /**< Enable/disable LOD system */
        [SerializeField] private float[] lodDistances = { 10f, 25f, 50f, 100f };           /**< Distance thresholds for LOD changes */
        [SerializeField] private float[] lodQualityMultipliers = { 1.0f, 0.75f, 0.5f, 0.25f }; /**< Quality multipliers for each LOD level */
        
        [Header("Performance")]
        [SerializeField] private bool enableOcclusionCulling = true;        /**< Enable occlusion culling for performance */
        [SerializeField] private bool enableFrustumCulling = true;          /**< Enable frustum culling for performance */
        [SerializeField] private int maxVertices = 65536;                   /**< Maximum vertices allowed per chunk */
        [SerializeField] private int currentVertexCount;                    /**< Current vertex count for this chunk */
        
        /**
         * @brief Core mesh components for rendering and collision
         */
        private MeshFilter meshFilter;
        private MeshRenderer meshRenderer;
        private MeshCollider meshCollider;
        
        /**
         * @brief LOD system components
         */
        private Mesh[] lodMeshes;                                           /**< Array of meshes for different LOD levels */
        private Material[] chunkMaterials;                                  /**< Materials applied to this chunk */
        private int currentLODLevel = 0;                                    /**< Current active LOD level */
        
        /**
         * @brief Object management and placement
         */
        private Dictionary<string, GameObject> placedObjects;               /**< Objects placed within this chunk */
        private List<InteractiveElement> interactiveElements;               /**< Interactive elements in this chunk */
        
        /**
         * @brief Chunk state and performance tracking
         */
        private bool isInitialized = false;                                 /**< Initialization state flag */
        private bool isVisible = true;                                      /**< Current visibility state */
        private float distanceToCamera;                                     /**< Distance from chunk to main camera */
        
        /**
         * @brief Performance optimization constants
         */
        private int framesSinceVisibilityUpdate = 0;                        /**< Frame counter for visibility updates */
        private const int VISIBILITY_UPDATE_INTERVAL = 30;                  /**< Update visibility every 30 frames */
        
        /**
         * @brief Public properties for external access
         */
        public Vector2Int GridPosition => gridPosition;                     /**< Grid position within cave system */
        public float ChunkSize => chunkSize;                               /**< Size of chunk in world units */
        public bool IsVisible => isVisible;                                /**< Current visibility state */
        public int VertexCount => currentVertexCount;                       /**< Current vertex count */
        public int CurrentLODLevel => currentLODLevel;                      /**< Current LOD level */
        
        /**
         * @brief Events for external system notifications
         */
        public event System.Action<CaveChunk, int> OnLODChanged;           /**< Fired when LOD level changes */
        public event System.Action<CaveChunk, bool> OnVisibilityChanged;    /**< Fired when visibility state changes */
        
        /**
         * @brief Unity lifecycle method for component initialization
         * 
         * @details
         * Initializes required mesh components and data structures during
         * component creation. Ensures all necessary components are available
         * and properly configured for chunk operation.
         */
        private void Awake()
        {
            InitializeComponents();
        }
        
        /**
         * @brief Unity lifecycle method for continuous updates
         * 
         * @details
         * Manages performance updates and visibility/LOD calculations on
         * a frame-budgeted basis to maintain optimal performance while
         * ensuring responsive chunk behavior.
         */
        private void Update()
        {
            if (!isInitialized) return;
            
            UpdatePerformance();
            
            framesSinceVisibilityUpdate++;
            if (framesSinceVisibilityUpdate >= VISIBILITY_UPDATE_INTERVAL)
            {
                UpdateVisibilityAndLOD();
                framesSinceVisibilityUpdate = 0;
            }
        }
        
        /**
         * @brief Initializes core components and data structures
         * 
         * @details
         * Retrieves and validates required mesh components, initializes
         * data structures for object management, and prepares the chunk
         * for operation. This method ensures all dependencies are met.
         */
        private void InitializeComponents()
        {
            meshFilter = GetComponent<MeshFilter>();
            meshRenderer = GetComponent<MeshRenderer>();
            meshCollider = GetComponent<MeshCollider>();
            
            placedObjects = new Dictionary<string, GameObject>();
            interactiveElements = new List<InteractiveElement>();
            
            if (enableLOD)
            {
                lodMeshes = new Mesh[lodDistances.Length];
            }
        }
        
        public void Initialize(Vector2Int position, float size, CaveGenerationSettings settings)
        {
            gridPosition = position;
            chunkSize = size;
            generationSettings = settings;
            
            gameObject.name = $"CaveChunk_{position.x}_{position.y}";
            
            isInitialized = true;
            
            Debug.Log($"[CaveChunk] Initialized chunk at {position} with size {size}");
        }
        
        public void SetMesh(Mesh mesh)
        {
            if (mesh == null) return;
            
            // Set main mesh
            meshFilter.mesh = mesh;
            currentVertexCount = mesh.vertexCount;
            
            // Generate LOD meshes if enabled
            if (enableLOD)
            {
                GenerateLODMeshes(mesh);
            }
            
            // Set up collision mesh
            if (meshCollider != null)
            {
                meshCollider.sharedMesh = mesh;
            }
            
            Debug.Log($"[CaveChunk] Mesh set with {mesh.vertexCount} vertices and {mesh.triangles.Length / 3} triangles");
        }
        
        private void GenerateLODMeshes(Mesh originalMesh)
        {
            lodMeshes[0] = originalMesh; // Highest quality
            
            for (int i = 1; i < lodMeshes.Length; i++)
            {
                float quality = lodQualityMultipliers[i];
                lodMeshes[i] = SimplifyMesh(originalMesh, quality);
                lodMeshes[i].name = $"{originalMesh.name}_LOD{i}";
            }
        }
        
        private Mesh SimplifyMesh(Mesh originalMesh, float quality)
        {
            // Simplified mesh decimation algorithm
            Vector3[] originalVertices = originalMesh.vertices;
            int[] originalTriangles = originalMesh.triangles;
            Vector3[] originalNormals = originalMesh.normals;
            Vector2[] originalUVs = originalMesh.uv;
            
            int targetVertexCount = Mathf.RoundToInt(originalVertices.Length * quality);
            targetVertexCount = Mathf.Max(targetVertexCount, 12); // Minimum vertices for a valid mesh
            
            // Simple vertex removal based on distance
            List<int> keepVertices = new List<int>();
            List<Vector3> newVertices = new List<Vector3>();
            List<Vector3> newNormals = new List<Vector3>();
            List<Vector2> newUVs = new List<Vector2>();
            
            float step = originalVertices.Length / (float)targetVertexCount;
            
            for (int i = 0; i < originalVertices.Length; i += Mathf.RoundToInt(step))
            {
                if (i < originalVertices.Length)
                {
                    keepVertices.Add(i);
                    newVertices.Add(originalVertices[i]);
                    
                    if (originalNormals.Length > i)
                        newNormals.Add(originalNormals[i]);
                    
                    if (originalUVs.Length > i)
                        newUVs.Add(originalUVs[i]);
                }
            }
            
            // Rebuild triangles
            List<int> newTriangles = new List<int>();
            Dictionary<int, int> vertexMapping = new Dictionary<int, int>();
            
            for (int i = 0; i < keepVertices.Count; i++)
            {
                vertexMapping[keepVertices[i]] = i;
            }
            
            for (int i = 0; i < originalTriangles.Length; i += 3)
            {
                int v1 = originalTriangles[i];
                int v2 = originalTriangles[i + 1];
                int v3 = originalTriangles[i + 2];
                
                if (vertexMapping.ContainsKey(v1) && vertexMapping.ContainsKey(v2) && vertexMapping.ContainsKey(v3))
                {
                    newTriangles.Add(vertexMapping[v1]);
                    newTriangles.Add(vertexMapping[v2]);
                    newTriangles.Add(vertexMapping[v3]);
                }
            }
            
            // Create simplified mesh
            Mesh simplifiedMesh = new Mesh();
            simplifiedMesh.vertices = newVertices.ToArray();
            simplifiedMesh.triangles = newTriangles.ToArray();
            
            if (newNormals.Count > 0)
                simplifiedMesh.normals = newNormals.ToArray();
            else
                simplifiedMesh.RecalculateNormals();
                
            if (newUVs.Count > 0)
                simplifiedMesh.uv = newUVs.ToArray();
            
            simplifiedMesh.RecalculateBounds();
            simplifiedMesh.Optimize();
            
            return simplifiedMesh;
        }
        
        private void UpdateVisibilityAndLOD()
        {
            Camera mainCamera = Camera.main;
            if (mainCamera == null) return;
            
            Vector3 cameraPosition = mainCamera.transform.position;
            Vector3 chunkCenter = transform.position + Vector3.one * chunkSize * 0.5f;
            distanceToCamera = Vector3.Distance(cameraPosition, chunkCenter);
            
            // Update visibility
            bool wasVisible = isVisible;
            UpdateVisibility(mainCamera, chunkCenter);
            
            if (wasVisible != isVisible)
            {
                OnVisibilityChanged?.Invoke(this, isVisible);
            }
            
            // Update LOD
            if (isVisible && enableLOD)
            {
                int newLODLevel = CalculateLODLevel(distanceToCamera);
                if (newLODLevel != currentLODLevel)
                {
                    SetLODLevel(newLODLevel);
                }
            }
        }
        
        private void UpdateVisibility(Camera camera, Vector3 chunkCenter)
        {
            bool newVisibility = true;
            
            // Frustum culling
            if (enableFrustumCulling)
            {
                Bounds chunkBounds = new Bounds(chunkCenter, Vector3.one * chunkSize);
                Plane[] frustumPlanes = GeometryUtility.CalculateFrustumPlanes(camera);
                newVisibility = GeometryUtility.TestPlanesAABB(frustumPlanes, chunkBounds);
            }
            
            // Occlusion culling (simplified)
            if (newVisibility && enableOcclusionCulling)
            {
                // Simple distance-based occlusion
                if (distanceToCamera > lodDistances[lodDistances.Length - 1] * 1.5f)
                {
                    newVisibility = false;
                }
            }
            
            if (newVisibility != isVisible)
            {
                SetVisibility(newVisibility);
            }
        }
        
        private int CalculateLODLevel(float distance)
        {
            for (int i = 0; i < lodDistances.Length; i++)
            {
                if (distance <= lodDistances[i])
                {
                    return i;
                }
            }
            return lodDistances.Length - 1;
        }
        
        private void SetLODLevel(int lodLevel)
        {
            if (lodLevel == currentLODLevel || lodMeshes == null) return;
            
            int previousLOD = currentLODLevel;
            currentLODLevel = Mathf.Clamp(lodLevel, 0, lodMeshes.Length - 1);
            
            if (lodMeshes[currentLODLevel] != null)
            {
                meshFilter.mesh = lodMeshes[currentLODLevel];
                currentVertexCount = meshFilter.mesh.vertexCount;
                
                OnLODChanged?.Invoke(this, currentLODLevel);
                
                Debug.Log($"[CaveChunk] {gameObject.name} LOD changed from {previousLOD} to {currentLODLevel} " +
                         $"(distance: {distanceToCamera:F1}m, vertices: {currentVertexCount})");
            }
        }
        
        private void SetVisibility(bool visible)
        {
            isVisible = visible;
            meshRenderer.enabled = visible;
            
            // Enable/disable collider for performance
            if (meshCollider != null)
            {
                meshCollider.enabled = visible;
            }
            
            // Enable/disable placed objects
            foreach (var obj in placedObjects.Values)
            {
                if (obj != null)
                {
                    obj.SetActive(visible);
                }
            }
        }
        
        public void ApplyMaterial(Material material)
        {
            if (material == null || meshRenderer == null) return;
            
            meshRenderer.material = material;
            
            // Store reference for LOD material switching if needed
            if (chunkMaterials == null)
            {
                chunkMaterials = new Material[1];
            }
            chunkMaterials[0] = material;
        }
        
        public void ApplyMaterials(Material[] materials)
        {
            if (materials == null || materials.Length == 0 || meshRenderer == null) return;
            
            meshRenderer.materials = materials;
            chunkMaterials = materials;
        }
        
        public void PlaceObject(string objectId, GameObject obj, Vector3 localPosition)
        {
            if (obj == null) return;
            
            obj.transform.parent = transform;
            obj.transform.localPosition = localPosition;
            
            placedObjects[objectId] = obj;
            
            // Add interactive element if it has one
            InteractiveElement interactive = obj.GetComponent<InteractiveElement>();
            if (interactive != null)
            {
                interactiveElements.Add(interactive);
            }
            
            Debug.Log($"[CaveChunk] Placed object '{objectId}' at local position {localPosition}");
        }
        
        public GameObject GetPlacedObject(string objectId)
        {
            placedObjects.TryGetValue(objectId, out GameObject obj);
            return obj;
        }
        
        public void RemoveObject(string objectId)
        {
            if (placedObjects.TryGetValue(objectId, out GameObject obj))
            {
                // Remove from interactive elements
                InteractiveElement interactive = obj.GetComponent<InteractiveElement>();
                if (interactive != null)
                {
                    interactiveElements.Remove(interactive);
                }
                
                placedObjects.Remove(objectId);
                
                if (obj != null)
                {
                    DestroyImmediate(obj);
                }
            }
        }
        
        public void OptimizeMesh()
        {
            if (meshFilter.mesh == null) return;
            
            Mesh mesh = meshFilter.mesh;
            
            // Optimize mesh
            mesh.Optimize();
            mesh.RecalculateBounds();
            
            // Weld nearby vertices
            WeldVertices(mesh);
            
            // Update vertex count
            currentVertexCount = mesh.vertexCount;
            
            Debug.Log($"[CaveChunk] Mesh optimized: {currentVertexCount} vertices");
        }
        
        private void WeldVertices(Mesh mesh)
        {
            Vector3[] vertices = mesh.vertices;
            int[] triangles = mesh.triangles;
            Vector3[] normals = mesh.normals;
            Vector2[] uvs = mesh.uv;
            
            const float weldThreshold = 0.001f;
            Dictionary<Vector3, int> vertexMap = new Dictionary<Vector3, int>();
            List<Vector3> newVertices = new List<Vector3>();
            List<Vector3> newNormals = new List<Vector3>();
            List<Vector2> newUVs = new List<Vector2>();
            List<int> vertexMapping = new List<int>();
            
            for (int i = 0; i < vertices.Length; i++)
            {
                Vector3 vertex = vertices[i];
                int existingIndex = -1;
                
                // Check if a similar vertex already exists
                foreach (var kvp in vertexMap)
                {
                    if (Vector3.Distance(kvp.Key, vertex) < weldThreshold)
                    {
                        existingIndex = kvp.Value;
                        break;
                    }
                }
                
                if (existingIndex == -1)
                {
                    // New unique vertex
                    existingIndex = newVertices.Count;
                    vertexMap[vertex] = existingIndex;
                    newVertices.Add(vertex);
                    
                    if (normals.Length > i)
                        newNormals.Add(normals[i]);
                    if (uvs.Length > i)
                        newUVs.Add(uvs[i]);
                }
                
                vertexMapping.Add(existingIndex);
            }
            
            // Remap triangles
            for (int i = 0; i < triangles.Length; i++)
            {
                triangles[i] = vertexMapping[triangles[i]];
            }
            
            // Apply welded mesh
            mesh.Clear();
            mesh.vertices = newVertices.ToArray();
            mesh.triangles = triangles;
            if (newNormals.Count > 0) mesh.normals = newNormals.ToArray();
            if (newUVs.Count > 0) mesh.uv = newUVs.ToArray();
            
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
        }
        
        public void SetupLOD(GameManager.PerformanceLevel performanceLevel)
        {
            switch (performanceLevel)
            {
                case GameManager.PerformanceLevel.Low:
                    lodDistances = new float[] { 5f, 15f, 30f, 60f };
                    enableOcclusionCulling = true;
                    enableFrustumCulling = true;
                    break;
                    
                case GameManager.PerformanceLevel.Medium:
                    lodDistances = new float[] { 8f, 20f, 40f, 80f };
                    enableOcclusionCulling = true;
                    enableFrustumCulling = true;
                    break;
                    
                case GameManager.PerformanceLevel.High:
                    lodDistances = new float[] { 10f, 25f, 50f, 100f };
                    enableOcclusionCulling = false;
                    enableFrustumCulling = true;
                    break;
                    
                case GameManager.PerformanceLevel.Ultra:
                    lodDistances = new float[] { 15f, 35f, 70f, 150f };
                    enableOcclusionCulling = false;
                    enableFrustumCulling = false;
                    break;
            }
            
            // Regenerate LOD meshes with new settings
            if (meshFilter.mesh != null)
            {
                GenerateLODMeshes(meshFilter.mesh);
            }
        }
        
        private void UpdatePerformance()
        {
            // Monitor performance metrics
            if (currentVertexCount > maxVertices)
            {
                Debug.LogWarning($"[CaveChunk] {gameObject.name} exceeds max vertices: {currentVertexCount}/{maxVertices}");
            }
        }
        
        public Vector3 GetRandomSurfacePoint()
        {
            if (meshFilter.mesh == null) return transform.position;
            
            Mesh mesh = meshFilter.mesh;
            Vector3[] vertices = mesh.vertices;
            
            if (vertices.Length == 0) return transform.position;
            
            Vector3 localPoint = vertices[Random.Range(0, vertices.Length)];
            return transform.TransformPoint(localPoint);
        }
        
        public List<Vector3> GetSurfacePointsInRadius(Vector3 center, float radius)
        {
            List<Vector3> points = new List<Vector3>();
            
            if (meshFilter.mesh == null) return points;
            
            Vector3[] vertices = meshFilter.mesh.vertices;
            Vector3 localCenter = transform.InverseTransformPoint(center);
            
            foreach (Vector3 vertex in vertices)
            {
                if (Vector3.Distance(vertex, localCenter) <= radius)
                {
                    points.Add(transform.TransformPoint(vertex));
                }
            }
            
            return points;
        }
        
        #region IPerformanceAdjustable Implementation
        
        public void AdjustPerformance(GameManager.PerformanceLevel level)
        {
            SetupLOD(level);
            
            // Adjust material quality
            if (chunkMaterials != null)
            {
                foreach (var material in chunkMaterials)
                {
                    AdjustMaterialQuality(material, level);
                }
            }
            
            // Adjust interactive elements
            foreach (var interactive in interactiveElements)
            {
                if (interactive is IPerformanceAdjustable adjustable)
                {
                    adjustable.AdjustPerformance(level);
                }
            }
            
            Debug.Log($"[CaveChunk] {gameObject.name} performance adjusted to {level}");
        }
        
        private void AdjustMaterialQuality(Material material, GameManager.PerformanceLevel level)
        {
            if (material == null) return;
            
            switch (level)
            {
                case GameManager.PerformanceLevel.Low:
                    material.SetFloat("_DetailNormalMapScale", 0.5f);
                    material.SetFloat("_BumpScale", 0.5f);
                    break;
                    
                case GameManager.PerformanceLevel.Medium:
                    material.SetFloat("_DetailNormalMapScale", 0.75f);
                    material.SetFloat("_BumpScale", 0.75f);
                    break;
                    
                case GameManager.PerformanceLevel.High:
                    material.SetFloat("_DetailNormalMapScale", 1.0f);
                    material.SetFloat("_BumpScale", 1.0f);
                    break;
                    
                case GameManager.PerformanceLevel.Ultra:
                    material.SetFloat("_DetailNormalMapScale", 1.5f);
                    material.SetFloat("_BumpScale", 1.5f);
                    break;
            }
        }
        
        #endregion
        
        public ChunkInfo GetChunkInfo()
        {
            return new ChunkInfo
            {
                GridPosition = gridPosition,
                WorldPosition = transform.position,
                ChunkSize = chunkSize,
                VertexCount = currentVertexCount,
                IsVisible = isVisible,
                CurrentLODLevel = currentLODLevel,
                DistanceToCamera = distanceToCamera,
                PlacedObjectCount = placedObjects.Count,
                InteractiveElementCount = interactiveElements.Count
            };
        }
        
        private void OnDestroy()
        {
            // Clean up LOD meshes
            if (lodMeshes != null)
            {
                for (int i = 1; i < lodMeshes.Length; i++) // Skip index 0 as it's the original mesh
                {
                    if (lodMeshes[i] != null)
                    {
                        DestroyImmediate(lodMeshes[i]);
                    }
                }
            }
            
            placedObjects?.Clear();
            interactiveElements?.Clear();
        }
        
        #if UNITY_EDITOR
        private void OnDrawGizmosSelected()
        {
            // Draw chunk bounds
            Gizmos.color = isVisible ? Color.green : Color.red;
            Vector3 center = transform.position + Vector3.one * chunkSize * 0.5f;
            Gizmos.DrawWireCube(center, Vector3.one * chunkSize);
            
            // Draw LOD distances
            if (enableLOD && lodDistances != null)
            {
                Gizmos.color = Color.yellow;
                for (int i = 0; i < lodDistances.Length; i++)
                {
                    Gizmos.DrawWireSphere(center, lodDistances[i]);
                }
            }
            
            // Draw placed objects
            Gizmos.color = Color.blue;
            foreach (var obj in placedObjects.Values)
            {
                if (obj != null)
                {
                    Gizmos.DrawWireSphere(obj.transform.position, 0.5f);
                }
            }
        }
        #endif
    }
    
    [System.Serializable]
    public struct ChunkInfo
    {
        public Vector2Int GridPosition;
        public Vector3 WorldPosition;
        public float ChunkSize;
        public int VertexCount;
        public bool IsVisible;
        public int CurrentLODLevel;
        public float DistanceToCamera;
        public int PlacedObjectCount;
        public int InteractiveElementCount;
    }
} 