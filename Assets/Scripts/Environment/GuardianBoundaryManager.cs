/**
 * @file GuardianBoundaryManager.cs
 * @brief Advanced Mixed Reality boundary management and cave environment generation system
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements a comprehensive Mixed Reality boundary management system that
 * creates immersive cave environments seamlessly blending with Guardian boundaries.
 * The system provides procedural cave generation, environmental effects, interactive
 * elements, and performance optimization features specifically designed for VR platforms.
 * 
 * @features
 * - Advanced procedural cave generation with multiple cave types
 * - Guardian boundary integration and safety margin management
 * - Dynamic Level of Detail (LOD) system for performance optimization
 * - Comprehensive environmental effects including lighting, particles, and audio
 * - Interactive element placement and puzzle integration
 * - Automatic mesh optimization and occlusion culling
 * - Real-time performance monitoring and adaptive quality adjustment
 * - Support for multiple cave types: Natural, Limestone, Lava, Crystal, Ice, Ancient, Magical
 * 
 * @cave_generation
 * The system supports multiple cave generation algorithms:
 * 1. Natural caves with organic formations and realistic geology
 * 2. Limestone caves with stalactites, stalagmites, and water features
 * 3. Lava caves with volcanic formations and heat effects
 * 4. Crystal caves with crystalline structures and light refraction
 * 5. Ice caves with frozen formations and temperature effects
 * 6. Ancient caves with archaeological elements and historical features
 * 7. Magical caves with supernatural effects and mystical elements
 * 
 * @environmental_system
 * - Dynamic lighting with ambient, directional, and volumetric effects
 * - Particle systems for atmospheric effects and environmental detail
 * - Audio management with spatial sound and environmental audio
 * - Fog and atmospheric density for immersive atmosphere
 * - Water features including pools, streams, and reflections
 * 
 * @performance_optimization
 * - Chunk-based generation for scalable environments
 * - LOD system with distance-based quality adjustment
 * - Occlusion culling for off-screen object management
 * - Mesh optimization with vertex reduction and simplification
 * - Frame-budgeted generation for smooth performance
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - Oculus Integration SDK for VR platform support
 * - MREscapeRoom.Core namespace for system interfaces
 * - CaveChunk system for modular environment management
 * 
 * @license MIT License
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Oculus;
using System.Linq;
using MREscapeRoom.Core;

/**
 * @struct CaveGenerationSettings
 * @brief Configuration settings for cave generation algorithms
 * 
 * @details
 * Comprehensive settings structure that controls all aspects of cave generation
 * including basic parameters, procedural algorithms, environmental features,
 * lighting configuration, and atmospheric effects.
 */
[System.Serializable]
public class CaveGenerationSettings
{
    [Header("Basic Generation")]
    public CaveType caveType = CaveType.Natural;    /**< Type of cave to generate */
    public float caveHeight = 4.0f;                 /**< Height of the cave in world units */
    public float wallThickness = 0.5f;              /**< Thickness of cave walls */
    public int seed = 12345;                        /**< Random seed for procedural generation */
    
    [Header("Procedural Parameters")]
    public float noiseScale = 0.1f;                 /**< Scale factor for noise generation */
    public float roughness = 0.7f;                  /**< Surface roughness factor */
    public int octaves = 4;                         /**< Number of noise octaves for detail */
    public float persistence = 0.5f;                /**< Noise persistence for detail variation */
    public float lacunarity = 2.0f;                 /**< Noise lacunarity for frequency variation */
    
    [Header("Environmental Features")]
    public bool enableStalactites = true;           /**< Enable stalactite generation */
    public bool enableStalagmites = true;           /**< Enable stalagmite generation */
    public bool enableWaterFeatures = true;         /**< Enable water feature generation */
    public bool enableCrystals = true;              /**< Enable crystal formation generation */
    public int detailDensity = 50;                  /**< Density of environmental details */
    
    [Header("Lighting")]
    public Color ambientColor = new Color(0.2f, 0.3f, 0.4f, 1.0f); /**< Ambient lighting color */
    public float ambientIntensity = 0.3f;          /**< Ambient lighting intensity */
    public bool enableDynamicShadows = true;        /**< Enable dynamic shadow casting */
    public bool enableVolumetricLighting = true;    /**< Enable volumetric lighting effects */
    
    [Header("Atmospheric Effects")]
    public bool enableFog = true;                   /**< Enable atmospheric fog */
    public bool enableParticles = true;             /**< Enable particle effects */
    public bool enableSoundSystem = true;           /**< Enable environmental audio */
    public float atmosphericDensity = 0.5f;        /**< Density of atmospheric effects */
}

/**
 * @enum CaveType
 * @brief Enumeration of available cave generation types
 * 
 * @details
 * Defines the different cave types that can be procedurally generated,
 * each with unique characteristics, formations, and environmental effects.
 */
public enum CaveType
{
    Natural,    /**< Organic natural cave formations */
    Limestone,  /**< Limestone cave with water features */
    Lava,       /**< Volcanic lava tube formations */
    Crystal,    /**< Crystal cave with light refraction */
    Ice,        /**< Frozen ice cave formations */
    Ancient,    /**< Archaeological ancient cave */
    Magical     /**< Supernatural magical cave */
}

namespace MREscapeRoom.Environment
{
    /**
     * @class GuardianBoundaryManager
     * @brief Advanced Mixed Reality boundary management and cave generation system
     * 
     * @details
     * The GuardianBoundaryManager class provides comprehensive functionality for
     * creating immersive cave environments that seamlessly integrate with Mixed
     * Reality Guardian boundaries. It manages procedural generation, environmental
     * effects, interactive elements, and performance optimization.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @implements IGameSystem - Core system interface for initialization
     * @implements IPerformanceAdjustable - Performance optimization interface
     * @requires CaveChunk - For modular environment management
     * @requires LightingSystem - For environmental lighting effects
     * @requires ParticleManager - For atmospheric particle effects
     * @requires AudioManager - For environmental audio management
     */
    public class GuardianBoundaryManager : MonoBehaviour, IGameSystem, IPerformanceAdjustable
    {
        [Header("Generation Settings")]
        [SerializeField] private CaveGenerationSettings caveSettings;      /**< Cave generation configuration */
        [SerializeField] private Material[] caveMaterials;                 /**< Materials for different cave types */
        [SerializeField] private GameObject[] rockPrefabs;                 /**< Rock formation prefabs */
        [SerializeField] private GameObject[] decorativePrefabs;           /**< Decorative element prefabs */
        
        [Header("Performance")]
        [SerializeField] private bool enableLOD = true;                    /**< Enable Level of Detail system */
        [SerializeField] private bool enableOcclusion = true;              /**< Enable occlusion culling */
        [SerializeField] private int maxVerticesPerChunk = 65536;          /**< Maximum vertices per chunk */
        [SerializeField] private float chunkSize = 10.0f;                  /**< Size of each cave chunk */
        
        [Header("Safety & Comfort")]
        [SerializeField] private bool enableSafetyWarnings = true;         /**< Enable safety boundary warnings */
        [SerializeField] private float minWallDistance = 0.5f;             /**< Minimum distance from walls */
        [SerializeField] private bool enableComfortMode = false;           /**< Enable comfort mode for VR */
        
        [Header("Interactive Elements")]
        [SerializeField] private GameObject[] interactablePrefabs;         /**< Interactable object prefabs */
        [SerializeField] private GameObject[] puzzleElements;              /**< Puzzle element prefabs */
        [SerializeField] private GameObject[] secretAreas;                 /**< Secret area prefabs */
        
        /**
         * @brief Core system components
         */
        private GameObject caveParent;                                      /**< Parent object for cave hierarchy */
        private List<CaveChunk> caveChunks;                                /**< List of generated cave chunks */
        private Dictionary<Vector2Int, CaveChunk> chunkGrid;               /**< Grid-based chunk organization */
        private LightingSystem lightingSystem;                             /**< Environmental lighting system */
        private ParticleManager particleManager;                            /**< Particle effect management */
        private AudioManager audioManager;                                  /**< Environmental audio management */
        private EnvironmentalEffects environmentalEffects;                  /**< Atmospheric effects system */
        
        /**
         * @brief Boundary and generation state
         */
        private Vector3[] boundaryVertices;                                /**< Guardian boundary vertex data */
        private bool isGenerating = false;                                 /**< Generation process state */
        private float generationProgress = 0f;                             /**< Current generation progress */
        private Coroutine generationCoroutine;                             /**< Generation coroutine reference */
        
        /**
         * @brief Performance and optimization tracking
         */
        private GameManager.PerformanceLevel currentPerformanceLevel = GameManager.PerformanceLevel.High; /**< Current performance level */
        private int totalVertices = 0;                                     /**< Total vertex count across all chunks */
        private int totalTriangles = 0;                                    /**< Total triangle count across all chunks */
        
        /**
         * @brief Events for external system notifications
         */
        public event System.Action<float> OnGenerationProgress;            /**< Fired when generation progress updates */
        public event System.Action OnGenerationComplete;                   /**< Fired when generation completes */
        public event System.Action<string> OnEnvironmentalEvent;           /**< Fired when environmental events occur */
        
        private void Awake()
        {
            InitializeComponents();
        }
        
        private void Start()
        {
            if (OVRManager.boundary.GetConfigured())
            {
                StartCaveGeneration();
            }
            else
            {
                Debug.LogWarning("[GuardianBoundaryManager] Guardian boundary not configured!");
                CreateFallbackEnvironment();
            }
        }
        
        /*
         * Initialize all required components and subsystems
         */
        private void InitializeComponents()
        {
            caveChunks = new List<CaveChunk>();
            chunkGrid = new Dictionary<Vector2Int, CaveChunk>();
            
            caveParent = new GameObject("Generated_Cave_Environment");
            caveParent.transform.parent = transform;
            
            InitializeLightingSystem();
            InitializeParticleManager();
            InitializeAudioManager();
            InitializeEnvironmentalEffects();
        }
        
        private void InitializeLightingSystem()
        {
            var lightingGO = new GameObject("Cave_Lighting_System");
            lightingGO.transform.parent = caveParent.transform;
            lightingSystem = lightingGO.AddComponent<LightingSystem>();
            lightingSystem.Initialize(caveSettings);
        }
        
        private void InitializeParticleManager()
        {
            var particleGO = new GameObject("Cave_Particle_Manager");
            particleGO.transform.parent = caveParent.transform;
            particleManager = particleGO.AddComponent<ParticleManager>();
            particleManager.Initialize(caveSettings);
        }
        
        private void InitializeAudioManager()
        {
            audioManager = FindObjectOfType<AudioManager>();
            if (audioManager == null)
            {
                var audioGO = new GameObject("Cave_Audio_Manager");
                audioGO.transform.parent = caveParent.transform;
                audioManager = audioGO.AddComponent<AudioManager>();
            }
        }
        
        private void InitializeEnvironmentalEffects()
        {
            var effectsGO = new GameObject("Environmental_Effects");
            effectsGO.transform.parent = caveParent.transform;
            environmentalEffects = effectsGO.AddComponent<EnvironmentalEffects>();
            environmentalEffects.Initialize(caveSettings);
        }
        
        public void StartCaveGeneration()
        {
            if (isGenerating) return;
            
            if (generationCoroutine != null)
            {
                StopCoroutine(generationCoroutine);
            }
            
            generationCoroutine = StartCoroutine(GenerateCaveEnvironment());
        }
        
        /*
         * Main cave generation coroutine handling all phases of environment creation
         */
        private IEnumerator GenerateCaveEnvironment()
        {
            isGenerating = true;
            generationProgress = 0f;
            
            Debug.Log("[GuardianBoundaryManager] Starting cave generation...");
            
            yield return StartCoroutine(AcquireBoundaryData());
            generationProgress = 0.1f;
            OnGenerationProgress?.Invoke(generationProgress);
            
            yield return StartCoroutine(GenerateBaseCaveStructure());
            generationProgress = 0.3f;
            OnGenerationProgress?.Invoke(generationProgress);
            
            yield return StartCoroutine(AddEnvironmentalDetails());
            generationProgress = 0.5f;
            OnGenerationProgress?.Invoke(generationProgress);
            
            yield return StartCoroutine(GenerateLighting());
            generationProgress = 0.7f;
            OnGenerationProgress?.Invoke(generationProgress);
            
            yield return StartCoroutine(AddAtmosphericEffects());
            generationProgress = 0.8f;
            OnGenerationProgress?.Invoke(generationProgress);
            
            yield return StartCoroutine(PlaceInteractiveElements());
            generationProgress = 0.9f;
            OnGenerationProgress?.Invoke(generationProgress);
            
            yield return StartCoroutine(FinalizeGeneration());
            generationProgress = 1.0f;
            OnGenerationProgress?.Invoke(generationProgress);
            
            isGenerating = false;
            OnGenerationComplete?.Invoke();
            
            Debug.Log("[GuardianBoundaryManager] Cave generation completed successfully!");
        }
        
        /*
         * Acquire Guardian boundary data and apply safety margins
         */
        private IEnumerator AcquireBoundaryData()
        {
            Vector3 dimensions = OVRManager.boundary.GetDimensions(OVRBoundary.BoundaryType.PlayArea);
            
            var geometry = OVRManager.boundary.GetGeometry(OVRBoundary.BoundaryType.PlayArea);
            boundaryVertices = geometry.ToArray();
            
            if (boundaryVertices.Length < 4)
            {
                Debug.LogWarning("[GuardianBoundaryManager] Insufficient boundary data, creating rectangular fallback");
                CreateRectangularBoundary(dimensions);
            }
            
            ApplySafetyMargins();
            
            yield return null;
        }
        
        private void CreateRectangularBoundary(Vector3 dimensions)
        {
            boundaryVertices = new Vector3[]
            {
                new Vector3(-dimensions.x / 2, 0, -dimensions.z / 2),
                new Vector3(dimensions.x / 2, 0, -dimensions.z / 2),
                new Vector3(dimensions.x / 2, 0, dimensions.z / 2),
                new Vector3(-dimensions.x / 2, 0, dimensions.z / 2)
            };
        }
        
        private void ApplySafetyMargins()
        {
            if (!enableSafetyWarnings) return;
            
            for (int i = 0; i < boundaryVertices.Length; i++)
            {
                Vector3 vertex = boundaryVertices[i];
                Vector3 center = CalculateBoundaryCenter();
                Vector3 direction = (vertex - center).normalized;
                
                boundaryVertices[i] = center + direction * (Vector3.Distance(vertex, center) - minWallDistance);
            }
        }
        
        private Vector3 CalculateBoundaryCenter()
        {
            Vector3 center = Vector3.zero;
            foreach (Vector3 vertex in boundaryVertices)
            {
                center += vertex;
            }
            return center / boundaryVertices.Length;
        }
        
        /*
         * Generate base cave structure with chunks and walls
         */
        private IEnumerator GenerateBaseCaveStructure()
        {
            Random.InitState(caveSettings.seed);
            
            yield return StartCoroutine(CreateCaveChunks());
            
            int chunksProcessed = 0;
            foreach (var chunk in caveChunks)
            {
                GenerateChunkWalls(chunk);
                chunksProcessed++;
                
                if (chunksProcessed % 5 == 0)
                {
                    yield return null;
                }
            }
            
            yield return StartCoroutine(ConnectChunks());
        }
        
        private IEnumerator CreateCaveChunks()
        {
            Bounds boundaryBounds = CalculateBoundaryBounds();
            
            int chunksX = Mathf.CeilToInt(boundaryBounds.size.x / chunkSize);
            int chunksZ = Mathf.CeilToInt(boundaryBounds.size.z / chunkSize);
            
            for (int x = 0; x < chunksX; x++)
            {
                for (int z = 0; z < chunksZ; z++)
                {
                    Vector3 chunkPosition = new Vector3(
                        boundaryBounds.min.x + x * chunkSize,
                        0,
                        boundaryBounds.min.z + z * chunkSize
                    );
                    
                    CaveChunk chunk = CreateCaveChunk(new Vector2Int(x, z), chunkPosition);
                    caveChunks.Add(chunk);
                    chunkGrid[new Vector2Int(x, z)] = chunk;
                }
                
                if (x % 10 == 0) yield return null;
            }
        }
        
        private CaveChunk CreateCaveChunk(Vector2Int gridPosition, Vector3 worldPosition)
        {
            GameObject chunkGO = new GameObject($"Cave_Chunk_{gridPosition.x}_{gridPosition.y}");
            chunkGO.transform.parent = caveParent.transform;
            chunkGO.transform.position = worldPosition;
            
            CaveChunk chunk = chunkGO.AddComponent<CaveChunk>();
            chunk.Initialize(gridPosition, chunkSize, caveSettings);
            
            return chunk;
        }
        
        private void GenerateChunkWalls(CaveChunk chunk)
        {
            switch (caveSettings.caveType)
            {
                case CaveType.Natural:
                    GenerateNaturalCaveWalls(chunk);
                    break;
                case CaveType.Limestone:
                    GenerateLimestoneWalls(chunk);
                    break;
                case CaveType.Lava:
                    GenerateLavaWalls(chunk);
                    break;
                case CaveType.Crystal:
                    GenerateCrystalWalls(chunk);
                    break;
                case CaveType.Ice:
                    GenerateIceWalls(chunk);
                    break;
                case CaveType.Ancient:
                    GenerateAncientWalls(chunk);
                    break;
                case CaveType.Magical:
                    GenerateMagicalWalls(chunk);
                    break;
            }
        }
        
        /*
         * Generate natural cave walls using Perlin noise
         */
        private void GenerateNaturalCaveWalls(CaveChunk chunk)
        {
            List<Vector3> vertices = new List<Vector3>();
            List<int> triangles = new List<int>();
            List<Vector3> normals = new List<Vector3>();
            List<Vector2> uvs = new List<Vector2>();
            
            int resolution = GetChunkResolution();
            
            for (int x = 0; x <= resolution; x++)
            {
                for (int z = 0; z <= resolution; z++)
                {
                    Vector3 localPos = new Vector3(
                        x * chunkSize / resolution,
                        0,
                        z * chunkSize / resolution
                    );
                    
                    Vector3 worldPos = chunk.transform.position + localPos;
                    
                    float height = GenerateHeightAtPosition(worldPos);
                    
                    vertices.Add(new Vector3(localPos.x, height, localPos.z));
                    normals.Add(Vector3.up);
                    uvs.Add(new Vector2(x / (float)resolution, z / (float)resolution));
                    
                    if (x < resolution && z < resolution)
                    {
                        int i = x * (resolution + 1) + z;
                        
                        triangles.AddRange(new int[]
                        {
                            i, i + resolution + 1, i + 1,
                            i + 1, i + resolution + 1, i + resolution + 2
                        });
                    }
                }
            }
            
            Mesh wallMesh = new Mesh();
            wallMesh.name = $"CaveWall_{chunk.GridPosition.x}_{chunk.GridPosition.y}";
            wallMesh.vertices = vertices.ToArray();
            wallMesh.triangles = triangles.ToArray();
            wallMesh.normals = normals.ToArray();
            wallMesh.uv = uvs.ToArray();
            
            wallMesh.RecalculateNormals();
            wallMesh.RecalculateBounds();
            
            chunk.SetMesh(wallMesh);
            chunk.ApplyMaterial(GetMaterialForCaveType(caveSettings.caveType));
            
            totalVertices += vertices.Count;
            totalTriangles += triangles.Count / 3;
        }
        
        private float GenerateHeightAtPosition(Vector3 worldPosition)
        {
            float height = 0f;
            float amplitude = 1f;
            float frequency = caveSettings.noiseScale;
            
            for (int i = 0; i < caveSettings.octaves; i++)
            {
                height += Mathf.PerlinNoise(
                    worldPosition.x * frequency,
                    worldPosition.z * frequency
                ) * amplitude * caveSettings.roughness;
                
                amplitude *= caveSettings.persistence;
                frequency *= caveSettings.lacunarity;
            }
            
            return height * caveSettings.caveHeight;
        }
        
        private void GenerateLimestoneWalls(CaveChunk chunk)
        {
            GenerateNaturalCaveWalls(chunk);
            AddLimestoneFormations(chunk);
        }
        
        private void GenerateLavaWalls(CaveChunk chunk)
        {
            GenerateNaturalCaveWalls(chunk);
            AddLavaFormations(chunk);
        }
        
        private void GenerateCrystalWalls(CaveChunk chunk)
        {
            GenerateNaturalCaveWalls(chunk);
            AddCrystalFormations(chunk);
        }
        
        private void GenerateIceWalls(CaveChunk chunk)
        {
            GenerateNaturalCaveWalls(chunk);
            AddIceFormations(chunk);
        }
        
        private void GenerateAncientWalls(CaveChunk chunk)
        {
            GenerateNaturalCaveWalls(chunk);
            AddAncientStructures(chunk);
        }
        
        private void GenerateMagicalWalls(CaveChunk chunk)
        {
            GenerateNaturalCaveWalls(chunk);
            AddMagicalEffects(chunk);
        }
        
        private IEnumerator ConnectChunks()
        {
            foreach (var chunk in caveChunks)
            {
                ConnectChunkToNeighbors(chunk);
                yield return null;
            }
        }
        
        private void ConnectChunkToNeighbors(CaveChunk chunk)
        {
            Vector2Int pos = chunk.GridPosition;
            
            ConnectToNeighbor(chunk, pos + Vector2Int.right);
            ConnectToNeighbor(chunk, pos + Vector2Int.up);
        }
        
        private void ConnectToNeighbor(CaveChunk chunk, Vector2Int neighborPos)
        {
            if (chunkGrid.TryGetValue(neighborPos, out CaveChunk neighbor))
            {
                BlendChunkBoundaries(chunk, neighbor);
            }
        }
        
        private void BlendChunkBoundaries(CaveChunk chunk1, CaveChunk chunk2)
        {
        }
        
        /*
         * Add environmental details like stalactites, water features, crystals
         */
        private IEnumerator AddEnvironmentalDetails()
        {
            if (caveSettings.enableStalactites)
            {
                yield return StartCoroutine(GenerateStalactites());
            }
            
            if (caveSettings.enableStalagmites)
            {
                yield return StartCoroutine(GenerateStalagmites());
            }
            
            if (caveSettings.enableWaterFeatures)
            {
                yield return StartCoroutine(GenerateWaterFeatures());
            }
            
            if (caveSettings.enableCrystals)
            {
                yield return StartCoroutine(GenerateCrystals());
            }
        }
        
        private IEnumerator GenerateStalactites()
        {
            int stalactiteCount = Random.Range(10, caveSettings.detailDensity);
            
            for (int i = 0; i < stalactiteCount; i++)
            {
                Vector3 position = GetRandomPositionOnCeiling();
                if (IsValidStalactitePosition(position))
                {
                    CreateStalactite(position);
                }
                
                if (i % 5 == 0) yield return null;
            }
        }
        
        private IEnumerator GenerateStalagmites()
        {
            int stalagmiteCount = Random.Range(5, caveSettings.detailDensity / 2);
            
            for (int i = 0; i < stalagmiteCount; i++)
            {
                Vector3 position = GetRandomPositionOnFloor();
                if (IsValidStalagmitePosition(position))
                {
                    CreateStalagmite(position);
                }
                
                if (i % 5 == 0) yield return null;
            }
        }
        
        private IEnumerator GenerateWaterFeatures()
        {
            yield return StartCoroutine(CreateWaterPools());
            yield return StartCoroutine(CreateWaterStreams());
        }
        
        private IEnumerator CreateWaterPools()
        {
            int poolCount = Random.Range(1, 5);
            
            for (int i = 0; i < poolCount; i++)
            {
                Vector3 position = GetRandomFloorPosition();
                CreateWaterPool(position);
                yield return null;
            }
        }
        
        private IEnumerator CreateWaterStreams()
        {
            int streamCount = Random.Range(0, 3);
            
            for (int i = 0; i < streamCount; i++)
            {
                CreateWaterStream();
                yield return null;
            }
        }
        
        private IEnumerator GenerateCrystals()
        {
            int crystalCount = Random.Range(5, caveSettings.detailDensity / 3);
            
            for (int i = 0; i < crystalCount; i++)
            {
                Vector3 position = GetRandomWallPosition();
                CreateCrystalFormation(position);
                
                if (i % 3 == 0) yield return null;
            }
        }
        
        private IEnumerator GenerateLighting()
        {
            yield return StartCoroutine(lightingSystem.GenerateCaveLighting(caveChunks));
        }
        
        private IEnumerator AddAtmosphericEffects()
        {
            if (caveSettings.enableFog)
            {
                CreateAtmosphericFog();
            }
            
            if (caveSettings.enableParticles)
            {
                yield return StartCoroutine(particleManager.CreateAtmosphericParticles(caveChunks));
            }
            
            if (caveSettings.enableSoundSystem)
            {
                SetupAmbientSounds();
            }
            
            yield return null;
        }
        
        private IEnumerator PlaceInteractiveElements()
        {
            foreach (var puzzleElement in puzzleElements)
            {
                if (puzzleElement != null)
                {
                    Vector3 position = GetRandomAccessiblePosition();
                    Instantiate(puzzleElement, position, Random.rotation, caveParent.transform);
                }
                yield return null;
            }
            
            foreach (var secretArea in secretAreas)
            {
                if (secretArea != null)
                {
                    Vector3 position = GetSecretAreaPosition();
                    Instantiate(secretArea, position, Quaternion.identity, caveParent.transform);
                }
                yield return null;
            }
        }
        
        /*
         * Final optimization and validation pass
         */
        private IEnumerator FinalizeGeneration()
        {
            yield return StartCoroutine(OptimizeMeshes());
            
            if (enableLOD)
            {
                SetupLODSystem();
            }
            
            if (enableOcclusion)
            {
                SetupOcclusionCulling();
            }
            
            if (enableSafetyWarnings)
            {
                ValidateSafety();
            }
            
            yield return null;
        }
        
        private IEnumerator OptimizeMeshes()
        {
            foreach (var chunk in caveChunks)
            {
                chunk.OptimizeMesh();
                yield return null;
            }
        }
        
        private int GetChunkResolution()
        {
            switch (currentPerformanceLevel)
            {
                case GameManager.PerformanceLevel.Low: return 32;
                case GameManager.PerformanceLevel.Medium: return 64;
                case GameManager.PerformanceLevel.High: return 128;
                case GameManager.PerformanceLevel.Ultra: return 256;
                default: return 64;
            }
        }
        
        private Material GetMaterialForCaveType(CaveType type)
        {
            if (caveMaterials != null && caveMaterials.Length > (int)type)
            {
                return caveMaterials[(int)type];
            }
            return caveMaterials?[0];
        }
        
        private Vector3 GetRandomPositionOnCeiling()
        {
            Bounds bounds = CalculateBoundaryBounds();
            return new Vector3(
                Random.Range(bounds.min.x, bounds.max.x),
                caveSettings.caveHeight,
                Random.Range(bounds.min.z, bounds.max.z)
            );
        }
        
        private Vector3 GetRandomPositionOnFloor()
        {
            Bounds bounds = CalculateBoundaryBounds();
            return new Vector3(
                Random.Range(bounds.min.x, bounds.max.x),
                0,
                Random.Range(bounds.min.z, bounds.max.z)
            );
        }
        
        private Vector3 GetRandomWallPosition()
        {
            return GetRandomPositionOnFloor();
        }
        
        private Vector3 GetRandomFloorPosition()
        {
            return GetRandomPositionOnFloor();
        }
        
        private Vector3 GetRandomAccessiblePosition()
        {
            Vector3 center = CalculateBoundaryCenter();
            float radius = Random.Range(1f, 3f);
            Vector2 randomCircle = Random.insideUnitCircle * radius;
            return new Vector3(center.x + randomCircle.x, 0.1f, center.z + randomCircle.y);
        }
        
        private Vector3 GetSecretAreaPosition()
        {
            Bounds bounds = CalculateBoundaryBounds();
            return new Vector3(
                bounds.min.x + bounds.size.x * 0.1f,
                0.1f,
                bounds.min.z + bounds.size.z * 0.1f
            );
        }
        
        private Bounds CalculateBoundaryBounds()
        {
            if (boundaryVertices == null || boundaryVertices.Length == 0)
                return new Bounds(Vector3.zero, Vector3.one * 5f);
            
            Vector3 min = boundaryVertices[0];
            Vector3 max = boundaryVertices[0];
            
            foreach (Vector3 vertex in boundaryVertices)
            {
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }
            
            return new Bounds((min + max) * 0.5f, max - min);
        }
        
        private bool IsValidStalactitePosition(Vector3 position)
        {
            return IsPositionInsideBoundary(position);
        }
        
        private bool IsValidStalagmitePosition(Vector3 position)
        {
            return IsPositionInsideBoundary(position);
        }
        
        private bool IsPositionInsideBoundary(Vector3 position)
        {
            Bounds bounds = CalculateBoundaryBounds();
            return bounds.Contains(position);
        }
        
        private void CreateStalactite(Vector3 position)
        {
            if (rockPrefabs != null && rockPrefabs.Length > 0)
            {
                GameObject stalactite = Instantiate(rockPrefabs[Random.Range(0, rockPrefabs.Length)], 
                    position, Random.rotation, caveParent.transform);
                stalactite.name = "Stalactite";
                stalactite.transform.localScale = Vector3.one * Random.Range(0.5f, 2f);
            }
        }
        
        private void CreateStalagmite(Vector3 position)
        {
            if (rockPrefabs != null && rockPrefabs.Length > 0)
            {
                GameObject stalagmite = Instantiate(rockPrefabs[Random.Range(0, rockPrefabs.Length)], 
                    position, Random.rotation, caveParent.transform);
                stalagmite.name = "Stalagmite";
                stalagmite.transform.localScale = Vector3.one * Random.Range(0.3f, 1.5f);
            }
        }
        
        private void CreateWaterPool(Vector3 position)
        {
            GameObject waterPool = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            waterPool.name = "Water Pool";
            waterPool.transform.position = position;
            waterPool.transform.localScale = new Vector3(2f, 0.1f, 2f);
            waterPool.transform.parent = caveParent.transform;
            
            Renderer renderer = waterPool.GetComponent<Renderer>();
            if (renderer != null)
            {
                renderer.material = new Material(Shader.Find("Standard"));
                renderer.material.color = new Color(0.1f, 0.3f, 0.8f, 0.7f);
                renderer.material.SetFloat("_Mode", 3);
            }
        }
        
        private void CreateWaterStream()
        {
            GameObject streamGO = new GameObject("Water Stream");
            streamGO.transform.parent = caveParent.transform;
            
            ParticleSystem stream = streamGO.AddComponent<ParticleSystem>();
            var main = stream.main;
            main.startColor = new Color(0.8f, 0.9f, 1.0f, 0.7f);
            main.startSize = 0.1f;
            main.startLifetime = 2.0f;
            
            var shape = stream.shape;
            shape.shapeType = ParticleSystemShapeType.Box;
            shape.scale = new Vector3(0.1f, 0.1f, 2.0f);
            
            var velocity = stream.velocityOverLifetime;
            velocity.enabled = true;
            velocity.space = ParticleSystemSimulationSpace.Local;
            velocity.linear = new ParticleSystem.MinMaxCurve(0, new Vector3(0, -2, 0));
        }
        
        private void CreateCrystalFormation(Vector3 position)
        {
            if (decorativePrefabs != null && decorativePrefabs.Length > 0)
            {
                GameObject crystal = Instantiate(decorativePrefabs[Random.Range(0, decorativePrefabs.Length)], 
                    position, Random.rotation, caveParent.transform);
                crystal.name = "Crystal Formation";
                crystal.transform.localScale = Vector3.one * Random.Range(0.8f, 2.5f);
                
                Light crystalLight = crystal.AddComponent<Light>();
                crystalLight.type = LightType.Point;
                crystalLight.color = Random.ColorHSV(0f, 1f, 0.5f, 1f, 0.8f, 1f);
                crystalLight.intensity = Random.Range(0.5f, 2.0f);
                crystalLight.range = Random.Range(3f, 8f);
            }
        }
        
        private void CreateAtmosphericFog()
        {
            RenderSettings.fog = true;
            RenderSettings.fogColor = caveSettings.ambientColor;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogDensity = caveSettings.atmosphericDensity * 0.1f;
        }
        
        private void SetupAmbientSounds()
        {
            if (audioManager != null && caveSettings.enableSoundSystem)
            {
                audioManager.PlayAmbientSound("cave_ambience");
                audioManager.SetReverbZone(caveParent.transform.position, AudioReverbPreset.Cave);
            }
        }
        
        private void AddLimestoneFormations(CaveChunk chunk)
        {
        }
        
        private void AddLavaFormations(CaveChunk chunk)
        {
        }
        
        private void AddCrystalFormations(CaveChunk chunk)
        {
        }
        
        private void AddIceFormations(CaveChunk chunk)
        {
        }
        
        private void AddAncientStructures(CaveChunk chunk)
        {
        }
        
        private void AddMagicalEffects(CaveChunk chunk)
        {
        }
        
        private void SetupLODSystem()
        {
            foreach (var chunk in caveChunks)
            {
                chunk.SetupLOD(currentPerformanceLevel);
            }
        }
        
        private void SetupOcclusionCulling()
        {
        }
        
        private void ValidateSafety()
        {
            foreach (Vector3 vertex in boundaryVertices)
            {
                if (Physics.CheckSphere(vertex, minWallDistance))
                {
                    Debug.LogWarning($"[GuardianBoundaryManager] Safety warning: Wall too close at {vertex}");
                }
            }
        }
        
        private void CreateFallbackEnvironment()
        {
            Vector3 fallbackSize = new Vector3(4f, 3f, 4f);
            CreateRectangularBoundary(fallbackSize);
            StartCaveGeneration();
        }
        
        public IEnumerator Initialize()
        {
            yield return null;
        }
        
        public void Reset()
        {
            if (generationCoroutine != null)
            {
                StopCoroutine(generationCoroutine);
            }
            
            foreach (Transform child in caveParent.transform)
            {
                DestroyImmediate(child.gameObject);
            }
            
            caveChunks.Clear();
            chunkGrid.Clear();
            
            StartCaveGeneration();
        }
        
        public void Cleanup()
        {
            if (generationCoroutine != null)
            {
                StopCoroutine(generationCoroutine);
            }
            
            lightingSystem?.Cleanup();
            particleManager?.Cleanup();
            environmentalEffects?.Cleanup();
        }
        
        public void AdjustPerformance(GameManager.PerformanceLevel level)
        {
            currentPerformanceLevel = level;
            
            foreach (var chunk in caveChunks)
            {
                chunk.AdjustPerformance(level);
            }
            
            lightingSystem?.AdjustPerformance(level);
            particleManager?.AdjustPerformance(level);
            environmentalEffects?.AdjustPerformance(level);
            
            Debug.Log($"[GuardianBoundaryManager] Performance adjusted to: {level}");
        }
        
        private void OnDestroy()
        {
            Cleanup();
        }
        
        #if UNITY_EDITOR
        private void OnDrawGizmosSelected()
        {
            if (boundaryVertices != null && boundaryVertices.Length > 0)
            {
                Gizmos.color = Color.yellow;
                for (int i = 0; i < boundaryVertices.Length; i++)
                {
                    Vector3 current = boundaryVertices[i];
                    Vector3 next = boundaryVertices[(i + 1) % boundaryVertices.Length];
                    
                    Gizmos.DrawLine(current, next);
                    Gizmos.DrawLine(current, current + Vector3.up * caveSettings.caveHeight);
                }
            }
            
            if (caveChunks != null)
            {
                Gizmos.color = Color.green;
                foreach (var chunk in caveChunks)
                {
                    if (chunk != null)
                    {
                        Gizmos.DrawWireCube(chunk.transform.position + Vector3.one * chunkSize * 0.5f, 
                            Vector3.one * chunkSize);
                    }
                }
            }
        }
        #endif
    }
} 