/**
 * @file CaveWallGenerator.cs
 * @brief Procedural cave wall generation system for Mixed Reality environments
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements a procedural cave wall generation system that creates
 * dynamic, noise-based cave wall geometry for Mixed Reality escape room environments.
 * The system generates walls with configurable height, noise parameters, and resolution,
 * providing realistic cave-like surfaces with organic variations.
 * 
 * @features
 * - Procedural wall generation using Perlin noise algorithms
 * - Configurable wall height, noise scale, and mesh resolution
 * - Automatic mesh component management and validation
 * - UV mapping generation for texture application
 * - Runtime parameter modification with immediate regeneration
 * - Optimized triangle generation for efficient rendering
 * 
 * @generation_algorithm
 * The wall generation process follows these steps:
 * 1. Vertex generation with Perlin noise-based height variation
 * 2. Triangle generation for proper mesh topology
 * 3. UV coordinate generation for texture mapping
 * 4. Normal calculation for proper lighting
 * 5. Bounds calculation for efficient culling
 * 
 * @mesh_optimization
 * - Efficient vertex and triangle array allocation
 * - Minimal memory overhead with configurable resolution
 * - Automatic mesh component detection and creation
 * - Proper mesh cleanup and resource management
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - MeshFilter and MeshRenderer components for rendering
 * - Material system for visual appearance
 * 
 * @license MIT License
 */

using UnityEngine;

namespace MREscapeRoom.Environment
{
    /**
     * @class CaveWallGenerator
     * @brief Procedural cave wall generation system
     * 
     * @details
     * The CaveWallGenerator class provides functionality for creating procedural
     * cave wall geometry using noise-based algorithms. It automatically manages
     * mesh components, generates optimized geometry, and provides runtime
     * parameter modification capabilities for dynamic wall generation.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @requires MeshFilter - For mesh data storage and rendering
     * @requires MeshRenderer - For material application and visualization
     * @requires Material - For visual appearance and texture mapping
     */
    public class CaveWallGenerator : MonoBehaviour
    {
        [Header("Generation Settings")]
        [SerializeField] private Material wallMaterial;      /**< Material applied to generated wall */
        [SerializeField] private float wallHeight = 4f;      /**< Height of the generated wall in world units */
        [SerializeField] private float noiseScale = 0.1f;    /**< Scale factor for Perlin noise generation */
        [SerializeField] private int resolution = 64;        /**< Mesh resolution for wall generation */
        
        /**
         * @brief Generated wall mesh data
         */
        private Mesh wallMesh;
        
        /**
         * @brief Mesh filter component for mesh data storage
         */
        private MeshFilter meshFilter;
        
        /**
         * @brief Mesh renderer component for visual representation
         */
        private MeshRenderer meshRenderer;
        
        /**
         * @brief Unity lifecycle method for component initialization
         * 
         * @details
         * Automatically detects and creates required mesh components
         * (MeshFilter and MeshRenderer) if they don't exist. This
         * ensures the GameObject has all necessary components for
         * wall generation and rendering.
         */
        private void Awake()
        {
            meshFilter = GetComponent<MeshFilter>();
            meshRenderer = GetComponent<MeshRenderer>();
            
            if (meshFilter == null) meshFilter = gameObject.AddComponent<MeshFilter>();
            if (meshRenderer == null) meshRenderer = gameObject.AddComponent<MeshRenderer>();
        }
        
        /**
         * @brief Unity lifecycle method for component startup
         * 
         * @details
         * Triggers initial wall generation after component initialization
         * to ensure all required components are available and properly
         * configured for mesh generation.
         */
        private void Start()
        {
            GenerateWall();
        }
        
        /**
         * @brief Generates procedural cave wall mesh
         * 
         * @details
         * Creates a complete cave wall mesh using Perlin noise for height
         * variation. The generation process includes vertex creation with
         * noise-based height offsets, triangle generation for proper mesh
         * topology, UV coordinate generation for texture mapping, and
         * automatic normal and bounds calculation.
         * 
         * @algorithm
         * 1. Vertex generation with Perlin noise height variation
         * 2. Triangle generation for quad-based wall structure
         * 3. UV mapping for texture coordinate generation
         * 4. Normal calculation for proper lighting
         * 5. Bounds calculation for efficient rendering
         */
        public void GenerateWall()
        {
            wallMesh = new Mesh();
            wallMesh.name = "Generated_Cave_Wall";
            
            // Calculate array sizes for vertices, triangles, and UVs
            Vector3[] vertices = new Vector3[(resolution + 1) * 2];
            int[] triangles = new int[resolution * 6];
            Vector2[] uvs = new Vector2[vertices.Length];
            
            // Generate vertices with noise-based height variation
            for (int i = 0; i <= resolution; i++)
            {
                float x = (float)i / resolution;
                float noiseValue = Mathf.PerlinNoise(x * noiseScale, 0) * 2f - 1f;
                
                // Create bottom and top vertices with noise offset
                vertices[i] = new Vector3(x * 10f, noiseValue, 0);
                vertices[i + resolution + 1] = new Vector3(x * 10f, wallHeight + noiseValue, 0);
                
                // Generate UV coordinates for texture mapping
                uvs[i] = new Vector2(x, 0);
                uvs[i + resolution + 1] = new Vector2(x, 1);
                
                // Generate triangles for quad-based wall structure
                if (i < resolution)
                {
                    int triIndex = i * 6;
                    int vertIndex = i;
                    
                    // First triangle of the quad
                    triangles[triIndex] = vertIndex;
                    triangles[triIndex + 1] = vertIndex + resolution + 1;
                    triangles[triIndex + 2] = vertIndex + 1;
                    
                    // Second triangle of the quad
                    triangles[triIndex + 3] = vertIndex + 1;
                    triangles[triIndex + 4] = vertIndex + resolution + 1;
                    triangles[triIndex + 5] = vertIndex + resolution + 2;
                }
            }
            
            // Apply generated data to mesh
            wallMesh.vertices = vertices;
            wallMesh.triangles = triangles;
            wallMesh.uv = uvs;
            wallMesh.RecalculateNormals();
            wallMesh.RecalculateBounds();
            
            // Assign mesh to filter and apply material
            meshFilter.mesh = wallMesh;
            
            if (wallMaterial != null)
            {
                meshRenderer.material = wallMaterial;
            }
        }
        
        /**
         * @brief Sets the material for the generated wall
         * 
         * @details
         * Updates the wall material and immediately applies it to the
         * mesh renderer for visual feedback. This method enables
         * runtime material changes without requiring wall regeneration.
         * 
         * @param material New material to apply to the wall
         */
        public void SetMaterial(Material material)
        {
            wallMaterial = material;
            if (meshRenderer != null)
            {
                meshRenderer.material = material;
            }
        }
        
        /**
         * @brief Updates wall generation parameters and regenerates mesh
         * 
         * @details
         * Modifies the wall height, noise scale, and resolution parameters,
         * then triggers immediate wall regeneration to apply the new
         * settings. This provides runtime customization of wall appearance
         * and geometry complexity.
         * 
         * @param height New wall height in world units
         * @param noise New noise scale factor for height variation
         * @param res New mesh resolution for geometry detail
         */
        public void SetParameters(float height, float noise, int res)
        {
            wallHeight = height;
            noiseScale = noise;
            resolution = res;
            GenerateWall();
        }
    }
} 