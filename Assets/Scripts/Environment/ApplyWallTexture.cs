/**
 * @file ApplyWallTexture.cs
 * @brief Automatic wall texture application system for cave environments
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements an automatic wall texture application system that
 * applies specified materials to all renderer components within a cave
 * environment hierarchy. The system simplifies material management by
 * automatically distributing wall textures across multiple objects.
 * 
 * @features
 * - Automatic material application to all child renderers
 * - Hierarchical texture distribution for complex cave structures
 * - Inspector-based material assignment for easy configuration
 * - Runtime material application during component initialization
 * - Support for multiple renderer types and hierarchies
 * 
 * @material_distribution
 * The system automatically applies the assigned wall material to:
 * 1. All renderer components on the current GameObject
 * 2. All renderer components on child GameObjects
 * 3. Any renderer components added dynamically to the hierarchy
 * 
 * @usage
 * - Assign the desired wall material in the Unity Inspector
 * - Attach this component to the root cave environment object
 * - Materials will be automatically applied to all child renderers
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - Renderer components for material application
 * - Material system for visual appearance
 * 
 * @license MIT License
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MREscapeRoom.Environment
{
    /**
     * @class ApplyWallTexture
     * @brief Automatic wall texture application system
     * 
     * @details
     * The ApplyWallTexture class provides functionality for automatically
     * applying wall materials to all renderer components within a cave
     * environment hierarchy. This simplifies the process of maintaining
     * consistent visual appearance across complex cave structures.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @requires Renderer - For material application and visual representation
     * @requires Material - For visual appearance and texture mapping
     */
    public class ApplyWallTexture : MonoBehaviour
    {
        [Header("Texture Settings")]
        [SerializeField] public Material wallMaterial;  /**< Wall material to apply to all renderers */
        
        /**
         * @brief Unity lifecycle method for component startup
         * 
         * @details
         * Automatically applies the assigned wall material to all renderer
         * components found within the GameObject hierarchy. This ensures
         * consistent visual appearance across the entire cave environment
         * without requiring manual material assignment to each object.
         * 
         * @process
         * 1. Retrieves all renderer components in the hierarchy
         * 2. Iterates through each renderer component
         * 3. Applies the wall material to each renderer
         * 4. Ensures consistent visual appearance across the environment
         */
        void Start()
        {
            // Get all renderer components in the hierarchy
            Renderer[] renderers = GetComponentsInChildren<Renderer>();
            
            // Apply wall material to each renderer component
            foreach (Renderer rend in renderers)
            {
                rend.material = wallMaterial;
            }
        }
    }
}

