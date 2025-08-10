/**
 * @file OcclusionState.cs
 * @brief Occlusion state management system for Mixed Reality depth rendering
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements an occlusion state management system that tracks and controls
 * the visual representation of objects based on depth information in Mixed Reality
 * environments. The system maintains individual material instances for each object
 * and automatically manages shader keywords to achieve the desired occlusion effects.
 * 
 * @features
 * - Individual material instance management for each object
 * - Automatic renderer component detection and validation
 * - Dynamic occlusion type switching with immediate visual updates
 * - Material keyword management for shader-based occlusion effects
 * - Integration with Meta XR Depth API for environment depth handling
 * - Error handling for missing renderer components
 * 
 * @occlusion_types
 * The system supports three primary occlusion types:
 * 1. NoOcclusion - Standard rendering without depth-based effects
 * 2. SoftOcclusion - Subtle depth-based transparency effects
 * 3. HardOcclusion - Complete depth-based object hiding
 * 
 * @material_management
 * - Automatic creation of unique material instances per object
 * - Dynamic material keyword enabling/disabling based on occlusion state
 * - Support for custom shader keywords defined in depth API
 * - Immediate visual updates upon occlusion state changes
 * - Proper material assignment back to renderer components
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - Meta XR Depth API for occlusion functionality
 * - Oculus SDK for Mixed Reality integration
 * - Custom shader keywords for occlusion effects
 * 
 * @license MIT License
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Meta.XR.Depth;

namespace MREscapeRoom.Core
{
    /**
     * @class OcclusionState
     * @brief Occlusion state management system for depth-based rendering
     * 
     * @details
     * The OcclusionState class provides functionality for managing the occlusion
     * state of individual objects in Mixed Reality environments. It maintains
     * unique material instances for each object and automatically handles
     * shader keyword management to achieve the desired visual effects.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @requires Renderer - For material access and visual representation
     * @requires Material - For shader keyword management
     * @requires Meta XR Depth API - For occlusion functionality
     */
    public class OcclusionState : MonoBehaviour
    {
        /**
         * @brief Renderer component for this object
         * 
         * @details
         * Provides access to the renderer component attached to this GameObject,
         * enabling material management and visual representation control.
         */
        public Renderer Renderer { get; private set; }
        
        /**
         * @brief Current occlusion type applied to this object
         * 
         * @details
         * Defines how this object should be rendered based on depth information.
         * The property is publicly readable but privately settable, ensuring
         * controlled modification through the SetOcclusionType method.
         */
        public OcclusionType CurrentOcclusionType { get; private set; } = OcclusionType.NoOcclusion;
        
        /**
         * @brief Unity lifecycle method for component initialization
         * 
         * @details
         * Automatically detects the renderer component attached to this GameObject,
         * creates a unique material instance for individual control, and initializes
         * the material keywords based on the default occlusion type. This ensures
         * proper visual representation and material independence from the start.
         */
        private void Awake()
        {
            Renderer = GetComponent<Renderer>();
            
            if (!Renderer)
            {
                LogRendererComponentError();
            }
            else
            {
                InitializeMaterialInstance();
                UpdateMaterialKeywords();
            }
        }
        
        /**
         * @brief Logs error message for missing renderer component
         * 
         * @details
         * Provides detailed error information when a required renderer component
         * is not found, aiding in debugging and development.
         */
        private void LogRendererComponentError()
        {
            Debug.LogError("OcclusionState requires a Renderer component to function.", this);
        }
        
        /**
         * @brief Initializes unique material instance for this object
         * 
         * @details
         * Creates a unique material instance based on the shared material,
         * ensuring that each object can have independent occlusion settings
         * without affecting other objects using the same base material.
         */
        private void InitializeMaterialInstance()
        {
            Renderer.material = new Material(Renderer.sharedMaterial);
        }
        
        /**
         * @brief Sets the occlusion type for this object
         * 
         * @details
         * Changes the occlusion type and immediately updates the material keywords
         * to apply the visual changes. This method provides controlled access to
         * occlusion type modification while ensuring proper material updates.
         * 
         * @param type New occlusion type to apply
         */
        public void SetOcclusionType(OcclusionType type)
        {
            CurrentOcclusionType = type;
            UpdateMaterialKeywords();
        }
        
        /**
         * @brief Updates material keywords based on current occlusion state
         * 
         * @details
         * Manages shader keywords to achieve the desired visual effects for each
         * occlusion type. First disables all occlusion-related keywords, then
         * enables the appropriate one based on the current occlusion state.
         * Finally applies the modified material back to the renderer.
         */
        private void UpdateMaterialKeywords()
        {
            if (!Renderer) return;
            
            Material material = Renderer.material;
            
            DisableAllOcclusionKeywords(material);
            EnableAppropriateOcclusionKeyword(material);
            ApplyMaterialToRenderer(material);
        }
        
        /**
         * @brief Disables all occlusion-related material keywords
         * 
         * @details
         * Resets the material state by disabling all occlusion keywords,
         * ensuring a clean slate before applying the new occlusion type.
         * This prevents keyword conflicts and ensures predictable behavior.
         * 
         * @param material Material to modify
         */
        private void DisableAllOcclusionKeywords(Material material)
        {
            material.DisableKeyword(EnvironmentDepthOcclusionController.HardOcclusionKeyword);
            material.DisableKeyword(EnvironmentDepthOcclusionController.SoftOcclusionKeyword);
        }
        
        /**
         * @brief Enables the appropriate occlusion keyword for current state
         * 
         * @details
         * Enables the specific shader keyword that corresponds to the current
         * occlusion type, allowing the shader to apply the appropriate visual
         * effects for depth-based rendering.
         * 
         * @param material Material to modify
         */
        private void EnableAppropriateOcclusionKeyword(Material material)
        {
            switch (CurrentOcclusionType)
            {
                case OcclusionType.HardOcclusion:
                    material.EnableKeyword(EnvironmentDepthOcclusionController.HardOcclusionKeyword);
                    break;
                    
                case OcclusionType.SoftOcclusion:
                    material.EnableKeyword(EnvironmentDepthOcclusionController.SoftOcclusionKeyword);
                    break;
                    
                case OcclusionType.NoOcclusion:
                default:
                    break;
            }
        }
        
        /**
         * @brief Applies modified material back to renderer
         * 
         * @details
         * Assigns the modified material with updated keywords back to the
         * renderer component, ensuring that visual changes are immediately
         * visible in the scene.
         * 
         * @param material Modified material to apply
         */
        private void ApplyMaterialToRenderer(Material material)
        {
            Renderer.material = material;
        }
        
        /**
         * @brief Forces a material keyword update
         * 
         * @details
         * Triggers an immediate update of material keywords based on the
         * current occlusion state, useful for ensuring visual consistency
         * after external material modifications.
         */
        public void ForceMaterialUpdate()
        {
            UpdateMaterialKeywords();
        }
        
        /**
         * @brief Checks if a specific occlusion type is currently active
         * 
         * @details
         * Provides a convenient way to check the current occlusion state
         * without directly accessing the CurrentOcclusionType property.
         * 
         * @param occlusionType Occlusion type to check
         * @returns True if the specified occlusion type is currently active
         */
        public bool IsOcclusionTypeActive(OcclusionType occlusionType)
        {
            return CurrentOcclusionType == occlusionType;
        }
        
        /**
         * @brief Gets the current material instance
         * 
         * @details
         * Returns the unique material instance managed by this occlusion state,
         * allowing external systems to access the material for additional
         * modifications or queries.
         * 
         * @returns Current material instance for this object
         */
        public Material GetCurrentMaterial()
        {
            return Renderer != null ? Renderer.material : null;
        }
        
        /**
         * @brief Resets occlusion state to default
         * 
         * @details
         * Resets the occlusion type to NoOcclusion and updates the material
         * keywords accordingly, providing a way to return to the default
         * visual state.
         */
        public void ResetOcclusionState()
        {
            SetOcclusionType(OcclusionType.NoOcclusion);
        }
    }
}
