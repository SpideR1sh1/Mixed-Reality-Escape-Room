/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Licensed under the Oculus SDK License Agreement (the "License");
 * you may not use the Oculus SDK except in compliance with the License,
 * which is provided at the time of installation or download, or which
 * otherwise accompanies this software in either electronic or hard copy form.
 *
 * You may obtain a copy of the License at
 *
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Oculus SDK
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file OcclusionController.cs
 * @brief Occlusion type management system for Mixed Reality depth rendering
 * @author Meta Platforms, Inc. and affiliates
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements an occlusion controller system that manages the visual
 * representation of objects based on depth information in Mixed Reality environments.
 * The system supports multiple occlusion types including soft and hard occlusion,
 * automatically updating material keywords to achieve the desired visual effects.
 * 
 * @features
 * - Multiple occlusion type support including soft and hard occlusion
 * - Automatic material keyword management for shader-based effects
 * - Runtime occlusion type modification with immediate visual updates
 * - Integration with Meta XR Depth API for environment depth handling
 * - Automatic renderer component detection and material management
 * 
 * @occlusion_types
 * The system supports three primary occlusion types:
 * 1. NoOcclusion - Standard rendering without depth-based effects
 * 2. SoftOcclusion - Subtle depth-based transparency effects
 * 3. HardOcclusion - Complete depth-based object hiding
 * 
 * @material_management
 * - Automatic renderer component detection during initialization
 * - Dynamic material keyword enabling/disabling based on occlusion type
 * - Support for custom shader keywords defined in depth API
 * - Immediate visual updates upon occlusion type changes
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - Meta XR Depth API for occlusion functionality
 * - Oculus SDK for Mixed Reality integration
 * - Custom shader keywords for occlusion effects
 * 
 * @license Oculus SDK License Agreement
 */

using Meta.XR.Depth;
using UnityEngine;

namespace DepthAPISample
{
    /**
     * @class OcclusionController
     * @brief Occlusion type management system for depth-based rendering
     * 
     * @details
     * The OcclusionController class provides functionality for managing how objects
     * are rendered based on depth information in Mixed Reality environments. It
     * automatically handles material keyword management to achieve the desired
     * visual effects for different occlusion types.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @requires Renderer - For material access and visual representation
     * @requires Material - For shader keyword management
     * @requires Meta XR Depth API - For occlusion functionality
     */
    public class OcclusionController : MonoBehaviour
    {
        [Header("Occlusion Settings")]
        [SerializeField] private Renderer _renderer;
        
        /**
         * @brief Current occlusion type applied to this object
         * 
         * @details
         * Defines how this object should be rendered based on depth information.
         * The property is publicly readable but privately settable, ensuring
         * controlled modification through the SetOcclusionType method.
         */
        public OcclusionType OcclusionType { get; private set; } = OcclusionType.NoOcclusion;
        
        /**
         * @brief Unity lifecycle method for component initialization
         * 
         * @details
         * Automatically detects the renderer component attached to this GameObject
         * and initializes the material keywords based on the default occlusion type.
         * This ensures proper visual representation from the start.
         */
        private void Awake()
        {
            _renderer = GetComponent<Renderer>();
            UpdateMaterialKeywords();
        }
        
        /**
         * @brief Sets the occlusion type for this object
         * 
         * @details
         * Changes the occlusion type and immediately updates the material keywords
         * to apply the visual changes. This method provides controlled access to
         * occlusion type modification while ensuring proper material updates.
         * 
         * @param occlusionType New occlusion type to apply
         */
        public void SetOcclusionType(OcclusionType occlusionType)
        {
            this.OcclusionType = occlusionType;
            UpdateMaterialKeywords();
        }
        
        /**
         * @brief Updates material keywords based on current occlusion type
         * 
         * @details
         * Manages shader keywords to achieve the desired visual effects for each
         * occlusion type. First disables all occlusion-related keywords, then
         * enables the appropriate one based on the current occlusion type.
         * This approach ensures clean keyword state management.
         */
        private void UpdateMaterialKeywords()
        {
            Material mat = _renderer.material;
            
            DisableAllOcclusionKeywords(mat);
            EnableAppropriateOcclusionKeyword(mat);
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
            material.DisableKeyword(EnvironmentDepthOcclusionController.SoftOcclusionKeyword);
            material.DisableKeyword(EnvironmentDepthOcclusionController.HardOcclusionKeyword);
        }
        
        /**
         * @brief Enables the appropriate occlusion keyword for current type
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
            switch (OcclusionType)
            {
                case OcclusionType.SoftOcclusion:
                    material.EnableKeyword(EnvironmentDepthOcclusionController.SoftOcclusionKeyword);
                    break;
                    
                case OcclusionType.HardOcclusion:
                    material.EnableKeyword(EnvironmentDepthOcclusionController.HardOcclusionKeyword);
                    break;
                    
                case OcclusionType.NoOcclusion:
                default:
                    break;
            }
        }
        
        /**
         * @brief Gets the current renderer component
         * 
         * @details
         * Returns the renderer component managed by this occlusion controller,
         * allowing external systems to access the visual representation
         * component for additional modifications or queries.
         * 
         * @returns Renderer component for this object
         */
        public Renderer GetRenderer()
        {
            return _renderer;
        }
        
        /**
         * @brief Forces a material keyword update
         * 
         * @details
         * Triggers an immediate update of material keywords based on the
         * current occlusion type, useful for ensuring visual consistency
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
         * without directly accessing the OcclusionType property.
         * 
         * @param occlusionType Occlusion type to check
         * @returns True if the specified occlusion type is currently active
         */
        public bool IsOcclusionTypeActive(OcclusionType occlusionType)
        {
            return OcclusionType == occlusionType;
        }
    }
}

