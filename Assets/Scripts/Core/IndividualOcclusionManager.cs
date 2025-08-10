/**
 * @file IndividualOcclusionManager.cs
 * @brief Individual occlusion management system for Mixed Reality environments
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements an individual occlusion management system that allows users to
 * dynamically change occlusion types for specific objects in Mixed Reality environments.
 * The system utilizes reflection to access private fields and methods, enabling runtime
 * modification of occlusion behavior without requiring public API modifications.
 * 
 * @features
 * - Dynamic occlusion type switching using controller input
 * - Raycast-based object selection for targeted occlusion changes
 * - Reflection-based access to private occlusion controller fields
 * - Real-time UI updates displaying current occlusion state
 * - Support for multiple occlusion types including soft and hard occlusion
 * - Integration with Oculus VR input system for intuitive control
 * 
 * @occlusion_types
 * The system supports three primary occlusion types:
 * 1. NoOcclusion - Disables all occlusion effects for maximum visibility
 * 2. SoftOcclusion - Applies subtle occlusion with transparency effects
 * 3. HardOcclusion - Implements full occlusion with complete object hiding
 * 
 * @input_system
 * - Primary control: OVRInput.RawButton.Y (configurable)
 * - Raycast-based object selection using controller forward direction
 * - Automatic UI updates upon occlusion type changes
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - Oculus Integration SDK for VR input handling
 * - TextMeshPro for UI text display
 * - Meta XR Depth API for occlusion functionality
 * 
 * @license MIT License
 */

using UnityEngine;
using TMPro;
using System.Reflection;
using Meta.XR.Depth;
using DepthAPISample;

namespace MREscapeRoom.Core
{
    /**
     * @class IndividualOcclusionManager
     * @brief Individual occlusion management system for dynamic occlusion control
     * 
     * @details
     * The IndividualOcclusionManager class provides functionality for dynamically
     * changing occlusion types on individual objects within Mixed Reality environments.
     * It utilizes reflection to access private fields and methods, enabling runtime
     * modification of occlusion behavior without requiring public API modifications.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @requires Transform - For raycast origin positioning
     * @requires TextMeshProUGUI - For UI text display
     * @requires OcclusionController - For occlusion type management
     * @requires OVRInput - For VR controller input handling
     */
    public class IndividualOcclusionManager : MonoBehaviour
    {
        [Header("Occlusion Management Settings")]
        [SerializeField] private Transform rayOrigin;
        [SerializeField] private TextMeshProUGUI occlusionsModeText;
        [SerializeField] private OVRInput.RawButton occlusionChangeButton = OVRInput.RawButton.Y;
        
        /**
         * @brief Currently selected occlusion controller for modification
         */
        private OcclusionController _currentOcclusionController;
        
        /**
         * @brief Unity lifecycle method for continuous updates
         * 
         * @details
         * Monitors controller input for occlusion type changes and performs
         * raycast-based object selection to identify targets for occlusion
         * modification. This method runs every frame to ensure responsive
         * input handling.
         */
        private void Update()
        {
            if (OVRInput.GetDown(occlusionChangeButton))
            {
                ProcessOcclusionChangeRequest();
            }
        }
        
        /**
         * @brief Processes occlusion change requests from user input
         * 
         * @details
         * Performs raycast from the controller to identify objects with
         * occlusion controllers, then changes their occlusion type using
         * reflection-based field access. Updates the UI to reflect the
         * new occlusion state.
         */
        private void ProcessOcclusionChangeRequest()
        {
            RaycastHit hit;
            if (Physics.Raycast(rayOrigin.position, rayOrigin.forward, out hit))
            {
                _currentOcclusionController = hit.collider.GetComponent<OcclusionController>();
                if (_currentOcclusionController != null)
                {
                    ChangeOcclusionType();
                    UpdateUI();
                }
            }
        }
        
        /**
         * @brief Changes the occlusion type of the selected controller
         * 
         * @details
         * Uses reflection to access the private _occlusionType field of the
         * occlusion controller, changes it to the next type in the sequence,
         * and invokes the UpdateMaterialKeywords method to apply changes.
         * This approach enables runtime modification without public API access.
         */
        private void ChangeOcclusionType()
        {
            FieldInfo occlusionTypeField = typeof(OcclusionController).GetField(
                "_occlusionType", 
                BindingFlags.NonPublic | BindingFlags.Instance
            );
            
            if (occlusionTypeField != null)
            {
                OcclusionType currentType = (OcclusionType)occlusionTypeField.GetValue(_currentOcclusionController);
                OcclusionType nextType = GetNextOcclusionType(currentType);
                occlusionTypeField.SetValue(_currentOcclusionController, nextType);
                
                InvokeMaterialKeywordsUpdate();
            }
        }
        
        /**
         * @brief Invokes the material keywords update method
         * 
         * @details
         * Uses reflection to invoke the private UpdateMaterialKeywords method
         * on the occlusion controller to ensure material changes are properly
         * applied and visible in the scene.
         */
        private void InvokeMaterialKeywordsUpdate()
        {
            MethodInfo updateKeywordsMethod = typeof(OcclusionController).GetMethod(
                "UpdateMaterialKeywords", 
                BindingFlags.NonPublic | BindingFlags.Instance
            );
            
            updateKeywordsMethod?.Invoke(_currentOcclusionController, null);
        }
        
        /**
         * @brief Updates the UI text to display current occlusion state
         * 
         * @details
         * Retrieves the current occlusion type from the selected controller
         * and updates the UI text to inform the user of the current state.
         * This provides immediate feedback on occlusion type changes.
         */
        private void UpdateUI()
        {
            if (occlusionsModeText != null)
            {
                FieldInfo occlusionTypeField = typeof(OcclusionController).GetField(
                    "_occlusionType", 
                    BindingFlags.NonPublic | BindingFlags.Instance
                );
                
                if (occlusionTypeField != null)
                {
                    OcclusionType currentType = (OcclusionType)occlusionTypeField.GetValue(_currentOcclusionController);
                    occlusionsModeText.text = $"Occlusion mode: {currentType}";
                }
            }
        }
        
        /**
         * @brief Determines the next occlusion type in the sequence
         * 
         * @details
         * Implements a circular progression through available occlusion types:
         * NoOcclusion → SoftOcclusion → HardOcclusion → NoOcclusion.
         * This provides a predictable and intuitive user experience.
         * 
         * @param currentType The current occlusion type
         * @returns The next occlusion type in the sequence
         */
        private OcclusionType GetNextOcclusionType(OcclusionType currentType)
        {
            switch (currentType)
            {
                case OcclusionType.NoOcclusion:
                    return OcclusionType.SoftOcclusion;
                    
                case OcclusionType.SoftOcclusion:
                    return OcclusionType.HardOcclusion;
                    
                case OcclusionType.HardOcclusion:
                    return OcclusionType.NoOcclusion;
                    
                default:
                    return OcclusionType.NoOcclusion;
            }
        }
        
        /**
         * @brief Sets the raycast origin for object selection
         * 
         * @details
         * Allows runtime modification of the raycast origin point,
         * typically used to adjust the selection point based on
         * controller position or user preferences.
         * 
         * @param origin New raycast origin transform
         */
        public void SetRayOrigin(Transform origin)
        {
            rayOrigin = origin;
        }
        
        /**
         * @brief Sets the UI text component for status display
         * 
         * @details
         * Assigns the TextMeshPro component that will display
         * the current occlusion type and status information.
         * 
         * @param textComponent TextMeshPro component for status display
         */
        public void SetStatusText(TextMeshProUGUI textComponent)
        {
            occlusionsModeText = textComponent;
        }
        
        /**
         * @brief Sets the input button for occlusion changes
         * 
         * @details
         * Configures which controller button will trigger
         * occlusion type changes, allowing customization
         * of the input mapping.
         * 
         * @param button OVRInput button for occlusion changes
         */
        public void SetOcclusionButton(OVRInput.RawButton button)
        {
            occlusionChangeButton = button;
        }
        
        /**
         * @brief Gets the currently selected occlusion controller
         * 
         * @details
         * Returns the occlusion controller that was most recently
         * selected through raycast, enabling external systems to
         * query the current selection state.
         * 
         * @returns Currently selected occlusion controller
         */
        public OcclusionController GetCurrentOcclusionController()
        {
            return _currentOcclusionController;
        }
        
        /**
         * @brief Forces an immediate occlusion type change
         * 
         * @details
         * Triggers an immediate occlusion type change without
         * requiring user input, useful for automated testing
         * or programmatic control scenarios.
         */
        public void ForceOcclusionChange()
        {
            ProcessOcclusionChangeRequest();
        }
    }
}
