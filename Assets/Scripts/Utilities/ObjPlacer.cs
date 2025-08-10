/**
 * @file ObjPlacer.cs
 * @brief Advanced object placement and interaction system for Mixed Reality environments
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements a comprehensive object placement and interaction system designed
 * specifically for Mixed Reality environments. It provides advanced raycasting-based
 * object placement, preview systems, occlusion state management, and interactive
 * highlighting capabilities. The system supports multiple placeable object types,
 * real-time preview positioning, and seamless integration with occlusion management
 * systems for enhanced Mixed Reality experiences.
 * 
 * @features
 * - Advanced raycasting-based object placement system
 * - Real-time preview object positioning and rotation
 * - Multiple placeable object type support with cycling
 * - Interactive object highlighting and material management
 * - Occlusion state detection and management integration
 * - Laser pointer visualization with configurable distance
 * - Thumbstick-based rotation and distance control
 * - Automatic object highlighting for interactable elements
 * - Comprehensive input handling for VR controllers
 * - Material preservation and restoration for highlighting
 * 
 * @interaction_system
 * The object placement system operates through the following interaction workflow:
 * 1. Raycast-based object detection and positioning
 * 2. Real-time preview object display with rotation controls
 * 3. Interactive highlighting of interactable objects
 * 4. Occlusion state detection and display
 * 5. Object placement with precise positioning and rotation
 * 6. Material management for visual feedback
 * 
 * @input_handling
 * - Primary thumbstick: Object rotation control (X and Y axes)
 * - Secondary thumbstick: Raycast distance adjustment (Y axis)
 * - Index trigger: Object placement confirmation
 * - B button: Laser pointer toggle
 * - X button: Object type cycling
 * - Y button: Occlusion state modification
 * 
 * @preview_system
 * - Real-time preview object positioning
 * - Rotation synchronization with thumbstick input
 * - Distance-based raycast positioning
 * - Automatic activation/deactivation based on laser state
 * - Material preview for different object types
 * 
 * @occlusion_integration
 * - Automatic detection of interactable objects
 * - Occlusion state display and management
 * - State cycling through available occlusion types
 * - Visual feedback for current occlusion mode
 * - Seamless integration with occlusion management systems
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - Oculus Integration for VR input handling
 * - TextMeshPro for UI text display
 * - LineRenderer for laser pointer visualization
 * - OcclusionState system for state management
 * - Meta XR Depth API for advanced depth functionality
 * 
 * @license MIT License
 */

using System.Collections.Generic;
using UnityEngine;
using TMPro;
using Meta.XR.Depth;
using DepthAPISample;

namespace DepthAPISample
{
    /**
     * @struct PlaceableObject
     * @brief Data structure for placeable object configuration
     * 
     * @details
     * Defines the configuration for placeable objects in the Mixed Reality environment,
     * including both the actual prefab for placement and a preview prefab for
     * positioning visualization. This structure enables efficient object management
     * and preview systems.
     * 
     * @fields
     * - prefab: The actual GameObject to be placed in the environment
     * - previewPrefab: The preview GameObject for positioning visualization
     */
    [System.Serializable]
    public struct PlaceableObject
    {
        public GameObject prefab; /**< Actual GameObject prefab for placement */
        public GameObject previewPrefab; /**< Preview GameObject for positioning visualization */
    }

    /**
     * @class ObjPlacer
     * @brief Advanced object placement and interaction system for Mixed Reality environments
     * 
     * @details
     * The ObjPlacer class provides comprehensive functionality for object placement,
     * interaction, and management in Mixed Reality environments. It implements
     * advanced raycasting systems, preview positioning, interactive highlighting,
     * and seamless integration with occlusion management systems for enhanced
     * user experience and interaction capabilities.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @features Raycast-based object placement and detection
     * @features Real-time preview positioning and rotation
     * @features Interactive highlighting and material management
     * @features Occlusion state integration and management
     * @features Multi-object type support with cycling
     * @features VR controller input handling and visualization
     */
    public class ObjPlacer : MonoBehaviour
    {
        [Header("Core Configuration")]
        [SerializeField] private Transform _rayOrigin; /**< Origin point for raycast operations */
        [SerializeField] private GameObject _cubePrefab; /**< Default cube prefab for placement */
        [SerializeField] private LineRenderer _lineRenderer; /**< Line renderer for laser pointer visualization */
        [SerializeField] private float _initialRayDistance = 10f; /**< Initial raycast distance for object placement */
        
        [Header("Input Configuration")]
        [SerializeField] private OVRInput.RawButton _objectPlacingButton = OVRInput.RawButton.RIndexTrigger; /**< Button for object placement confirmation */
        [SerializeField] private OVRInput.RawButton _toggleLaserButton = OVRInput.RawButton.B; /**< Button for laser pointer toggle */
        [SerializeField] private OVRInput.RawButton _changeObjectButton = OVRInput.RawButton.X; /**< Button for object type cycling */
        [SerializeField] private OVRInput.RawButton occlusionChangeButton = OVRInput.RawButton.Y; /**< Button for occlusion state modification */
        
        [Header("Object Configuration")]
        [SerializeField] private GameObject _previewObjectPrefab; /**< Preview object prefab for positioning visualization */
        [SerializeField] private PlaceableObject[] _placeableObjects; /**< Array of placeable object configurations */
        
        [Header("UI Configuration")]
        [SerializeField] private TextMeshProUGUI occlusionsModeText; /**< UI text for occlusion mode display */
        
        [Header("Visual Configuration")]
        [SerializeField] private Material highlightMaterial; /**< Material for object highlighting */
        
        [Header("Runtime State")]
        private GameObject currentlyHighlighted; /**< Currently highlighted object reference */
        private Material originalMaterial; /**< Original material of highlighted object */
        private List<GameObject> _placedObjects; /**< List of successfully placed objects */
        private Vector3 _rayDirection; /**< Current raycast direction vector */
        private bool _isLaserActive = true; /**< Laser pointer activation state */
        private GameObject _previewObject; /**< Current preview object instance */
        private float _currentRayDistance; /**< Current raycast distance */
        private Vector2 _currentRotation; /**< Current object rotation values */
        private int _currentPrefabIndex = 0; /**< Current prefab index for cycling */
        private int _currentObjectIndex; /**< Current object type index */

        /**
         * @brief Unity lifecycle method for component initialization
         * 
         * @details
         * Initializes the object placer component by setting up initial state,
         * creating the preview object, and configuring default values for
         * raycast distance, rotation, and object management.
         */
        private void Awake()
        {
            // Initialize object management and state variables
            _placedObjects = new List<GameObject>();
            _rayDirection = _rayOrigin.forward;
            _currentRayDistance = _initialRayDistance;
            _currentObjectIndex = 0;
            
            // Create initial preview object for positioning visualization
            CreatePreviewObject();
        }

        /**
         * @brief Unity lifecycle method for per-frame updates
         * 
         * @details
         * Handles all per-frame operations including input processing, laser
         * toggling, raycast updates, and object placement interactions.
         * This method coordinates the entire interaction workflow for each frame.
         */
        private void Update()
        {
            // Process continuous input for rotation and distance control
            HandleThumbstickInput();
            
            // Handle laser pointer toggle functionality
            ToggleLaser();
            
            // Update raycast positioning and preview object positioning
            UpdateRayAndPreviewPosition();

            // Process discrete input actions when laser is active
            if (_isLaserActive)
            {
                // Handle object type cycling input
                if (OVRInput.GetDown(_changeObjectButton))
                {
                    CyclePreviewObject();
                }
                
                // Handle object placement confirmation input
                if (OVRInput.GetDown(_objectPlacingButton))
                {
                    AttemptObjectPlacement();
                }
                
                // Handle occlusion state modification input
                if (OVRInput.GetDown(occlusionChangeButton))
                {
                    AttemptToggleOcclusion();
                }
            }
        }

        /**
         * @brief Attempts to toggle occlusion state of interactable objects
         * 
         * @details
         * Performs raycast detection to identify interactable objects and
         * attempts to modify their occlusion state. This method integrates
         * with the occlusion management system to provide seamless state
         * modification capabilities.
         * 
         * @process
         * 1. Perform raycast to detect interactable objects
         * 2. Verify object has OcclusionState component
         * 3. Cycle to next available occlusion type
         * 4. Update object's occlusion state
         */
        private void AttemptToggleOcclusion()
        {
            RaycastHit hit;
            
            // Perform raycast to detect interactable objects
            if (Physics.Raycast(_rayOrigin.position, _rayOrigin.forward, out hit, _currentRayDistance))
            {
                // Verify object is interactable and button press detected
                if (hit.collider.gameObject.CompareTag("Interactable") && OVRInput.GetDown(occlusionChangeButton))
                {
                    // Retrieve occlusion state component for modification
                    OcclusionState occlusionState = hit.collider.GetComponent<OcclusionState>();
                    if (occlusionState != null)
                    {
                        // Cycle to next available occlusion type
                        OcclusionType nextOcclusionType = GetNextOcclusionType(occlusionState.CurrentOcclusionType);
                        
                        // Apply new occlusion state to object
                        occlusionState.SetOcclusionType(nextOcclusionType);
                    }
                }
            }
        }

        /**
         * @brief Toggles occlusion state of a renderer component
         * 
         * @details
         * Modifies the occlusion state of a renderer component by cycling
         * through available occlusion types. This method provides direct
         * occlusion state modification for renderer-based objects.
         * 
         * @param renderer Renderer component to modify occlusion state
         * @process
         * 1. Retrieve OcclusionState component from renderer
         * 2. Cycle to next available occlusion type
         * 3. Apply new occlusion state
         * 4. Update UI text display
         */
        private void ToggleOcclusion(Renderer renderer)
        {
            // Retrieve occlusion state component for state modification
            OcclusionState occlusionState = renderer.GetComponent<OcclusionState>();
            if (occlusionState == null) return;

            // Cycle to next available occlusion type
            OcclusionType nextOcclusionType = GetNextOcclusionType(occlusionState.CurrentOcclusionType);
            occlusionState.SetOcclusionType(nextOcclusionType);

            // Update UI text to reflect current occlusion mode
            occlusionsModeText.text = $"Occlusion Mode: {nextOcclusionType}";
        }

        /**
         * @brief Cycles through available occlusion types in predefined order
         * 
         * @details
         * Implements a state machine for cycling through available occlusion
         * types. The cycling follows a specific pattern: NoOcclusion →
         * HardOcclusion → SoftOcclusion → NoOcclusion (repeat).
         * 
         * @param currentType Current occlusion type to cycle from
         * @returns Next occlusion type in the cycling sequence
         * @cycle_pattern NoOcclusion → HardOcclusion → SoftOcclusion → NoOcclusion
         */
        private OcclusionType GetNextOcclusionType(OcclusionType currentType)
        {
            switch (currentType)
            {
                case OcclusionType.NoOcclusion:
                    return OcclusionType.HardOcclusion;
                case OcclusionType.SoftOcclusion:
                    return OcclusionType.NoOcclusion;
                case OcclusionType.HardOcclusion:
                    return OcclusionType.SoftOcclusion;
                default:
                    return OcclusionType.NoOcclusion;
            }
        }

        /**
         * @brief Creates the initial preview object for positioning visualization
         * 
         * @details
         * Instantiates the preview object from the current placeable object
         * configuration and sets it to inactive initially. This method
         * establishes the foundation for the preview positioning system.
         * 
         * @process
         * 1. Verify placeable objects array has content
         * 2. Instantiate preview prefab from current object configuration
         * 3. Set preview object to inactive initially
         */
        private void CreatePreviewObject()
        {
            if (_placeableObjects.Length > 0)
            {
                // Create preview object from current configuration
                _previewObject = Instantiate(_placeableObjects[_currentObjectIndex].previewPrefab);
                _previewObject.SetActive(false);
            }
        }

        /**
         * @brief Processes thumbstick input for rotation and distance control
         * 
         * @details
         * Handles continuous thumbstick input for object rotation control
         * and raycast distance adjustment. Primary thumbstick controls
         * rotation, secondary thumbstick controls distance with appropriate
         * sensitivity and clamping for smooth interaction.
         * 
         * @input_mapping
         * - Primary thumbstick: Object rotation (X: Yaw, Y: Pitch)
         * - Secondary thumbstick: Raycast distance adjustment (Y axis)
         * @sensitivity Rotation: 100°/s, Distance: 5 units/s
         * @clamping Distance: _initialRayDistance to 100 units
         */
        private void HandleThumbstickInput()
        {
            // Process primary thumbstick for rotation control
            Vector2 rotationInput = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick);
            _currentRotation += new Vector2(rotationInput.x * 100f * Time.deltaTime, rotationInput.y * 100f * Time.deltaTime); 

            // Process secondary thumbstick for distance control
            float lengthInput = OVRInput.Get(OVRInput.Axis2D.SecondaryThumbstick).y;
            _currentRayDistance += lengthInput * Time.deltaTime * 5f; 
            _currentRayDistance = Mathf.Clamp(_currentRayDistance, _initialRayDistance, 100f);
        }

        /**
         * @brief Toggles laser pointer visibility and preview object state
         * 
         * @details
         * Handles laser pointer toggle functionality by switching between
         * active and inactive states. When toggled, both the line renderer
         * and preview object visibility are synchronized to provide
         * consistent visual feedback.
         * 
         * @input OVRInput.RawButton.B for toggle activation
         * @effects
         * - Line renderer visibility toggle
         * - Preview object visibility synchronization
         * - Laser state variable update
         */
        private void ToggleLaser()
        {
            if (OVRInput.GetDown(_toggleLaserButton))
            {
                // Toggle laser active state
                _isLaserActive = !_isLaserActive;
                
                // Synchronize line renderer visibility with laser state
                _lineRenderer.enabled = _isLaserActive;
                
                // Synchronize preview object visibility with laser state
                if (_previewObject != null)
                {
                    _previewObject.SetActive(_isLaserActive);
                }
            }
        }

        /**
         * @brief Updates raycast positioning and preview object positioning
         * 
         * @details
         * Performs raycast operations to determine object positioning and
         * manages preview object visibility and positioning. This method
         * coordinates the entire positioning workflow including object
         * highlighting, occlusion state detection, and preview positioning.
         * 
         * @workflow
         * 1. Calculate raycast end point based on current distance
         * 2. Perform raycast to detect objects and surfaces
         * 3. Handle interactable object highlighting and occlusion display
         * 4. Position preview object when no interactable objects detected
         * 5. Update line renderer for laser pointer visualization
         * 6. Manage object highlighting and material restoration
         */
        private void UpdateRayAndPreviewPosition()
        {
            // Calculate raycast end point based on current distance and direction
            Vector3 rayEndPoint = _rayOrigin.position + _rayOrigin.forward * _currentRayDistance;
            RaycastHit hit;

            // Perform raycast to detect objects and surfaces
            if (Physics.Raycast(_rayOrigin.position, _rayOrigin.forward, out hit, _currentRayDistance))
            {
                // Handle interactable object detection and highlighting
                if (hit.collider.gameObject.CompareTag("Interactable") && _isLaserActive)
                {
                    // Highlight detected interactable object
                    HighlightObject(hit.collider.gameObject);

                    // Hide preview object when interactable object detected
                    if (_previewObject.activeSelf)
                    {
                        _previewObject.SetActive(false);
                    }

                    // Update occlusion mode display for interactable objects
                    OcclusionState occlusionState = hit.collider.GetComponent<OcclusionState>();
                    if (occlusionState != null)
                    {
                        UpdateOcclusionModeText(occlusionState.CurrentOcclusionType);
                    }
                    else
                    {
                        occlusionsModeText.text = "Occlusion mode: N/A";
                    }
                }
                else
                {
                    // Handle non-interactable object or empty space detection
                    occlusionsModeText.text = "Occlusion mode: N/A";
                    PositionPreviewObject(rayEndPoint);
                    RemoveHighlight();
                }
            }
            else
            {
                // Handle empty space detection (no raycast hit)
                occlusionsModeText.text = "Occlusion mode: N/A";
                PositionPreviewObject(rayEndPoint);
                RemoveHighlight();
            }

            // Update line renderer for laser pointer visualization
            _lineRenderer.SetPositions(new Vector3[] { _rayOrigin.position, rayEndPoint });
            _lineRenderer.startColor = _lineRenderer.endColor = Color.green;
        }

        /**
         * @brief Updates UI text to display current occlusion mode
         * 
         * @details
         * Updates the occlusion mode text display to reflect the current
         * occlusion state of the highlighted object. This provides real-time
         * feedback to the user about the current interaction state.
         * 
         * @param occlusionType Current occlusion type to display
         * @ui_update Text display format: "Occlusion Mode: {occlusionType}"
         */
        private void UpdateOcclusionModeText(OcclusionType occlusionType)
        {
            occlusionsModeText.text = $"Occlusion Mode: {occlusionType}";
        }

        /**
         * @brief Positions preview object at specified location with current rotation
         * 
         * @details
         * Positions the preview object at the specified world position and
         * applies the current rotation values. This method ensures the
         * preview object is visible and properly oriented for positioning
         * visualization.
         * 
         * @param position World position for preview object placement
         * @rotation_application
         * - X rotation: Applied to Y axis (pitch)
         * - Y rotation: Applied to X axis (yaw)
         * - Z rotation: Fixed at 0 (no roll)
         * @visibility_management Automatic activation when laser is active
         */
        private void PositionPreviewObject(Vector3 position)
        {
            // Activate preview object if laser is active and object is inactive
            if (!_previewObject.activeSelf && _isLaserActive)
            {
                _previewObject.SetActive(true);
            }
            
            // Set preview object position and rotation
            _previewObject.transform.position = position;
            _previewObject.transform.rotation = Quaternion.Euler(-_currentRotation.y, _currentRotation.x, 0);
        }

        /**
         * @brief Highlights object by applying highlight material
         * 
         * @details
         * Applies highlighting to the specified object by replacing its
         * current material with the highlight material. This method
         * preserves the original material for restoration and prevents
         * duplicate highlighting of the same object.
         * 
         * @param obj GameObject to apply highlighting to
         * @material_management
         * - Stores original material for restoration
         * - Applies highlight material for visual feedback
         * - Prevents duplicate highlighting of same object
         * @highlighting_effect Visual distinction for interactable objects
         */
        private void HighlightObject(GameObject obj)
        {
            // Prevent duplicate highlighting of the same object
            if (currentlyHighlighted == obj) return;

            // Remove previous highlighting before applying new
            RemoveHighlight();

            // Apply highlighting to new object
            currentlyHighlighted = obj;
            var renderer = currentlyHighlighted.GetComponent<Renderer>();
            originalMaterial = renderer.material;
            renderer.material = highlightMaterial;
        }

        /**
         * @brief Removes object highlighting and restores original material
         * 
         * @details
         * Restores the original material of the currently highlighted object
         * and clears the highlighting state. This method ensures proper
         * material restoration and state cleanup for the highlighting system.
         * 
         * @material_restoration
         * - Restores original material to highlighted object
         * - Clears highlighting state variables
         * - Ensures proper cleanup of highlighting effects
         */
        private void RemoveHighlight()
        {
            if (currentlyHighlighted != null)
            {
                // Restore original material to highlighted object
                var renderer = currentlyHighlighted.GetComponent<Renderer>();
                renderer.material = originalMaterial;
                
                // Clear highlighting state
                currentlyHighlighted = null;
            }
        }

        /**
         * @brief Cycles through available placeable object types
         * 
         * @details
         * Destroys the current preview object and creates a new one
         * from the next available placeable object configuration. This
         * method enables users to cycle through different object types
         * for placement in the environment.
         * 
         * @cycling_behavior
         * - Incremental cycling through object array
         * - Wraps around to beginning when reaching end
         * - Automatic preview object recreation
         * @preview_synchronization Maintains laser state synchronization
         */
        private void CyclePreviewObject()
        {
            // Destroy current preview object for replacement
            if (_previewObject != null)
                Destroy(_previewObject);

            // Cycle to next object type with wraparound
            _currentObjectIndex = (_currentObjectIndex + 1) % _placeableObjects.Length;
            
            // Create new preview object from current configuration
            _previewObject = Instantiate(_placeableObjects[_currentObjectIndex].previewPrefab);
            _previewObject.SetActive(_isLaserActive);
        }

        /**
         * @brief Attempts to place object at preview position
         * 
         * @details
         * Creates the actual object instance at the preview object's
         * position and rotation when placement is confirmed. This method
         * finalizes the object placement process and adds the placed
         * object to the management system.
         * 
         * @placement_process
         * 1. Verify preview object is active and visible
         * 2. Extract placement position and rotation from preview
         * 3. Instantiate actual object prefab at specified location
         * 4. Add placed object to management list
         * @positioning_accuracy Matches preview object position and rotation exactly
         */
        private void AttemptObjectPlacement()
        {
            if (_previewObject != null && _previewObject.activeSelf)
            {
                // Extract placement position and rotation from preview object
                Vector3 placementPosition = _previewObject.transform.position;
                Quaternion placementRotation = Quaternion.Euler(-_currentRotation.y, _currentRotation.x, 0);

                // Instantiate actual object at preview position and rotation
                GameObject placedObject = Instantiate(_placeableObjects[_currentObjectIndex].prefab, placementPosition, placementRotation);
                
                // Add placed object to management system
                _placedObjects.Add(placedObject);
            }
        }
    }
}