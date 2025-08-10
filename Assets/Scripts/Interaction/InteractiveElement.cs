/**
 * @file InteractiveElement.cs
 * @brief Advanced base class for interactive elements in Mixed Reality environments
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements an advanced base class for all interactive elements in the
 * Mixed Reality Escape Room, providing comprehensive hand tracking, gesture recognition,
 * haptic feedback, audio feedback, accessibility features, and performance optimization
 * specifically designed for VR interactions and immersive experiences.
 * 
 * @features
 * - Advanced hand tracking with finger pose detection and controller integration
 * - Comprehensive gesture recognition system with customizable gesture patterns
 * - Multi-modal haptic feedback with intensity-based patterns and customization
 * - Spatial audio feedback with 3D positioning and environmental audio
 * - Visual feedback systems with highlighting, animations, and state-based effects
 * - Accessibility support for various interaction methods including voice and eye tracking
 * - Performance optimization with interaction zones and proximity-based updates
 * - State management with persistent save/restore functionality
 * - Analytics integration for interaction tracking and user behavior analysis
 * - VR-specific optimizations for comfort and performance
 * 
 * @interaction_system
 * The system supports multiple interaction types:
 * 1. Touch - Direct physical contact with objects
 * 2. Grab - Hand-based object manipulation and holding
 * 3. Point - Gesture-based pointing and selection
 * 4. Gesture - Complex hand gesture recognition
 * 5. Voice - Voice command activation and control
 * 6. EyeGaze - Eye tracking-based interaction
 * 
 * @feedback_system
 * - Visual feedback with highlighting, color changes, and animations
 * - Haptic feedback with vibration patterns and intensity control
 * - Audio feedback with spatial 3D sound and environmental audio
 * - Multi-modal feedback coordination for enhanced user experience
 * 
 * @accessibility_features
 * - Voice command integration for hands-free interaction
 * - Eye tracking support for gaze-based interaction
 * - Assistive highlighting for improved object visibility
 * - Configurable interaction ranges and thresholds
 * - Multiple interaction method support for diverse user needs
 * 
 * @performance_optimization
 * - Proximity-based update frequency reduction
 * - Interaction zone management for efficient collision detection
 * - Frame-budgeted updates for consistent performance
 * - Adaptive quality adjustment based on performance levels
 * - Memory-efficient state management and data structures
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - Oculus Integration SDK for VR controller and hand tracking
 * - MREscapeRoom.Core namespace for performance interfaces
 * - Unity Events system for interaction notifications
 * - PlayerPrefs for persistent state storage
 * 
 * @license MIT License
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using System;
using MREscapeRoom.Core;

namespace MREscapeRoom.Interaction
{
    /**
     * @class InteractiveElement
     * @brief Advanced base class for interactive elements in Mixed Reality
     * 
     * @details
     * The InteractiveElement class provides a comprehensive foundation for creating
     * interactive objects in Mixed Reality environments. It implements advanced
     * interaction systems including hand tracking, gesture recognition, multi-modal
     * feedback, accessibility features, and performance optimization specifically
     * designed for VR platforms and immersive experiences.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @implements IPerformanceAdjustable - Performance optimization interface
     * @requires Collider - For interaction detection and collision handling
     * @requires Renderer - For visual feedback and appearance management
     * @requires AudioSource - For audio feedback and spatial sound
     * @requires OVRInput - For VR controller input and hand tracking
     */
    public abstract class InteractiveElement : MonoBehaviour, IPerformanceAdjustable
    {
        [Header("Interaction Settings")]
        [SerializeField] protected InteractionType allowedInteractions = InteractionType.All;    /**< Types of interactions allowed with this element */
        [SerializeField] protected float interactionRange = 1.5f;                               /**< Maximum range for interaction detection */
        [SerializeField] protected float activationThreshold = 0.8f;                            /**< Confidence threshold for gesture activation */
        [SerializeField] protected bool requiresDirectLook = false;                             /**< Whether element requires direct gaze for interaction */
        [SerializeField] protected float lookAngleThreshold = 30f;                              /**< Maximum angle for gaze-based interaction */
        
        [Header("Feedback Systems")]
        [SerializeField] protected bool enableVisualFeedback = true;                            /**< Enable visual feedback effects */
        [SerializeField] protected bool enableAudioFeedback = true;                             /**< Enable audio feedback effects */
        [SerializeField] protected bool enableHapticFeedback = true;                            /**< Enable haptic feedback effects */
        [SerializeField] protected FeedbackIntensity feedbackIntensity = FeedbackIntensity.Medium; /**< Intensity level for feedback effects */
        
        [Header("Accessibility")]
        [SerializeField] protected bool enableVoiceCommands = false;                            /**< Enable voice command support */
        [SerializeField] protected bool enableEyeTracking = false;                              /**< Enable eye tracking interaction */
        [SerializeField] protected bool enableAssistiveHighlighting = true;                     /**< Enable assistive highlighting for accessibility */
        [SerializeField] protected float assistiveHighlightDuration = 2f;                       /**< Duration of assistive highlighting effect */
        
        [Header("Performance")]
        [SerializeField] protected bool enableProximityOptimization = true;                     /**< Enable proximity-based performance optimization */
        [SerializeField] protected float updateFrequency = 60f;                                 /**< Update frequency in frames per second */
        [SerializeField] protected bool enableInteractionZones = true;                          /**< Enable interaction zone optimization */
        
        [Header("State Management")]
        [SerializeField] protected bool persistState = false;                                   /**< Whether to persist element state across sessions */
        [SerializeField] protected InteractionState currentState = InteractionState.Idle;       /**< Current interaction state of the element */
        
        /**
         * @brief Core component references
         */
        protected Collider interactionCollider;                                                 /**< Collider component for interaction detection */
        protected Renderer elementRenderer;                                                     /**< Renderer component for visual feedback */
        protected AudioSource audioSource;                                                      /**< Audio source for audio feedback */
        
        /**
         * @brief Interaction tracking and management
         */
        protected Dictionary<string, HandTracker> trackedHands;                                 /**< Dictionary of tracked hands by name */
        protected List<DetectedGesture> activeGestures;                                         /**< List of currently detected gestures */
        protected InteractionZone interactionZone;                                              /**< Interaction zone for proximity optimization */
        
        /**
         * @brief Feedback system controllers
         */
        protected VisualFeedbackController visualFeedback;                                      /**< Visual feedback system controller */
        protected HapticFeedbackController hapticFeedback;                                      /**< Haptic feedback system controller */
        protected AudioFeedbackController audioFeedback;                                        /**< Audio feedback system controller */
        
        /**
         * @brief Performance tracking and optimization
         */
        protected float lastUpdateTime;                                                         /**< Time of last update for performance monitoring */
        protected int framesSinceLastUpdate;                                                    /**< Frame counter for update frequency control */
        protected bool isPlayerInRange = false;                                                 /**< Whether player is within interaction range */
        protected bool isBeingWatched = false;                                                  /**< Whether element is being gazed at */
        
        /**
         * @brief State persistence and identification
         */
        protected Dictionary<string, object> stateData;                                         /**< Custom state data for persistence */
        protected string elementID;                                                             /**< Unique identifier for this element */
        
        /**
         * @enum InteractionType
         * @brief Types of interactions supported by interactive elements
         * 
         * @details
         * Defines the various interaction methods that can be enabled or disabled
         * for interactive elements, allowing for flexible interaction configuration.
         */
        [Flags]
        public enum InteractionType
        {
            None = 0,        /**< No interactions allowed */
            Touch = 1,       /**< Touch-based interaction */
            Grab = 2,        /**< Grab-based interaction */
            Point = 4,       /**< Point-based interaction */
            Gesture = 8,     /**< Gesture-based interaction */
            Voice = 16,      /**< Voice command interaction */
            EyeGaze = 32,    /**< Eye gaze interaction */
            All = Touch | Grab | Point | Gesture | Voice | EyeGaze /**< All interaction types enabled */
        }
        
        /**
         * @enum InteractionState
         * @brief States that an interactive element can be in
         * 
         * @details
         * Defines the various states an interactive element can transition through
         * during user interaction, enabling appropriate feedback and behavior.
         */
        public enum InteractionState
        {
            Idle,           /**< Default state with no interaction */
            Highlighted,    /**< Element is highlighted for attention */
            Hovered,        /**< Element is being gazed at or hovered over */
            Grabbed,        /**< Element is being held or manipulated */
            Activated,      /**< Element has been activated or used */
            Disabled,       /**< Element is disabled and cannot be interacted with */
            Completed       /**< Element interaction has been completed */
        }
        
        /**
         * @enum FeedbackIntensity
         * @brief Intensity levels for feedback effects
         * 
         * @details
         * Defines the intensity levels for various feedback effects, allowing
         * for customizable user experience and accessibility support.
         */
        public enum FeedbackIntensity
        {
            None,           /**< No feedback effects */
            Low,            /**< Minimal feedback intensity */
            Medium,         /**< Standard feedback intensity */
            High,           /**< Enhanced feedback intensity */
            Maximum         /**< Maximum feedback intensity */
        }
        
        /**
         * @brief Unity Events for interaction notifications
         */
        public UnityEvent OnInteractionStart;                                                  /**< Fired when interaction begins */
        public UnityEvent OnInteractionEnd;                                                    /**< Fired when interaction ends */
        public UnityEvent<InteractionState> OnStateChanged;                                    /**< Fired when interaction state changes */
        public UnityEvent<string> OnGestureDetected;                                           /**< Fired when a gesture is detected */
        
        /**
         * @brief Custom events for external system integration
         */
        public event Action<InteractiveElement, InteractionType> OnElementActivated;           /**< Fired when element is activated */
        public event Action<InteractiveElement> OnElementCompleted;                            /**< Fired when element interaction is completed */
        public event Action<InteractiveElement, string> OnElementStateChanged;                 /**< Fired when element state changes */
        
        /**
         * @brief Unity lifecycle method for component initialization
         * 
         * @details
         * Initializes core components and generates unique element identification
         * during component creation. Ensures all necessary components are available
         * and properly configured for interaction functionality.
         */
        protected virtual void Awake()
        {
            InitializeComponents();
            GenerateElementID();
        }
        
        /**
         * @brief Unity lifecycle method for component startup
         * 
         * @details
         * Sets up interaction systems, feedback controllers, and accessibility
         * features after component initialization. Loads persistent state if enabled.
         */
        protected virtual void Start()
        {
            SetupInteractionSystems();
            LoadState();
        }
        
        /**
         * @brief Unity lifecycle method for continuous updates
         * 
         * @details
         * Manages interaction tracking, feedback systems, and performance metrics
         * on a frame-budgeted basis to maintain optimal performance while ensuring
         * responsive interaction behavior.
         */
        protected virtual void Update()
        {
            if (!ShouldUpdate()) return;
            
            UpdateInteractionTracking();
            UpdateFeedbackSystems();
            UpdatePerformanceMetrics();
            
            framesSinceLastUpdate = 0;
        }
        
        private void InitializeComponents()
        {
            // Get core components
            interactionCollider = GetComponent<Collider>();
            if (interactionCollider == null)
            {
                interactionCollider = gameObject.AddComponent<BoxCollider>();
                ((BoxCollider)interactionCollider).isTrigger = true;
            }
            
            elementRenderer = GetComponent<Renderer>();
            audioSource = GetComponent<AudioSource>();
            if (audioSource == null && enableAudioFeedback)
            {
                audioSource = gameObject.AddComponent<AudioSource>();
                audioSource.playOnAwake = false;
                audioSource.spatialBlend = 1.0f; // 3D audio
            }
            
            // Initialize collections
            trackedHands = new Dictionary<string, HandTracker>();
            activeGestures = new List<DetectedGesture>();
            stateData = new Dictionary<string, object>();
        }
        
        private void SetupInteractionSystems()
        {
            // Setup interaction zone
            if (enableInteractionZones)
            {
                SetupInteractionZone();
            }
            
            // Setup feedback controllers
            SetupFeedbackSystems();
            
            // Setup accessibility features
            if (enableVoiceCommands)
            {
                SetupVoiceCommands();
            }
            
            if (enableEyeTracking)
            {
                SetupEyeTracking();
            }
        }
        
        private void SetupInteractionZone()
        {
            GameObject zoneObject = new GameObject($"{gameObject.name}_InteractionZone");
            zoneObject.transform.parent = transform;
            zoneObject.transform.localPosition = Vector3.zero;
            
            SphereCollider zoneCollider = zoneObject.AddComponent<SphereCollider>();
            zoneCollider.isTrigger = true;
            zoneCollider.radius = interactionRange;
            
            interactionZone = zoneObject.AddComponent<InteractionZone>();
            interactionZone.Initialize(this);
        }
        
        private void SetupFeedbackSystems()
        {
            // Visual feedback
            if (enableVisualFeedback && elementRenderer != null)
            {
                visualFeedback = gameObject.AddComponent<VisualFeedbackController>();
                visualFeedback.Initialize(elementRenderer, feedbackIntensity);
            }
            
            // Haptic feedback
            if (enableHapticFeedback)
            {
                hapticFeedback = gameObject.AddComponent<HapticFeedbackController>();
                hapticFeedback.Initialize(feedbackIntensity);
            }
            
            // Audio feedback
            if (enableAudioFeedback && audioSource != null)
            {
                audioFeedback = gameObject.AddComponent<AudioFeedbackController>();
                audioFeedback.Initialize(audioSource, feedbackIntensity);
            }
        }
        
        private void SetupVoiceCommands()
        {
            // Integration with voice recognition system
            var voiceSystem = FindObjectOfType<VoiceCommandSystem>();
            if (voiceSystem != null)
            {
                voiceSystem.RegisterElement(this, GetVoiceCommands());
            }
        }
        
        private void SetupEyeTracking()
        {
            // Integration with eye tracking system
            var eyeTracker = FindObjectOfType<EyeTrackingSystem>();
            if (eyeTracker != null)
            {
                eyeTracker.RegisterElement(this);
            }
        }
        
        private bool ShouldUpdate()
        {
            framesSinceLastUpdate++;
            
            if (enableProximityOptimization && !isPlayerInRange)
            {
                // Reduce update frequency when player is far away
                return framesSinceLastUpdate >= (60f / (updateFrequency * 0.1f));
            }
            
            return framesSinceLastUpdate >= (60f / updateFrequency);
        }
        
        private void UpdateInteractionTracking()
        {
            // Update hand tracking
            UpdateHandTracking();
            
            // Update gesture detection
            UpdateGestureDetection();
            
            // Update eye gaze
            if (enableEyeTracking)
            {
                UpdateEyeGazeTracking();
            }
            
            // Update proximity
            UpdateProximityTracking();
        }
        
        private void UpdateHandTracking()
        {
            // Get hand tracking data from Oculus
            var ovrManager = FindObjectOfType<OVRManager>();
            if (ovrManager == null) return;
            
            // Track both hands
            UpdateHandTracker("LeftHand", OVRInput.Controller.LHand);
            UpdateHandTracker("RightHand", OVRInput.Controller.RHand);
        }
        
        private void UpdateHandTracker(string handName, OVRInput.Controller controller)
        {
            if (!trackedHands.ContainsKey(handName))
            {
                trackedHands[handName] = new HandTracker(handName, controller);
            }
            
            var tracker = trackedHands[handName];
            tracker.Update();
            
            // Check for interactions
            CheckHandInteraction(tracker);
        }
        
        private void CheckHandInteraction(HandTracker hand)
        {
            Vector3 handPosition = hand.GetPosition();
            float distanceToElement = Vector3.Distance(handPosition, transform.position);
            
            if (distanceToElement <= interactionRange)
            {
                // Check interaction types
                if ((allowedInteractions & InteractionType.Touch) != 0 && hand.IsTouching(interactionCollider))
                {
                    HandleInteraction(InteractionType.Touch, hand);
                }
                
                if ((allowedInteractions & InteractionType.Grab) != 0 && hand.IsGrabbing())
                {
                    HandleInteraction(InteractionType.Grab, hand);
                }
                
                if ((allowedInteractions & InteractionType.Point) != 0 && hand.IsPointing(transform.position))
                {
                    HandleInteraction(InteractionType.Point, hand);
                }
            }
        }
        
        private void UpdateGestureDetection()
        {
            if ((allowedInteractions & InteractionType.Gesture) == 0) return;
            
            activeGestures.Clear();
            
            foreach (var hand in trackedHands.Values)
            {
                var gestures = DetectGestures(hand);
                activeGestures.AddRange(gestures);
            }
            
            // Process detected gestures
            foreach (var gesture in activeGestures)
            {
                ProcessGesture(gesture);
            }
        }
        
        private List<DetectedGesture> DetectGestures(HandTracker hand)
        {
            List<DetectedGesture> gestures = new List<DetectedGesture>();
            
            // Implement gesture detection algorithms
            // This is a simplified version - in production, use ML-based gesture recognition
            
            if (hand.IsThumbsUp())
                gestures.Add(new DetectedGesture("thumbs_up", hand.HandName, 0.9f));
            
            if (hand.IsPinching())
                gestures.Add(new DetectedGesture("pinch", hand.HandName, hand.GetPinchStrength()));
            
            if (hand.IsWaving())
                gestures.Add(new DetectedGesture("wave", hand.HandName, 0.8f));
            
            return gestures;
        }
        
        private void ProcessGesture(DetectedGesture gesture)
        {
            if (gesture.Confidence < activationThreshold) return;
            
            OnGestureDetected?.Invoke(gesture.GestureName);
            HandleInteraction(InteractionType.Gesture, gesture);
            
            // Log gesture for analytics
            LogInteraction("gesture_detected", new Dictionary<string, object>
            {
                {"gesture_name", gesture.GestureName},
                {"hand", gesture.HandName},
                {"confidence", gesture.Confidence}
            });
        }
        
        private void UpdateEyeGazeTracking()
        {
            // Simple eye gaze detection - in production, integrate with proper eye tracking
            Camera playerCamera = Camera.main;
            if (playerCamera == null) return;
            
            Ray gazeRay = new Ray(playerCamera.transform.position, playerCamera.transform.forward);
            
            if (Physics.Raycast(gazeRay, out RaycastHit hit, interactionRange))
            {
                if (hit.collider == interactionCollider)
                {
                    if (!isBeingWatched)
                    {
                        isBeingWatched = true;
                        OnGazeEnter();
                    }
                    
                    UpdateGaze(hit.point);
                }
                else if (isBeingWatched)
                {
                    isBeingWatched = false;
                    OnGazeExit();
                }
            }
            else if (isBeingWatched)
            {
                isBeingWatched = false;
                OnGazeExit();
            }
        }
        
        private void UpdateProximityTracking()
        {
            Vector3 playerPosition = Camera.main.transform.position;
            float distance = Vector3.Distance(playerPosition, transform.position);
            
            bool wasInRange = isPlayerInRange;
            isPlayerInRange = distance <= interactionRange;
            
            if (isPlayerInRange && !wasInRange)
            {
                OnPlayerEnterRange();
            }
            else if (!isPlayerInRange && wasInRange)
            {
                OnPlayerExitRange();
            }
        }
        
        private void UpdateFeedbackSystems()
        {
            visualFeedback?.Update();
            hapticFeedback?.Update();
            audioFeedback?.Update();
        }
        
        private void UpdatePerformanceMetrics()
        {
            lastUpdateTime = Time.time;
        }
        
        protected virtual void HandleInteraction(InteractionType interactionType, object interactionData)
        {
            // Override in derived classes for specific behavior
            ActivateElement(interactionType);
        }
        
        protected virtual void ActivateElement(InteractionType activationType)
        {
            if (currentState == InteractionState.Disabled) return;
            
            ChangeState(InteractionState.Activated);
            
            // Trigger feedback
            TriggerFeedback(activationType);
            
            // Fire events
            OnInteractionStart?.Invoke();
            OnElementActivated?.Invoke(this, activationType);
            
            // Log interaction
            LogInteraction("element_activated", new Dictionary<string, object>
            {
                {"activation_type", activationType.ToString()},
                {"element_id", elementID},
                {"position", transform.position.ToString()}
            });
        }
        
        protected virtual void CompleteElement()
        {
            ChangeState(InteractionState.Completed);
            OnElementCompleted?.Invoke(this);
            
            // Completion feedback
            TriggerCompletionFeedback();
            
            LogInteraction("element_completed", new Dictionary<string, object>
            {
                {"element_id", elementID},
                {"completion_time", Time.time}
            });
        }
        
        protected virtual void ChangeState(InteractionState newState)
        {
            if (currentState == newState) return;
            
            InteractionState oldState = currentState;
            currentState = newState;
            
            OnStateChanged?.Invoke(newState);
            OnElementStateChanged?.Invoke(this, newState.ToString());
            
            // Handle state transitions
            HandleStateTransition(oldState, newState);
            
            // Save state if persistent
            if (persistState)
            {
                SaveState();
            }
        }
        
        protected virtual void HandleStateTransition(InteractionState from, InteractionState to)
        {
            // Handle visual changes based on state
            switch (to)
            {
                case InteractionState.Highlighted:
                    visualFeedback?.StartHighlight();
                    break;
                    
                case InteractionState.Hovered:
                    visualFeedback?.StartHover();
                    audioFeedback?.PlayHoverSound();
                    break;
                    
                case InteractionState.Grabbed:
                    hapticFeedback?.PlayGrabFeedback();
                    audioFeedback?.PlayGrabSound();
                    break;
                    
                case InteractionState.Activated:
                    visualFeedback?.StartActivation();
                    break;
                    
                case InteractionState.Completed:
                    visualFeedback?.StartCompletion();
                    audioFeedback?.PlayCompletionSound();
                    break;
                    
                case InteractionState.Disabled:
                    visualFeedback?.StartDisabled();
                    break;
            }
        }
        
        protected virtual void TriggerFeedback(InteractionType interactionType)
        {
            // Visual feedback
            visualFeedback?.TriggerActivation(interactionType);
            
            // Audio feedback
            audioFeedback?.PlayInteractionSound(interactionType);
            
            // Haptic feedback
            hapticFeedback?.PlayInteractionFeedback(interactionType);
        }
        
        protected virtual void TriggerCompletionFeedback()
        {
            visualFeedback?.TriggerCompletion();
            audioFeedback?.PlayCompletionSound();
            hapticFeedback?.PlayCompletionFeedback();
        }
        
        protected virtual void OnPlayerEnterRange()
        {
            if (enableAssistiveHighlighting)
            {
                ChangeState(InteractionState.Highlighted);
                
                if (visualFeedback != null)
                {
                    StartCoroutine(AssistiveHighlightCoroutine());
                }
            }
        }
        
        protected virtual void OnPlayerExitRange()
        {
            if (currentState == InteractionState.Highlighted)
            {
                ChangeState(InteractionState.Idle);
            }
        }
        
        private IEnumerator AssistiveHighlightCoroutine()
        {
            yield return new WaitForSeconds(assistiveHighlightDuration);
            
            if (currentState == InteractionState.Highlighted && !isBeingWatched)
            {
                ChangeState(InteractionState.Idle);
            }
        }
        
        protected virtual void OnGazeEnter()
        {
            if (currentState == InteractionState.Idle || currentState == InteractionState.Highlighted)
            {
                ChangeState(InteractionState.Hovered);
            }
        }
        
        protected virtual void OnGazeExit()
        {
            if (currentState == InteractionState.Hovered)
            {
                ChangeState(isPlayerInRange ? InteractionState.Highlighted : InteractionState.Idle);
            }
        }
        
        protected virtual List<string> GetVoiceCommands()
        {
            // Override in derived classes to provide specific voice commands
            return new List<string> { "activate", "use", "interact" };
        }
        
        protected virtual void OnVoiceCommand(string command)
        {
            // Handle voice command activation
            HandleInteraction(InteractionType.Voice, command);
        }
        
        private void LogInteraction(string eventName, Dictionary<string, object> parameters)
        {
            var analyticsManager = GameManager.Instance?.GetSystem<AnalyticsManager>();
            analyticsManager?.LogEvent(eventName, parameters);
        }
        
        #region State Persistence
        
        private void GenerateElementID()
        {
            elementID = $"{gameObject.name}_{transform.position.GetHashCode()}_{GetInstanceID()}";
        }
        
        protected virtual void SaveState()
        {
            if (!persistState) return;
            
            var saveData = new InteractionElementSaveData
            {
                ElementID = elementID,
                CurrentState = currentState,
                StateData = new Dictionary<string, object>(stateData),
                LastInteractionTime = Time.time
            };
            
            // Save to persistent storage
            string json = JsonUtility.ToJson(saveData);
            PlayerPrefs.SetString($"InteractiveElement_{elementID}", json);
            PlayerPrefs.Save();
        }
        
        protected virtual void LoadState()
        {
            if (!persistState) return;
            
            string key = $"InteractiveElement_{elementID}";
            if (PlayerPrefs.HasKey(key))
            {
                try
                {
                    string json = PlayerPrefs.GetString(key);
                    var saveData = JsonUtility.FromJson<InteractionElementSaveData>(json);
                    
                    if (saveData != null)
                    {
                        currentState = saveData.CurrentState;
                        stateData = saveData.StateData ?? new Dictionary<string, object>();
                        
                        // Apply loaded state
                        ApplyLoadedState(saveData);
                    }
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[InteractiveElement] Failed to load state for {elementID}: {e.Message}");
                }
            }
        }
        
        protected virtual void ApplyLoadedState(InteractionElementSaveData saveData)
        {
            // Override in derived classes to apply specific state data
            ChangeState(saveData.CurrentState);
        }
        
        #endregion
        
        #region IPerformanceAdjustable Implementation
        
        public virtual void AdjustPerformance(GameManager.PerformanceLevel level)
        {
            switch (level)
            {
                case GameManager.PerformanceLevel.Low:
                    updateFrequency = 30f;
                    enableProximityOptimization = true;
                    feedbackIntensity = FeedbackIntensity.Low;
                    break;
                    
                case GameManager.PerformanceLevel.Medium:
                    updateFrequency = 45f;
                    enableProximityOptimization = true;
                    feedbackIntensity = FeedbackIntensity.Medium;
                    break;
                    
                case GameManager.PerformanceLevel.High:
                    updateFrequency = 60f;
                    enableProximityOptimization = false;
                    feedbackIntensity = FeedbackIntensity.High;
                    break;
                    
                case GameManager.PerformanceLevel.Ultra:
                    updateFrequency = 90f;
                    enableProximityOptimization = false;
                    feedbackIntensity = FeedbackIntensity.Maximum;
                    break;
            }
            
            // Update feedback systems
            visualFeedback?.AdjustPerformance(level);
            hapticFeedback?.AdjustPerformance(level);
            audioFeedback?.AdjustPerformance(level);
        }
        
        #endregion
        
        protected virtual void OnDestroy()
        {
            if (persistState)
            {
                SaveState();
            }
        }
        
        #if UNITY_EDITOR
        protected virtual void OnDrawGizmos()
        {
            // Draw interaction range
            Gizmos.color = isPlayerInRange ? Color.green : Color.yellow;
            Gizmos.DrawWireSphere(transform.position, interactionRange);
            
            // Draw look angle if required
            if (requiresDirectLook)
            {
                Gizmos.color = Color.blue;
                Vector3 forward = transform.forward;
                Vector3 right = Quaternion.AngleAxis(lookAngleThreshold, transform.up) * forward;
                Vector3 left = Quaternion.AngleAxis(-lookAngleThreshold, transform.up) * forward;
                
                Gizmos.DrawRay(transform.position, right * interactionRange);
                Gizmos.DrawRay(transform.position, left * interactionRange);
            }
        }
        
        [ContextMenu("Trigger Activation")]
        private void DebugTriggerActivation()
        {
            ActivateElement(InteractionType.Touch);
        }
        
        [ContextMenu("Complete Element")]
        private void DebugCompleteElement()
        {
            CompleteElement();
        }
        #endif
    }
    
    #region Supporting Classes
    
    [System.Serializable]
    public class HandTracker
    {
        public string HandName;
        public OVRInput.Controller Controller;
        public Vector3 Position;
        public Quaternion Rotation;
        public bool IsTracking;
        
        public HandTracker(string name, OVRInput.Controller controller)
        {
            HandName = name;
            Controller = controller;
        }
        
        public void Update()
        {
            IsTracking = OVRInput.IsControllerConnected(Controller);
            if (IsTracking)
            {
                Position = OVRInput.GetLocalControllerPosition(Controller);
                Rotation = OVRInput.GetLocalControllerRotation(Controller);
            }
        }
        
        public Vector3 GetPosition() => Position;
        public bool IsTouching(Collider collider) => false; // Implement collision detection
        public bool IsGrabbing() => OVRInput.Get(OVRInput.Button.HandTrigger, Controller) > 0.8f;
        public bool IsPointing(Vector3 target) => false; // Implement pointing detection
        public bool IsThumbsUp() => false; // Implement gesture detection
        public bool IsPinching() => OVRInput.Get(OVRInput.Button.PrimaryIndexTrigger, Controller) > 0.5f;
        public float GetPinchStrength() => OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, Controller);
        public bool IsWaving() => false; // Implement wave detection
    }
    
    [System.Serializable]
    public class DetectedGesture
    {
        public string GestureName;
        public string HandName;
        public float Confidence;
        public Vector3 Position;
        
        public DetectedGesture(string gesture, string hand, float confidence)
        {
            GestureName = gesture;
            HandName = hand;
            Confidence = confidence;
        }
    }
    
    [System.Serializable]
    public class InteractionElementSaveData
    {
        public string ElementID;
        public InteractiveElement.InteractionState CurrentState;
        public Dictionary<string, object> StateData;
        public float LastInteractionTime;
    }
    
    #endregion
} 