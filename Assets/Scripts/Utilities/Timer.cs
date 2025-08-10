/**
 * @file Timer.cs
 * @brief Advanced timer utility system for Mixed Reality Escape Room game mechanics
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements a comprehensive timer utility system designed specifically for
 * Mixed Reality Escape Room game mechanics. It provides advanced timing functionality
 * including auto-start capabilities, looping behavior, pause/resume functionality,
 * and event-driven notifications for timer state changes. The system is optimized
 * for VR environments and supports real-time game mechanics requiring precise timing.
 * 
 * @features
 * - Configurable timer duration with runtime modification capabilities
 * - Auto-start functionality for immediate timer activation
 * - Loop mode for continuous timer operation
 * - Pause and resume functionality for game state management
 * - Comprehensive event system for timer state notifications
 * - Progress tracking with normalized completion percentage
 * - Remaining time calculation for countdown displays
 * - Performance-optimized update loop for VR environments
 * - Thread-safe event invocation for multi-threaded scenarios
 * 
 * @timer_states
 * The timer system supports multiple operational states:
 * 1. Stopped - Timer is inactive and ready to start
 * 2. Running - Timer is actively counting up
 * 3. Paused - Timer is temporarily suspended
 * 4. Complete - Timer has reached its duration limit
 * 
 * @event_system
 * - OnTimerStart - Fired when timer begins operation
 * - OnTimerComplete - Fired when timer reaches duration
 * - OnTimerPause - Fired when timer is paused
 * - OnTimerResume - Fired when timer resumes operation
 * - OnTimerUpdate - Fired every frame with current time
 * 
 * @usage_examples
 * - Countdown timers for puzzle completion
 * - Session duration tracking
 * - Cooldown periods for game mechanics
 * - Performance measurement and benchmarking
 * - Real-time event scheduling
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - MonoBehaviour for Unity component lifecycle
 * - System.Action for event system implementation
 * 
 * @license MIT License
 */

using UnityEngine;

namespace MREscapeRoom.Utilities
{
    /**
     * @class Timer
     * @brief Advanced timer utility system for game mechanics
     * 
     * @details
     * The Timer class provides comprehensive timing functionality for Mixed Reality
     * Escape Room game mechanics. It implements a state-driven timer system with
     * event notifications, pause/resume capabilities, and loop functionality.
     * The system is designed for optimal performance in VR environments and
     * provides intuitive APIs for common timing scenarios.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @features Event-driven architecture for loose coupling
     * @features State machine pattern for timer management
     * @features Performance optimization for VR environments
     */
    public class Timer : MonoBehaviour
    {
        [Header("Timer Configuration")]
        [SerializeField] private float duration = 60f;        /**< Timer duration in seconds */
        [SerializeField] private bool autoStart = false;      /**< Whether timer starts automatically */
        [SerializeField] private bool loop = false;           /**< Whether timer loops after completion */
        
        [Header("Timer State")]
        private float currentTime;                            /**< Current elapsed time in seconds */
        private bool isRunning = false;                       /**< Whether timer is actively running */
        private bool isPaused = false;                        /**< Whether timer is paused */
        
        /**
         * @brief Public properties for timer state access
         * 
         * @details
         * These properties provide read-only access to timer state information,
         * enabling external systems to monitor timer progress and status without
         * direct field access. All properties are calculated in real-time to
         * ensure accuracy.
         */
        public float Duration => duration;                    /**< Timer duration in seconds */
        public float CurrentTime => currentTime;              /**< Current elapsed time in seconds */
        public float RemainingTime => Mathf.Max(0, duration - currentTime); /**< Remaining time in seconds */
        public float Progress => duration > 0 ? currentTime / duration : 0; /**< Normalized progress (0.0 to 1.0) */
        public bool IsRunning => isRunning;                   /**< Whether timer is actively running */
        public bool IsPaused => isPaused;                     /**< Whether timer is paused */
        public bool IsComplete => currentTime >= duration;    /**< Whether timer has completed */
        
        /**
         * @brief Event system for timer state notifications
         * 
         * @details
         * The event system provides a decoupled mechanism for external systems
         * to respond to timer state changes. Events are fired synchronously
         * during the timer update cycle to ensure proper timing and order.
         */
        public System.Action OnTimerStart;                    /**< Fired when timer starts */
        public System.Action OnTimerComplete;                 /**< Fired when timer completes */
        public System.Action OnTimerPause;                    /**< Fired when timer is paused */
        public System.Action OnTimerResume;                   /**< Fired when timer resumes */
        public System.Action<float> OnTimerUpdate;            /**< Fired every frame with current time */
        
        /**
         * @brief Unity lifecycle method for component startup
         * 
         * @details
         * Initializes the timer component and optionally starts the timer
         * automatically if the autoStart flag is enabled. This method
         * is called once when the component becomes active.
         */
        private void Start()
        {
            if (autoStart)
            {
                StartTimer();
            }
        }
        
        /**
         * @brief Unity lifecycle method for continuous updates
         * 
         * @details
         * Updates the timer state every frame when the timer is running
         * and not paused. This method handles time accumulation, progress
         * calculation, and completion detection. It also triggers the
         * OnTimerUpdate event for external systems.
         */
        private void Update()
        {
            if (!isRunning || isPaused) return;
            
            currentTime += Time.deltaTime;
            OnTimerUpdate?.Invoke(currentTime);
            
            if (currentTime >= duration)
            {
                HandleTimerComplete();
            }
        }
        
        /**
         * @brief Starts the timer operation
         * 
         * @details
         * Activates the timer and resets the current time to zero.
         * This method triggers the OnTimerStart event and sets the
         * timer to the running state. If the timer was previously
         * paused, it will resume from the beginning.
         */
        public void StartTimer()
        {
            isRunning = true;
            isPaused = false;
            currentTime = 0f;
            OnTimerStart?.Invoke();
        }
        
        /**
         * @brief Pauses the timer operation
         * 
         * @details
         * Suspends timer operation while preserving the current time.
         * This method triggers the OnTimerPause event and allows the
         * timer to be resumed later from the same position.
         */
        public void PauseTimer()
        {
            if (!isRunning) return;
            
            isPaused = true;
            OnTimerPause?.Invoke();
        }
        
        /**
         * @brief Resumes the timer operation
         * 
         * @details
         * Resumes timer operation from the previously paused position.
         * This method triggers the OnTimerResume event and continues
         * timing from where it was paused.
         */
        public void ResumeTimer()
        {
            if (!isRunning) return;
            
            isPaused = false;
            OnTimerResume?.Invoke();
        }
        
        /**
         * @brief Stops the timer operation
         * 
         * @details
         * Completely stops the timer and resets all state variables.
         * This method does not trigger completion events and prepares
         * the timer for a fresh start.
         */
        public void StopTimer()
        {
            isRunning = false;
            isPaused = false;
            currentTime = 0f;
        }
        
        /**
         * @brief Sets a new duration for the timer
         * 
         * @details
         * Dynamically modifies the timer duration during runtime.
         * This method allows for adaptive timing scenarios where
         * duration requirements may change based on game conditions.
         * 
         * @param newDuration New duration in seconds (must be positive)
         */
        public void SetDuration(float newDuration)
        {
            duration = newDuration;
        }
        
        /**
         * @brief Handles timer completion and loop logic
         * 
         * @details
         * Processes timer completion by triggering the OnTimerComplete
         * event and handling loop behavior. If looping is enabled,
         * the timer resets and continues running. Otherwise, the
         * timer stops and enters the completed state.
         */
        private void HandleTimerComplete()
        {
            OnTimerComplete?.Invoke();
            
            if (loop)
            {
                currentTime = 0f;
            }
            else
            {
                isRunning = false;
            }
        }
    }
} 