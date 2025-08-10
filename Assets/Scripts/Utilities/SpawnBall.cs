/**
 * @file SpawnBall.cs
 * @brief Ball spawning utility system for Mixed Reality physics demonstrations
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements a ball spawning utility system designed for Mixed Reality
 * physics demonstrations and interactive experiences. It provides real-time ball
 * instantiation with configurable spawn speed and direction, enabling users to
 * create dynamic physics objects through VR controller input. The system is
 * optimized for VR environments and supports various ball prefabs for different
 * demonstration scenarios.
 * 
 * @features
 * - Real-time ball spawning through VR controller input
 * - Configurable spawn speed for physics demonstrations
 * - Automatic rigidbody velocity assignment for realistic motion
 * - Forward-direction spawning based on object orientation
 * - Instant prefab instantiation with proper physics setup
 * - VR-optimized input handling using Oculus Integration SDK
 * - Support for various ball prefab types and configurations
 * - Performance-optimized spawning for smooth VR experiences
 * 
 * @spawning_mechanics
 * The ball spawning system operates through the following process:
 * 1. Controller input detection using OVRInput system
 * 2. Prefab instantiation at the spawner's position
 * 3. Rigidbody component retrieval and velocity assignment
 * 4. Forward-direction velocity application based on spawn speed
 * 5. Physics-based ball motion and interaction
 * 
 * @input_system
 * - Primary trigger button (IndexTrigger) for ball spawning
 * - Real-time input detection for responsive spawning
 * - Configurable spawn speed through inspector interface
 * - Automatic velocity calculation and application
 * 
 * @physics_integration
 * - Rigidbody component detection and manipulation
 * - Velocity-based motion for realistic physics behavior
 * - Forward-direction spawning for intuitive control
 * - Configurable spawn speed for demonstration flexibility
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - Oculus Integration SDK for VR input handling
 * - Rigidbody component for physics simulation
 * - GameObject prefab system for ball instantiation
 * 
 * @license MIT License
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace MREscapeRoom.Utilities
{
    /**
     * @class SpawnBall
     * @brief Ball spawning utility system for physics demonstrations
     * 
     * @details
     * The SpawnBall class provides functionality for real-time ball spawning
     * in Mixed Reality environments. It enables users to create physics objects
     * through VR controller input, with configurable spawn speed and automatic
     * velocity assignment for realistic motion simulation.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @requires GameObject prefab - For ball instantiation
     * @requires Rigidbody - For physics simulation and motion
     * @requires OVRInput - For VR controller input handling
     */
    public class SpawnBall : MonoBehaviour
    {
        [Header("Spawn Configuration")]
        [SerializeField] public GameObject prefab;            /**< Ball prefab to spawn */
        [SerializeField] public float spawnSpeed = 5f;        /**< Initial velocity magnitude for spawned balls */
        
        /**
         * @brief Unity lifecycle method for component startup
         * 
         * @details
         * Initializes the ball spawning component. This method is called once
         * when the component becomes active and prepares the system for
         * ball spawning operations.
         */
        void Start()
        {
            // Component initialization completed
        }

        /**
         * @brief Unity lifecycle method for continuous updates
         * 
         * @details
         * Monitors VR controller input for ball spawning requests and handles
         * the spawning process when the primary index trigger is pressed.
         * This method runs every frame to ensure responsive input handling.
         * 
         * @process
         * 1. Detects primary index trigger button press
         * 2. Instantiates ball prefab at spawner position
         * 3. Retrieves rigidbody component from spawned ball
         * 4. Applies forward velocity based on spawn speed
         * 5. Enables physics-based motion and interaction
         */
        void Update()
        {
            // Check for primary index trigger button press
            if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger))
            {
                // Spawn ball prefab at current position with default rotation
                GameObject spawnedBall = Instantiate(prefab, transform.position, Quaternion.identity);
                
                // Retrieve rigidbody component for physics manipulation
                Rigidbody spawnedBallRB = spawnedBall.GetComponent<Rigidbody>();
                
                // Apply forward velocity based on spawn speed and object orientation
                spawnedBallRB.velocity = transform.forward * spawnSpeed;
            }
        }
    }
}

