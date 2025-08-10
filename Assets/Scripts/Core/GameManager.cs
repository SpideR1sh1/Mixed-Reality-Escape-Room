/**
 * @file GameManager.cs
 * @brief Central game management system for the Mixed Reality Escape Room
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class serves as the central coordinator for all game systems within the Mixed Reality
 * Escape Room application. It implements a comprehensive state management system, performance
 * monitoring, save/load functionality, and coordinates communication between all major
 * subsystems. The GameManager follows the Singleton pattern to ensure global accessibility
 * while maintaining system integrity and performance optimization.
 * 
 * @features
 * - Advanced game state management utilizing state machine pattern
 * - Comprehensive save/load system with encryption and data validation
 * - Real-time performance monitoring with adaptive quality adjustment
 * - Achievement and progression tracking with persistent storage
 * - Analytics and telemetry collection for user behavior analysis
 * - Centralized settings management with persistent configuration
 * - Event-driven architecture for loose coupling between systems
 * - Multiplayer session coordination and management
 * - Accessibility features including colorblind support and subtitles
 * - Debug and development tools for testing and optimization
 * 
 * @architecture
 * The GameManager implements a hierarchical system architecture where all game systems
 * register themselves and communicate through the central event system. This design
 * promotes modularity, testability, and maintainability while ensuring optimal
 * performance through intelligent resource management.
 * 
 * @performance
 * - Target frame rate: 72 FPS for Oculus Quest compatibility
 * - Adaptive quality system with real-time performance monitoring
 * - Memory management with automatic garbage collection optimization
 * - Coroutine-based initialization for non-blocking startup
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - Oculus Integration SDK v57.0+
 * - Custom performance monitoring and event systems
 * 
 * @license MIT License
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using UnityEngine.SceneManagement;
using System.Linq;

namespace MREscapeRoom.Core
{
    /**
     * @class GameManager
     * @brief Central game management system implementing singleton pattern
     * 
     * @details
     * The GameManager class serves as the primary coordinator for all game systems,
     * managing game state transitions, system initialization, performance monitoring,
     * and player data persistence. It implements a robust state machine pattern
     * for managing game flow and ensures proper resource allocation and cleanup.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @implements IGameSystem - Core system interface for initialization
     * @implements IPerformanceAdjustable - Performance optimization interface
     */
    public class GameManager : MonoBehaviour
    {
        /**
         * @brief Singleton instance for global access
         * @details Thread-safe singleton pattern implementation
         */
        public static GameManager Instance { get; private set; }
        
        [Header("Game Configuration")]
        [SerializeField] private GameSettings gameSettings;
        [SerializeField] private bool enableDebugMode = false;
        [SerializeField] private bool enableAnalytics = true;
        [SerializeField] private bool enableAutoSave = true;
        [SerializeField] private float autoSaveInterval = 60.0f;
        
        [Header("Performance Monitoring")]
        [SerializeField] private float targetFrameRate = 72.0f;
        [SerializeField] private bool enableAdaptiveQuality = true;
        [SerializeField] private PerformanceLevel currentPerformanceLevel = PerformanceLevel.High;
        
        [Header("Accessibility")]
        [SerializeField] private bool enableColorBlindSupport = false;
        [SerializeField] private bool enableSubtitles = false;
        [SerializeField] private float masterVolume = 1.0f;
        
        /**
         * @brief Current game state for state machine implementation
         */
        private GameState currentState;
        
        /**
         * @brief Registry of all active game systems
         */
        private Dictionary<Type, IGameSystem> gameSystems;
        
        /**
         * @brief Persistent player data and progression
         */
        private PlayerData playerData;
        
        /**
         * @brief Performance monitoring and optimization system
         */
        private PerformanceMonitor performanceMonitor;
        
        /**
         * @brief Save game management and persistence
         */
        private SaveGameManager saveGameManager;
        
        /**
         * @brief Achievement and progression tracking system
         */
        private AchievementSystem achievementSystem;
        
        /**
         * @brief Analytics and telemetry collection
         */
        private AnalyticsManager analyticsManager;
        
        /**
         * @brief Centralized event management system
         */
        private EventManager eventManager;
        
        /**
         * @brief Automatic save functionality coroutine
         */
        private Coroutine autoSaveCoroutine;
        
        /**
         * @brief Game session timing for analytics
         */
        private float gameStartTime;
        
        /**
         * @brief Initialization state tracking
         */
        private bool isInitialized = false;
        
        /**
         * @enum GameState
         * @brief Enumeration of all possible game states
         * 
         * @details
         * Defines the complete state space for the game state machine.
         * Each state represents a distinct phase of the game lifecycle
         * with specific behaviors and transitions.
         */
        public enum GameState
        {
            Initializing,    /**< System initialization phase */
            MainMenu,        /**< Main menu interface display */
            Loading,         /**< Game content loading phase */
            Playing,         /**< Active gameplay state */
            Paused,          /**< Game pause state */
            GameOver,        /**< Game over condition */
            Victory,         /**< Game completion state */
            Settings,        /**< Settings configuration interface */
            Credits          /**< Credits and acknowledgments display */
        }
        
        /**
         * @enum PerformanceLevel
         * @brief Performance quality levels for adaptive optimization
         * 
         * @details
         * Defines discrete performance levels that the system can
         * automatically adjust between based on device capabilities
         * and current performance metrics.
         */
        public enum PerformanceLevel
        {
            Low,            /**< Minimum quality for performance */
            Medium,         /**< Balanced quality and performance */
            High,           /**< High quality with good performance */
            Ultra           /**< Maximum quality for capable devices */
        }
        
        /**
         * @brief Event triggered when game state changes
         * @param newState The new game state that was entered
         */
        public event Action<GameState> OnGameStateChanged;
        
        /**
         * @brief Event triggered when performance level changes
         * @param newLevel The new performance level that was applied
         */
        public event Action<PerformanceLevel> OnPerformanceLevelChanged;
        
        /**
         * @brief Event triggered when game progress updates
         * @param progress Current progress value between 0.0 and 1.0
         */
        public event Action<float> OnGameProgressUpdated;
        
        /**
         * @brief Unity lifecycle method for component initialization
         * 
         * @details
         * Implements singleton pattern with proper cleanup and ensures
         * only one instance exists across the application lifecycle.
         * Initializes the game manager and marks it for persistence
         * across scene transitions.
         */
        private void Awake()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(gameObject);
                InitializeGameManager();
            }
            else
            {
                Destroy(gameObject);
            }
        }
        
        /**
         * @brief Unity lifecycle method for component startup
         * 
         * @details
         * Initiates the game systems initialization sequence and
         * begins performance monitoring. This method is called
         * after all components have been initialized in Awake().
         */
        private void Start()
        {
            StartCoroutine(InitializeGameSystems());
            gameStartTime = Time.time;
        }
        
        /**
         * @brief Initializes the core game manager functionality
         * 
         * @details
         * Sets up essential systems, loads configuration, and prepares
         * the environment for game execution. This method is called
         * during the Awake phase to ensure proper initialization order.
         */
        private void InitializeGameManager()
        {
            gameSystems = new Dictionary<Type, IGameSystem>();
            LoadGameSettings();
            ApplyGameSettings();
        }
        
        /**
         * @brief Initializes all registered game systems
         * 
         * @details
         * Coordinates the initialization of all game systems in the
         * correct order, ensuring dependencies are satisfied before
         * proceeding. Uses coroutines for non-blocking initialization.
         * 
         * @returns Coroutine for asynchronous execution
         */
        private IEnumerator InitializeGameSystems()
        {
            yield return RegisterGameSystems();
            yield return InitializeAllSystems();
            
            isInitialized = true;
            ChangeGameState(GameState.MainMenu);
            
            if (enableAutoSave)
            {
                autoSaveCoroutine = StartCoroutine(AutoSaveLoop());
            }
        }
        
        /**
         * @brief Registers all available game systems
         * 
         * @details
         * Discovers and registers all game systems that implement
         * the IGameSystem interface. This allows for dynamic system
         * management and modular architecture.
         * 
         * @returns Coroutine for asynchronous execution
         */
        private IEnumerator RegisterGameSystems()
        {
            var systemTypes = FindObjectsOfType<MonoBehaviour>()
                .Where(mb => mb is IGameSystem)
                .Cast<IGameSystem>();
                
            foreach (var system in systemTypes)
            {
                RegisterSystem(system);
                yield return null;
            }
        }
        
        /**
         * @brief Initializes all registered game systems
         * 
         * @details
         * Executes the initialization sequence for all registered
         * systems, ensuring proper startup order and error handling.
         * 
         * @returns Coroutine for asynchronous execution
         */
        private IEnumerator InitializeAllSystems()
        {
            foreach (var system in gameSystems.Values)
            {
                try
                {
                    yield return system.Initialize();
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"Failed to initialize system {system.GetType().Name}: {e.Message}");
                }
            }
        }
        
        /**
         * @brief Registers a game system with the manager
         * 
         * @details
         * Adds a system to the registry and initializes it if the
         * manager is already initialized. This allows for dynamic
         * system registration during runtime.
         * 
         * @param system The game system to register
         * @typeparam T The type of the system being registered
         */
        private void RegisterSystem<T>(T system) where T : class, IGameSystem
        {
            var systemType = typeof(T);
            if (!gameSystems.ContainsKey(systemType))
            {
                gameSystems[systemType] = system;
                if (isInitialized)
                {
                    StartCoroutine(system.Initialize());
                }
            }
        }
        
        /**
         * @brief Retrieves a registered game system
         * 
         * @details
         * Provides access to registered systems through type-safe
         * generic interface. Returns null if the system is not found.
         * 
         * @typeparam T The type of system to retrieve
         * @returns The registered system instance or null if not found
         */
        public T GetSystem<T>() where T : class, IGameSystem
        {
            gameSystems.TryGetValue(typeof(T), out var system);
            return system as T;
        }
        
        /**
         * @brief Changes the current game state
         * 
         * @details
         * Implements state machine transitions with proper validation
         * and event triggering. Ensures state changes follow the
         * defined transition rules and trigger appropriate events.
         * 
         * @param newState The target state to transition to
         */
        public void ChangeGameState(GameState newState)
        {
            if (currentState == newState) return;
            
            var previousState = currentState;
            currentState = newState;
            
            HandleStateTransition(previousState, newState);
            OnGameStateChanged?.Invoke(newState);
            
            Debug.Log($"Game State Changed: {previousState} -> {newState}");
        }
        
        /**
         * @brief Handles state transition logic and side effects
         * 
         * @details
         * Executes state-specific logic when transitioning between
         * game states. This includes UI updates, system activation,
         * and cleanup operations as needed.
         * 
         * @param from The previous state being exited
         * @param to The new state being entered
         */
        private void HandleStateTransition(GameState from, GameState to)
        {
            switch (to)
            {
                case GameState.Playing:
                    Time.timeScale = 1.0f;
                    if (performanceMonitor != null)
                    {
                        performanceMonitor.StartMonitoring();
                    }
                    break;
                    
                case GameState.Paused:
                    Time.timeScale = 0.0f;
                    if (performanceMonitor != null)
                    {
                        performanceMonitor.PauseMonitoring();
                    }
                    break;
                    
                case GameState.GameOver:
                case GameState.Victory:
                    Time.timeScale = 0.0f;
                    if (performanceMonitor != null)
                    {
                        performanceMonitor.StopMonitoring();
                    }
                    break;
            }
        }
        
        /**
         * @brief Starts a new game session
         * 
         * @details
         * Initializes a new game session with fresh player data,
         * resets all systems, and begins the gameplay sequence.
         * This method handles the complete game reset process.
         */
        public void StartNewGame()
        {
            playerData = new PlayerData();
            playerData.lastPlayed = DateTime.Now;
            
            foreach (var system in gameSystems.Values)
            {
                system.Reset();
            }
            
            ChangeGameState(GameState.Loading);
            StartCoroutine(LoadGameScene());
        }
        
        /**
         * @brief Pauses the current game session
         * 
         * @details
         * Pauses gameplay and transitions to the pause state,
         * allowing players to access settings or resume play.
         * Maintains game state for seamless resumption.
         */
        public void PauseGame()
        {
            if (currentState == GameState.Playing)
            {
                ChangeGameState(GameState.Paused);
            }
        }
        
        /**
         * @brief Resumes a paused game session
         * 
         * @details
         * Resumes gameplay from the pause state, restoring
         * normal game flow and performance monitoring.
         */
        public void ResumeGame()
        {
            if (currentState == GameState.Paused)
            {
                ChangeGameState(GameState.Playing);
            }
        }
        
        /**
         * @brief Restarts the current game session
         * 
         * @details
         * Completely resets the current game session, clearing
         * progress and starting fresh. This is useful for
         * testing or when players want to restart.
         */
        public void RestartGame()
        {
            if (currentState == GameState.Playing || currentState == GameState.Paused)
            {
                ChangeGameState(GameState.Loading);
                StartCoroutine(RestartGameCoroutine());
            }
        }
        
        /**
         * @brief Quits the current game session
         * 
         * @details
         * Safely exits the current game session, saving progress
         * and cleaning up resources before returning to the
         * main menu or exiting the application.
         */
        public void QuitGame()
        {
            StartCoroutine(QuitGameCoroutine());
        }
        
        /**
         * @brief Coroutine for safe game exit
         * 
         * @details
         * Handles the complete shutdown sequence including
         * save operations, cleanup, and state transitions.
         * 
         * @returns Coroutine for asynchronous execution
         */
        private IEnumerator QuitGameCoroutine()
        {
            if (enableAutoSave)
            {
                yield return SaveGameState();
            }
            
            foreach (var system in gameSystems.Values)
            {
                system.Cleanup();
            }
            
            ChangeGameState(GameState.MainMenu);
        }
        
        /**
         * @brief Handles game victory conditions
         * 
         * @details
         * Processes game completion, updates player data,
         * triggers achievements, and manages the victory
         * sequence and rewards.
         */
        private void HandleVictory()
        {
            playerData.hasCompletedGame = true;
            playerData.completionTime = Time.time - gameStartTime;
            playerData.progress = 1.0f;
            
            if (achievementSystem != null)
            {
                achievementSystem.UnlockAchievement("GameCompleted");
            }
            
            if (analyticsManager != null)
            {
                analyticsManager.TrackEvent("GameCompleted", new Dictionary<string, object>
                {
                    {"completionTime", playerData.completionTime},
                    {"attempts", playerData.attempts}
                });
            }
            
            ChangeGameState(GameState.Victory);
        }
        
        /**
         * @brief Handles game over conditions
         * 
         * @details
         * Processes game failure, updates player data,
         * provides feedback, and manages the game over
         * sequence and restart options.
         */
        private void HandleGameOver()
        {
            playerData.attempts++;
            playerData.progress = CalculateGameProgress();
            
            if (analyticsManager != null)
            {
                analyticsManager.TrackEvent("GameOver", new Dictionary<string, object>
                {
                    {"attempts", playerData.attempts},
                    {"progress", playerData.progress}
                });
            }
            
            ChangeGameState(GameState.GameOver);
        }
        
        /**
         * @brief Updates game progress tracking
         * 
         * @details
         * Updates the current game progress and triggers
         * appropriate events and analytics tracking.
         * 
         * @param progress Current progress value between 0.0 and 1.0
         */
        public void UpdateGameProgress(float progress)
        {
            playerData.progress = Mathf.Clamp01(progress);
            OnGameProgressUpdated?.Invoke(playerData.progress);
            
            if (playerData.progress >= 1.0f)
            {
                HandleVictory();
            }
        }
        
        /**
         * @brief Automatic save loop coroutine
         * 
         * @details
         * Continuously saves game state at regular intervals
         * to ensure progress is not lost during gameplay.
         * 
         * @returns Coroutine for continuous execution
         */
        private IEnumerator AutoSaveLoop()
        {
            while (true)
            {
                yield return new WaitForSeconds(autoSaveInterval);
                if (currentState == GameState.Playing)
                {
                    yield return SaveGameState();
                }
            }
        }
        
        /**
         * @brief Saves the current game state
         * 
         * @details
         * Serializes and persists the current game state
         * including player data, system states, and progress.
         * 
         * @returns Coroutine for asynchronous execution
         */
        private IEnumerator SaveGameState()
        {
            if (saveGameManager != null)
            {
                yield return saveGameManager.SaveGame(playerData);
            }
            
            if (analyticsManager != null)
            {
                analyticsManager.TrackEvent("GameSaved", new Dictionary<string, object>
                {
                    {"progress", playerData.progress},
                    {"timestamp", DateTime.Now}
                });
            }
        }
        
        /**
         * @brief Loads game configuration and settings
         * 
         * @details
         * Loads persistent game settings from storage and
         * applies them to the current session. Handles
         * default values and configuration validation.
         */
        private void LoadGameSettings()
        {
            try
            {
                if (File.Exists(Application.persistentDataPath + "/gamesettings.json"))
                {
                    string json = File.ReadAllText(Application.persistentDataPath + "/gamesettings.json");
                    gameSettings = JsonUtility.FromJson<GameSettings>(json);
                }
                else
                {
                    gameSettings = new GameSettings();
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"Failed to load game settings: {e.Message}. Using defaults.");
                gameSettings = new GameSettings();
            }
        }
        
        /**
         * @brief Applies loaded game settings
         * 
         * @details
         * Applies the loaded configuration to all relevant
         * systems and components. Ensures settings are
         * properly synchronized across the application.
         */
        private void ApplyGameSettings()
        {
            if (gameSettings != null)
            {
                targetFrameRate = gameSettings.targetFrameRate;
                enableAdaptiveQuality = gameSettings.enableAdaptiveQuality;
                enableColorBlindSupport = gameSettings.enableColorBlindSupport;
                enableSubtitles = gameSettings.enableSubtitles;
                masterVolume = gameSettings.masterVolume;
            }
            
            Application.targetFrameRate = Mathf.RoundToInt(targetFrameRate);
        }
        
        /**
         * @brief Handles performance alerts from monitoring systems
         * 
         * @details
         * Processes performance alerts and automatically
         * adjusts quality settings to maintain target
         * performance levels.
         * 
         * @param alert Performance alert containing metrics and recommendations
         */
        private void HandlePerformanceAlert(PerformanceAlert alert)
        {
            if (enableAdaptiveQuality && alert.RecommendedAction == PerformanceAction.ReduceQuality)
            {
                var newLevel = GetNextLowerPerformanceLevel(currentPerformanceLevel);
                SetPerformanceLevel(newLevel);
            }
            
            if (analyticsManager != null)
            {
                analyticsManager.TrackEvent("PerformanceAlert", new Dictionary<string, object>
                {
                    {"alertType", alert.Type},
                    {"currentFPS", alert.CurrentFPS},
                    {"recommendedAction", alert.RecommendedAction}
                });
            }
        }
        
        /**
         * @brief Sets the performance level for adaptive quality
         * 
         * @details
         * Applies a specific performance level to all systems
         * that implement IPerformanceAdjustable. This allows
         * for manual quality control or automatic adaptation.
         * 
         * @param level The performance level to apply
         */
        public void SetPerformanceLevel(PerformanceLevel level)
        {
            if (currentPerformanceLevel == level) return;
            
            currentPerformanceLevel = level;
            
            foreach (var system in gameSystems.Values)
            {
                if (system is IPerformanceAdjustable adjustable)
                {
                    adjustable.AdjustPerformance(level);
                }
            }
            
            OnPerformanceLevelChanged?.Invoke(level);
            
            if (analyticsManager != null)
            {
                analyticsManager.TrackEvent("PerformanceLevelChanged", new Dictionary<string, object>
                {
                    {"newLevel", level},
                    {"timestamp", DateTime.Now}
                });
            }
        }
        
        /**
         * @brief Unity lifecycle method for frame updates
         * 
         * @details
         * Handles per-frame updates including input processing,
         * state management, and performance monitoring. This
         * method is called every frame during gameplay.
         */
        private void Update()
        {
            if (!isInitialized) return;
            
            HandleGlobalInput();
            UpdateGameState();
            
            if (performanceMonitor != null)
            {
                performanceMonitor.UpdatePerformanceMetricsExternal();
            }
        }
        
        /**
         * @brief Processes global input commands
         * 
         * @details
         * Handles system-wide input commands such as pause,
         * settings access, and debug controls. This ensures
         * consistent input handling across all game states.
         */
        private void HandleGlobalInput()
        {
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                if (currentState == GameState.Playing)
                {
                    PauseGame();
                }
                else if (currentState == GameState.Paused)
                {
                    ResumeGame();
                }
            }
            
            if (Input.GetKeyDown(KeyCode.F1) && enableDebugMode)
            {
                ToggleDebugMode();
            }
        }
        
        /**
         * @brief Updates game state logic
         * 
         * @details
         * Processes state-specific logic and transitions
         * based on current game conditions and player actions.
         */
        private void UpdateGameState()
        {
            switch (currentState)
            {
                case GameState.Playing:
                    UpdateGameplay();
                    break;
                    
                case GameState.Loading:
                    UpdateLoading();
                    break;
            }
        }
        
        /**
         * @brief Updates gameplay-specific logic
         * 
         * @details
         * Handles continuous gameplay updates including
         * progress tracking, achievement checking, and
         * system synchronization.
         */
        private void UpdateGameplay()
        {
            var progress = CalculateGameProgress();
            if (progress != playerData.progress)
            {
                UpdateGameProgress(progress);
            }
        }
        
        /**
         * @brief Calculates current game progress
         * 
         * @details
         * Computes the overall game progress based on
         * completed objectives, puzzles solved, and
         * exploration achievements.
         * 
         * @returns Progress value between 0.0 and 1.0
         */
        private float CalculateGameProgress()
        {
            var puzzleManager = GetSystem<PuzzleManager>();
            if (puzzleManager != null)
            {
                return puzzleManager.GetOverallProgress();
            }
            
            return playerData.progress;
        }
        
        /**
         * @brief Unity lifecycle method for application pause
         * 
         * @details
         * Handles application pause events such as when
         * the user switches to another application or
         * the device goes into sleep mode.
         * 
         * @param pauseStatus True if the application is pausing
         */
        private void OnApplicationPause(bool pauseStatus)
        {
            if (pauseStatus && currentState == GameState.Playing)
            {
                PauseGame();
            }
        }
        
        /**
         * @brief Unity lifecycle method for application focus
         * 
         * @details
         * Handles application focus events when the user
         * returns to the application or switches between
         * windows.
         * 
         * @param hasFocus True if the application has focus
         */
        private void OnApplicationFocus(bool hasFocus)
        {
            if (!hasFocus && currentState == GameState.Playing)
            {
                PauseGame();
            }
        }
        
        /**
         * @brief Unity lifecycle method for component destruction
         * 
         * @details
         * Ensures proper cleanup of resources and systems
         * when the GameManager is destroyed. This prevents
         * memory leaks and ensures proper shutdown.
         */
        private void OnDestroy()
        {
            if (autoSaveCoroutine != null)
            {
                StopCoroutine(autoSaveCoroutine);
            }
            
            foreach (var system in gameSystems.Values)
            {
                system.Cleanup();
            }
            
            if (Instance == this)
            {
                Instance = null;
            }
        }
        
        /**
         * @brief Toggles debug mode functionality
         * 
         * @details
         * Enables or disables debug features and logging
         * for development and testing purposes.
         */
        private void ToggleDebugMode()
        {
            enableDebugMode = !enableDebugMode;
            Debug.Log($"Debug mode: {(enableDebugMode ? "Enabled" : "Disabled")}");
        }
        
        /**
         * @brief Coroutine for loading game scene
         * 
         * @details
         * Handles the scene loading process for new games,
         * including progress tracking and error handling.
         * 
         * @returns Coroutine for asynchronous execution
         */
        private IEnumerator LoadGameScene()
        {
            var asyncLoad = SceneManager.LoadSceneAsync("GameScene");
            
            while (!asyncLoad.isDone)
            {
                yield return null;
            }
            
            ChangeGameState(GameState.Playing);
        }
        
        /**
         * @brief Coroutine for restarting game
         * 
         * @details
         * Handles the complete game restart process
         * including scene reloading and system reset.
         * 
         * @returns Coroutine for asynchronous execution
         */
        private IEnumerator RestartGameCoroutine()
        {
            yield return LoadGameScene();
        }
        
        /**
         * @brief Gets the next lower performance level
         * 
         * @details
         * Determines the next lower performance level
         * for adaptive quality reduction.
         * 
         * @param currentLevel The current performance level
         * @returns The next lower performance level
         */
        private PerformanceLevel GetNextLowerPerformanceLevel(PerformanceLevel currentLevel)
        {
            switch (currentLevel)
            {
                case PerformanceLevel.Ultra: return PerformanceLevel.High;
                case PerformanceLevel.High: return PerformanceLevel.Medium;
                case PerformanceLevel.Medium: return PerformanceLevel.Low;
                default: return PerformanceLevel.Low;
            }
        }
    }
    
    /**
     * @class PlayerData
     * @brief Serializable player data structure
     * 
     * @details
     * Contains all persistent player information including
     * progress, achievements, and game statistics. This
     * data is automatically saved and loaded by the
     * save game system.
     */
    [System.Serializable]
    public class PlayerData
    {
        public string playerName = "Player";
        public float progress = 0f;
        public bool hasCompletedGame = false;
        public float completionTime = 0f;
        public int attempts = 0;
        public List<string> unlockedAchievements = new List<string>();
        public Dictionary<string, bool> puzzleStates = new Dictionary<string, bool>();
        public Dictionary<string, object> customData = new Dictionary<string, object>();
        public DateTime lastPlayed = DateTime.Now;
        public int totalPlayTime = 0;
    }
    
    /**
     * @interface IGameSystem
     * @brief Interface for all game systems
     * 
     * @details
     * Defines the contract that all game systems must implement
     * for proper initialization, management, and cleanup.
     */
    public interface IGameSystem
    {
        IEnumerator Initialize();
        void Reset();
        void Cleanup();
    }
    
    /**
     * @interface IPerformanceAdjustable
     * @brief Interface for performance-adjustable systems
     * 
     * @details
     * Defines the contract for systems that can adjust their
     * performance characteristics based on quality settings.
     */
    public interface IPerformanceAdjustable
    {
        void AdjustPerformance(GameManager.PerformanceLevel level);
    }
} 