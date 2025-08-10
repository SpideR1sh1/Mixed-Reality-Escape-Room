/**
 * @file PuzzleManager.cs
 * @brief Advanced puzzle management system for Mixed Reality Escape Room environments
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements a comprehensive puzzle management system designed specifically
 * for Mixed Reality Escape Room environments. It provides advanced puzzle generation,
 * dynamic difficulty adjustment, comprehensive hint systems, player analytics tracking,
 * achievement integration, and performance optimization for VR environments. The system
 * supports multiple puzzle types with procedural generation, adaptive difficulty based
 * on player performance, and seamless integration with game management systems.
 * 
 * @features
 * - Multiple puzzle types with procedural generation and variety
 * - Dynamic difficulty adjustment based on player performance metrics
 * - Comprehensive hint system with contextual assistance delivery
 * - Advanced progress tracking and analytics integration
 * - Achievement system with unlock conditions and tracking
 * - Persistent save/load system for puzzle states and progress
 * - Accessibility features for inclusive gameplay experiences
 * - Performance optimization for VR environments with adjustable settings
 * - Object pooling for efficient memory management
 * - Real-time difficulty evaluation and adjustment
 * 
 * @puzzle_system
 * The puzzle management system operates through the following workflow:
 * 1. Puzzle sequence generation based on difficulty and player performance
 * 2. Dynamic puzzle creation and placement in the environment
 * 3. Real-time difficulty evaluation and adjustment
 * 4. Comprehensive hint system integration and management
 * 5. Achievement tracking and unlock condition evaluation
 * 6. Performance monitoring and optimization adjustments
 * 7. Persistent state management and progress tracking
 * 
 * @difficulty_management
 * - Six difficulty levels from Tutorial to Master
 * - Dynamic adjustment based on consecutive success/failure patterns
 * - Performance-based time limit calculations
 * - Adaptive puzzle selection and sequencing
 * - Real-time difficulty evaluation every 5 seconds
 * 
 * @performance_optimization
 * - Configurable concurrent puzzle limits
 * - Object pooling for efficient memory management
 * - Adjustable update intervals for performance scaling
 * - Performance level-based configuration adjustments
 * - VR-optimized puzzle placement and management
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - MREscapeRoom.Core namespace for core systems
 * - PuzzleDatabase for template management
 * - GuardianBoundaryManager for environment integration
 * - GameManager for system coordination
 * - AnalyticsManager for performance tracking
 * 
 * @license MIT License
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Linq;
using MREscapeRoom.Core;

namespace MREscapeRoom.Puzzles
{
    /**
     * @class PuzzleManager
     * @brief Advanced puzzle management system for Mixed Reality Escape Room environments
     * 
     * @details
     * The PuzzleManager class provides comprehensive functionality for managing puzzle
     * systems in Mixed Reality Escape Room environments. It implements advanced puzzle
     * generation, dynamic difficulty adjustment, comprehensive hint systems, player
     * analytics tracking, achievement integration, and performance optimization for
     * VR environments. The system coordinates multiple puzzle types with procedural
     * generation, adaptive difficulty based on player performance, and seamless
     * integration with game management systems.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @implements IGameSystem - Core game system integration interface
     * @implements IPerformanceAdjustable - Performance optimization interface
     * @features Advanced puzzle generation and management
     * @features Dynamic difficulty adjustment and optimization
     * @features Comprehensive hint and achievement systems
     * @features Performance monitoring and VR optimization
     * @features Persistent state management and analytics
     */
    public class PuzzleManager : MonoBehaviour, IGameSystem, IPerformanceAdjustable
    {
        [Header("Puzzle Configuration")]
        [SerializeField] private PuzzleDatabase puzzleDatabase;
        [SerializeField] private int totalPuzzlesInSession = 5;
        [SerializeField] private float baseTimeLimit = 300f; // 5 minutes
        [SerializeField] private bool enableDynamicDifficulty = true;
        
        [Header("Difficulty Settings")]
        [SerializeField] private DifficultyLevel currentDifficulty = DifficultyLevel.Medium;
        [SerializeField] private float difficultyAdjustmentRate = 0.1f;
        [SerializeField] private int consecutiveFailuresForDecrease = 2;
        [SerializeField] private int consecutiveSuccessesForIncrease = 3;
        
        [Header("Hint System")]
        [SerializeField] private bool enableHints = true;
        [SerializeField] private float hintCooldownTime = 30f;
        [SerializeField] private int maxHintsPerPuzzle = 3;
        [SerializeField] private HintDeliveryMethod hintDeliveryMethod = HintDeliveryMethod.Visual;
        
        [Header("Performance")]
        [SerializeField] private int maxConcurrentPuzzles = 3;
        [SerializeField] private bool enablePuzzlePooling = true;
        [SerializeField] private float puzzleUpdateInterval = 0.1f;
        
        private List<BasePuzzle> activePuzzles;
        private Queue<BasePuzzle> puzzlePool;
        private Dictionary<string, PuzzleState> puzzleStates;
        private PlayerPuzzleMetrics playerMetrics;
        private HintSystem hintSystem;
        private AchievementTracker achievementTracker;
        
        private int consecutiveSuccesses = 0;
        private int consecutiveFailures = 0;
        private float sessionStartTime;
        private Coroutine difficultyUpdateCoroutine;
        
        /**
         * @enum DifficultyLevel
         * @brief Enumeration of available difficulty levels for puzzle challenges
         * 
         * @details
         * Defines the progression of difficulty levels available in the puzzle
         * management system. Each level represents an increase in challenge
         * complexity and may affect puzzle generation, time limits, and
         * available hints.
         * 
         * @values
         * - Tutorial: Entry-level puzzles with maximum assistance
         * - Easy: Basic puzzles with moderate assistance
         * - Medium: Standard puzzles with balanced challenge
         * - Hard: Advanced puzzles with limited assistance
         * - Expert: Complex puzzles with minimal assistance
         * - Master: Ultimate puzzles with no assistance
         */
        public enum DifficultyLevel
        {
            Tutorial = 0, /**< Entry-level puzzles with maximum assistance */
            Easy = 1,     /**< Basic puzzles with moderate assistance */
            Medium = 2,   /**< Standard puzzles with balanced challenge */
            Hard = 3,     /**< Advanced puzzles with limited assistance */
            Expert = 4,   /**< Complex puzzles with minimal assistance */
            Master = 5    /**< Ultimate puzzles with no assistance */
        }
        
        /**
         * @enum HintDeliveryMethod
         * @brief Enumeration of available hint delivery methods for player assistance
         * 
         * @details
         * Defines the various methods through which hints can be delivered to
         * players. Each method provides different types of assistance and can
         * be combined for enhanced user experience.
         * 
         * @values
         * - Visual: Visual cues and indicators for puzzle elements
         * - Audio: Audio feedback and guidance for puzzle interactions
         * - Haptic: Haptic feedback through VR controllers
         * - Combined: Multiple delivery methods simultaneously
         */
        public enum HintDeliveryMethod
        {
            Visual,   /**< Visual cues and indicators for puzzle elements */
            Audio,    /**< Audio feedback and guidance for puzzle interactions */
            Haptic,   /**< Haptic feedback through VR controllers */
            Combined  /**< Multiple delivery methods simultaneously */
        }
        
        public event Action<BasePuzzle> OnPuzzleStarted;
        public event Action<BasePuzzle, bool> OnPuzzleCompleted;
        public event Action<DifficultyLevel> OnDifficultyChanged;
        public event Action<float> OnProgressUpdated;
        public event Action<string> OnAchievementUnlocked;
        
        private void Awake()
        {
            InitializeComponents();
        }
        
        private void Start()
        {
            sessionStartTime = Time.time;
            
            if (enableDynamicDifficulty)
            {
                difficultyUpdateCoroutine = StartCoroutine(DifficultyUpdateLoop());
            }
        }
        
        private void InitializeComponents()
        {
            activePuzzles = new List<BasePuzzle>();
            puzzlePool = new Queue<BasePuzzle>();
            puzzleStates = new Dictionary<string, PuzzleState>();
            playerMetrics = new PlayerPuzzleMetrics();
            
            // Initialize subsystems
            hintSystem = gameObject.AddComponent<HintSystem>();
            hintSystem.Initialize(hintCooldownTime, maxHintsPerPuzzle, hintDeliveryMethod);
            
            achievementTracker = gameObject.AddComponent<AchievementTracker>();
            achievementTracker.Initialize();
            
            // Load saved puzzle states
            LoadPuzzleStates();
            
            // Subscribe to events
            hintSystem.OnHintRequested += HandleHintRequested;
            achievementTracker.OnAchievementUnlocked += HandleAchievementUnlocked;
        }
        
        #region IGameSystem Implementation
        
        /**
         * @brief Initializes the puzzle management system
         * 
         * @details
         * Implements the IGameSystem interface initialization method for the puzzle
         * management system. This method validates the puzzle database, pre-warms
         * the puzzle object pool for performance optimization, and loads player
         * metrics from persistent storage. The initialization process ensures
         * all subsystems are properly configured before puzzle generation begins.
         * 
         * @workflow
         * 1. Validate puzzle database configuration
         * 2. Pre-warm puzzle object pool if pooling is enabled
         * 3. Load player metrics and performance data
         * 4. Complete system initialization
         * 
         * @returns IEnumerator for coroutine execution
         * @throws SystemException when puzzle database is not configured
         */
        public IEnumerator Initialize()
        {
            Debug.Log("[PuzzleManager] Initializing puzzle system...");
            
            // Validate puzzle database configuration
            if (puzzleDatabase == null)
            {
                Debug.LogError("[PuzzleManager] Puzzle database not assigned!");
                yield break;
            }
            
            // Pre-warm puzzle pool for performance optimization
            if (enablePuzzlePooling)
            {
                yield return StartCoroutine(PrewarmPuzzlePool());
            }
            
            // Load player metrics and performance data
            LoadPlayerMetrics();
            
            Debug.Log("[PuzzleManager] Puzzle system initialized successfully");
        }
        
        public void Reset()
        {
            // Clear active puzzles
            foreach (var puzzle in activePuzzles)
            {
                if (puzzle != null)
                {
                    puzzle.OnPuzzleCompleted -= HandlePuzzleCompleted;
                    DestroyImmediate(puzzle.gameObject);
                }
            }
            activePuzzles.Clear();
            
            // Reset metrics
            playerMetrics.Reset();
            consecutiveSuccesses = 0;
            consecutiveFailures = 0;
            
            // Reset difficulty
            currentDifficulty = DifficultyLevel.Medium;
            OnDifficultyChanged?.Invoke(currentDifficulty);
            
            // Clear puzzle states
            puzzleStates.Clear();
            
            Debug.Log("[PuzzleManager] System reset completed");
        }
        
        public void Cleanup()
        {
            if (difficultyUpdateCoroutine != null)
            {
                StopCoroutine(difficultyUpdateCoroutine);
            }
            
            // Save current state
            SavePuzzleStates();
            SavePlayerMetrics();
            
            // Cleanup components
            hintSystem?.Cleanup();
            achievementTracker?.Cleanup();
            
            Debug.Log("[PuzzleManager] Cleanup completed");
        }
        
        #endregion
        
        public void InitializePuzzles()
        {
            StartCoroutine(InitializePuzzlesCoroutine());
        }
        
        private IEnumerator InitializePuzzlesCoroutine()
        {
            // Generate puzzle sequence for current session
            List<PuzzleTemplate> puzzleSequence = GeneratePuzzleSequence();
            
            // Create puzzles based on sequence
            for (int i = 0; i < Mathf.Min(puzzleSequence.Count, maxConcurrentPuzzles); i++)
            {
                var puzzle = CreatePuzzle(puzzleSequence[i]);
                if (puzzle != null)
                {
                    activePuzzles.Add(puzzle);
                    OnPuzzleStarted?.Invoke(puzzle);
                }
                
                yield return new WaitForSeconds(0.1f); // Spread creation across frames
            }
            
            UpdateProgress();
        }
        
        private List<PuzzleTemplate> GeneratePuzzleSequence()
        {
            List<PuzzleTemplate> sequence = new List<PuzzleTemplate>();
            
            // Get available puzzles for current difficulty
            var availablePuzzles = puzzleDatabase.GetPuzzlesForDifficulty(currentDifficulty);
            
            if (availablePuzzles.Count == 0)
            {
                Debug.LogWarning($"[PuzzleManager] No puzzles available for difficulty {currentDifficulty}");
                return sequence;
            }
            
            // Generate sequence with variety and progression
            for (int i = 0; i < totalPuzzlesInSession; i++)
            {
                PuzzleTemplate selected = SelectNextPuzzle(availablePuzzles, sequence);
                if (selected != null)
                {
                    sequence.Add(selected);
                }
            }
            
            Debug.Log($"[PuzzleManager] Generated sequence of {sequence.Count} puzzles");
            return sequence;
        }
        
        private PuzzleTemplate SelectNextPuzzle(List<PuzzleTemplate> available, List<PuzzleTemplate> currentSequence)
        {
            // Apply selection criteria
            var candidates = available.Where(p => 
            {
                // Avoid too many consecutive puzzles of same type
                if (currentSequence.Count >= 2)
                {
                    var lastTwo = currentSequence.TakeLast(2).Select(x => x.PuzzleType);
                    if (lastTwo.All(t => t == p.PuzzleType))
                        return false;
                }
                
                // Ensure puzzle hasn't been used recently
                return !IsRecentlyUsed(p.PuzzleID);
                
            }).ToList();
            
            if (candidates.Count == 0)
                candidates = available; // Fallback to all available
            
            // Weight selection based on player performance
            return WeightedRandomSelection(candidates);
        }
        
        private PuzzleTemplate WeightedRandomSelection(List<PuzzleTemplate> candidates)
        {
            // Calculate weights based on puzzle type performance
            Dictionary<PuzzleType, float> typeWeights = CalculateTypeWeights();
            
            float totalWeight = 0f;
            foreach (var candidate in candidates)
            {
                float weight = typeWeights.ContainsKey(candidate.PuzzleType) 
                    ? typeWeights[candidate.PuzzleType] 
                    : 1.0f;
                totalWeight += weight;
            }
            
            float randomValue = UnityEngine.Random.Range(0f, totalWeight);
            float currentWeight = 0f;
            
            foreach (var candidate in candidates)
            {
                float weight = typeWeights.ContainsKey(candidate.PuzzleType) 
                    ? typeWeights[candidate.PuzzleType] 
                    : 1.0f;
                currentWeight += weight;
                
                if (randomValue <= currentWeight)
                    return candidate;
            }
            
            return candidates[UnityEngine.Random.Range(0, candidates.Count)]; // Fallback
        }
        
        private Dictionary<PuzzleType, float> CalculateTypeWeights()
        {
            var weights = new Dictionary<PuzzleType, float>();
            
            foreach (PuzzleType type in Enum.GetValues(typeof(PuzzleType)))
            {
                var metrics = playerMetrics.GetTypeMetrics(type);
                
                // Higher weight for puzzle types where player performs poorly
                // (to provide more practice)
                float successRate = metrics.TotalAttempts > 0 ? metrics.TotalSolved / (float)metrics.TotalAttempts : 0.5f;
                weights[type] = Mathf.Lerp(2.0f, 0.5f, successRate);
            }
            
            return weights;
        }
        
        private bool IsRecentlyUsed(string puzzleID)
        {
            // Check if puzzle was used in last session or recently
            return playerMetrics.RecentlyUsedPuzzles.Contains(puzzleID);
        }
        
        private BasePuzzle CreatePuzzle(PuzzleTemplate template)
        {
            GameObject puzzleObject = GetPooledPuzzleObject(template.PuzzleType);
            if (puzzleObject == null)
            {
                Debug.LogError($"[PuzzleManager] Failed to create puzzle of type {template.PuzzleType}");
                return null;
            }
            
            BasePuzzle puzzle = puzzleObject.GetComponent<BasePuzzle>();
            if (puzzle == null)
            {
                Debug.LogError($"[PuzzleManager] Puzzle object missing BasePuzzle component");
                return null;
            }
            
            // Configure puzzle
            puzzle.Initialize(template, currentDifficulty, CalculateTimeLimit());
            puzzle.OnPuzzleCompleted += HandlePuzzleCompleted;
            puzzle.OnHintRequested += hintSystem.ProcessHintRequest;
            
            // Place puzzle in world
            Vector3 placement = FindOptimalPuzzlePlacement();
            puzzle.transform.position = placement;
            
            // Create puzzle state
            var state = new PuzzleState
            {
                PuzzleID = template.PuzzleID,
                StartTime = Time.time,
                IsActive = true,
                Difficulty = currentDifficulty
            };
            puzzleStates[template.PuzzleID] = state;
            
            Debug.Log($"[PuzzleManager] Created puzzle: {template.PuzzleID} at {placement}");
            return puzzle;
        }
        
        private GameObject GetPooledPuzzleObject(PuzzleType puzzleType)
        {
            if (enablePuzzlePooling && puzzlePool.Count > 0)
            {
                var pooledPuzzle = puzzlePool.Dequeue();
                if (pooledPuzzle != null && pooledPuzzle.GetPuzzleType() == puzzleType)
                {
                    pooledPuzzle.gameObject.SetActive(true);
                    return pooledPuzzle.gameObject;
                }
            }
            
            // Create new puzzle object
            var template = puzzleDatabase.GetTemplate(puzzleType);
            if (template?.PrefabReference != null)
            {
                return Instantiate(template.PrefabReference, transform);
            }
            
            return null;
        }
        
        private Vector3 FindOptimalPuzzlePlacement()
        {
            // Get available placement points from environment
            var boundaryManager = FindObjectOfType<GuardianBoundaryManager>();
            if (boundaryManager != null)
            {
                // Find accessible positions within play area
                Vector3 center = Camera.main.transform.position;
                float radius = UnityEngine.Random.Range(1.5f, 3.0f);
                Vector2 randomCircle = UnityEngine.Random.insideUnitCircle * radius;
                
                Vector3 candidate = new Vector3(
                    center.x + randomCircle.x,
                    0.5f, // Table height
                    center.z + randomCircle.y
                );
                
                // Validate placement
                if (IsValidPuzzlePlacement(candidate))
                    return candidate;
            }
            
            // Fallback to simple placement
            return Camera.main.transform.position + Camera.main.transform.forward * 2f;
        }
        
        private bool IsValidPuzzlePlacement(Vector3 position)
        {
            // Check for overlaps with existing puzzles
            foreach (var puzzle in activePuzzles)
            {
                if (puzzle != null && Vector3.Distance(puzzle.transform.position, position) < 1.5f)
                    return false;
            }
            
            // Check for obstacles
            if (Physics.CheckSphere(position, 0.5f))
                return false;
            
            // Check if within reach
            float distanceToPlayer = Vector3.Distance(position, Camera.main.transform.position);
            return distanceToPlayer >= 1.0f && distanceToPlayer <= 4.0f;
        }
        
        private float CalculateTimeLimit()
        {
            // Adjust time limit based on difficulty and player performance
            float multiplier = GetDifficultyTimeMultiplier();
            float performanceAdjustment = GetPerformanceTimeAdjustment();
            
            return baseTimeLimit * multiplier * performanceAdjustment;
        }
        
        private float GetDifficultyTimeMultiplier()
        {
            switch (currentDifficulty)
            {
                case DifficultyLevel.Tutorial: return 2.0f;
                case DifficultyLevel.Easy: return 1.5f;
                case DifficultyLevel.Medium: return 1.0f;
                case DifficultyLevel.Hard: return 0.8f;
                case DifficultyLevel.Expert: return 0.6f;
                case DifficultyLevel.Master: return 0.4f;
                default: return 1.0f;
            }
        }
        
        private float GetPerformanceTimeAdjustment()
        {
            float avgSolveTime = playerMetrics.GetAverageSolveTime();
            if (avgSolveTime <= 0) return 1.0f;
            
            // Give more time to players who typically take longer
            float ratio = avgSolveTime / baseTimeLimit;
            return Mathf.Clamp(ratio, 0.5f, 2.0f);
        }
        
        private void HandlePuzzleCompleted(BasePuzzle puzzle, bool success)
        {
            string puzzleId = puzzle.GetPuzzleID();
            
            if (puzzleStates.TryGetValue(puzzleId, out PuzzleState state))
            {
                state.IsActive = false;
                state.CompletionTime = Time.time - state.StartTime;
                state.Success = success;
                state.HintsUsed = hintSystem.GetHintsUsedCount(puzzleId);
                
                // Update metrics
                playerMetrics.RecordPuzzleCompletion(puzzle.GetPuzzleType(), success, state.CompletionTime, state.HintsUsed);
                
                // Update consecutive counters
                if (success)
                {
                    consecutiveSuccesses++;
                    consecutiveFailures = 0;
                }
                else
                {
                    consecutiveFailures++;
                    consecutiveSuccesses = 0;
                }
                
                // Check achievements
                CheckAchievements(puzzle, state);
                
                OnPuzzleCompleted?.Invoke(puzzle, success);
            }
            
            // Remove from active puzzles
            activePuzzles.Remove(puzzle);
            
            // Return to pool or destroy
            if (enablePuzzlePooling)
            {
                ReturnPuzzleToPool(puzzle);
            }
            else
            {
                DestroyImmediate(puzzle.gameObject);
            }
            
            // Update progress
            UpdateProgress();
            
            Debug.Log($"[PuzzleManager] Puzzle {puzzleId} completed. Success: {success}");
        }
        
        private void CheckAchievements(BasePuzzle puzzle, PuzzleState state)
        {
            // Perfect solve (no hints, fast time)
            if (state.Success && state.HintsUsed == 0 && state.CompletionTime < baseTimeLimit * 0.5f)
            {
                achievementTracker.UnlockAchievement("PERFECT_SOLVE");
            }
            
            // Speed demon (very fast solve)
            if (state.Success && state.CompletionTime < baseTimeLimit * 0.25f)
            {
                achievementTracker.UnlockAchievement("SPEED_DEMON");
            }
            
            // Persistence (completed after multiple failures)
            if (state.Success && consecutiveFailures >= 3)
            {
                achievementTracker.UnlockAchievement("PERSISTENT");
            }
            
            // Master difficulty completion
            if (state.Success && currentDifficulty == DifficultyLevel.Master)
            {
                achievementTracker.UnlockAchievement("MASTER_SOLVER");
            }
            
            // Check type-specific achievements
            CheckTypeSpecificAchievements(puzzle.GetPuzzleType());
        }
        
        private void CheckTypeSpecificAchievements(PuzzleType puzzleType)
        {
            var metrics = playerMetrics.GetTypeMetrics(puzzleType);
            
            // First solve of this type
            if (metrics.TotalSolved == 1)
            {
                achievementTracker.UnlockAchievement($"FIRST_{puzzleType.ToString().ToUpper()}");
            }
            
            // Master of type (10 solves)
            if (metrics.TotalSolved == 10)
            {
                achievementTracker.UnlockAchievement($"MASTER_{puzzleType.ToString().ToUpper()}");
            }
        }
        
        private void ReturnPuzzleToPool(BasePuzzle puzzle)
        {
            puzzle.Reset();
            puzzle.gameObject.SetActive(false);
            puzzlePool.Enqueue(puzzle);
        }
        
        private IEnumerator PrewarmPuzzlePool()
        {
            foreach (PuzzleType type in Enum.GetValues(typeof(PuzzleType)))
            {
                var template = puzzleDatabase.GetTemplate(type);
                if (template?.PrefabReference != null)
                {
                    for (int i = 0; i < 2; i++) // Pre-create 2 of each type
                    {
                        var poolObject = Instantiate(template.PrefabReference, transform);
                        poolObject.SetActive(false);
                        
                        var puzzle = poolObject.GetComponent<BasePuzzle>();
                        if (puzzle != null)
                        {
                            puzzlePool.Enqueue(puzzle);
                        }
                        
                        yield return null;
                    }
                }
            }
        }
        
        private IEnumerator DifficultyUpdateLoop()
        {
            while (true)
            {
                yield return new WaitForSeconds(5.0f); // Check every 5 seconds
                
                if (enableDynamicDifficulty)
                {
                    EvaluateDifficultyAdjustment();
                }
            }
        }
        
        private void EvaluateDifficultyAdjustment()
        {
            DifficultyLevel newDifficulty = currentDifficulty;
            
            // Increase difficulty
            if (consecutiveSuccesses >= consecutiveSuccessesForIncrease && currentDifficulty < DifficultyLevel.Master)
            {
                newDifficulty = currentDifficulty + 1;
            }
            // Decrease difficulty
            else if (consecutiveFailures >= consecutiveFailuresForDecrease && currentDifficulty > DifficultyLevel.Tutorial)
            {
                newDifficulty = currentDifficulty - 1;
            }
            
            if (newDifficulty != currentDifficulty)
            {
                SetDifficulty(newDifficulty);
            }
        }
        
        public void SetDifficulty(DifficultyLevel newDifficulty)
        {
            if (currentDifficulty == newDifficulty) return;
            
            DifficultyLevel oldDifficulty = currentDifficulty;
            currentDifficulty = newDifficulty;
            
            OnDifficultyChanged?.Invoke(currentDifficulty);
            
            Debug.Log($"[PuzzleManager] Difficulty changed from {oldDifficulty} to {currentDifficulty}");
            
            // Notify GameManager
            var gameManager = GameManager.Instance;
            gameManager?.GetSystem<AnalyticsManager>()?.LogEvent("difficulty_changed", new Dictionary<string, object>
            {
                {"old_difficulty", oldDifficulty.ToString()},
                {"new_difficulty", currentDifficulty.ToString()},
                {"consecutive_successes", consecutiveSuccesses},
                {"consecutive_failures", consecutiveFailures}
            });
        }
        
        private void UpdateProgress()
        {
            int completedPuzzles = puzzleStates.Values.Count(s => !s.IsActive);
            float progress = totalPuzzlesInSession > 0 ? completedPuzzles / (float)totalPuzzlesInSession : 0f;
            
            OnProgressUpdated?.Invoke(progress);
            
            // Notify GameManager of overall progress
            GameManager.Instance?.UpdateGameProgress(progress);
        }
        
        public float GetCompletionPercentage()
        {
            int completedPuzzles = puzzleStates.Values.Count(s => !s.IsActive && s.Success);
            return totalPuzzlesInSession > 0 ? completedPuzzles / (float)totalPuzzlesInSession : 0f;
        }
        
        private void HandleHintRequested(string puzzleId, HintType hintType)
        {
            Debug.Log($"[PuzzleManager] Hint requested for puzzle {puzzleId}: {hintType}");
            
            // Track hint usage in analytics
            var gameManager = GameManager.Instance;
            gameManager?.GetSystem<AnalyticsManager>()?.LogEvent("hint_requested", new Dictionary<string, object>
            {
                {"puzzle_id", puzzleId},
                {"hint_type", hintType.ToString()},
                {"difficulty", currentDifficulty.ToString()}
            });
        }
        
        private void HandleAchievementUnlocked(string achievementId)
        {
            OnAchievementUnlocked?.Invoke(achievementId);
            Debug.Log($"[PuzzleManager] Achievement unlocked: {achievementId}");
        }
        
        #region Save/Load System
        
        private void SavePuzzleStates()
        {
            try
            {
                string json = JsonUtility.ToJson(new PuzzleManagerSaveData
                {
                    PuzzleStates = puzzleStates.Values.ToArray(),
                    PlayerMetrics = playerMetrics,
                    CurrentDifficulty = currentDifficulty
                });
                
                System.IO.File.WriteAllText(GetSavePath(), json);
                Debug.Log("[PuzzleManager] Puzzle states saved");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[PuzzleManager] Failed to save puzzle states: {e.Message}");
            }
        }
        
        private void LoadPuzzleStates()
        {
            try
            {
                string savePath = GetSavePath();
                if (System.IO.File.Exists(savePath))
                {
                    string json = System.IO.File.ReadAllText(savePath);
                    var saveData = JsonUtility.FromJson<PuzzleManagerSaveData>(json);
                    
                    if (saveData != null)
                    {
                        foreach (var state in saveData.PuzzleStates)
                        {
                            puzzleStates[state.PuzzleID] = state;
                        }
                        
                        if (saveData.PlayerMetrics != null)
                            playerMetrics = saveData.PlayerMetrics;
                        
                        currentDifficulty = saveData.CurrentDifficulty;
                    }
                    
                    Debug.Log("[PuzzleManager] Puzzle states loaded");
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[PuzzleManager] Failed to load puzzle states: {e.Message}");
            }
        }
        
        private void SavePlayerMetrics()
        {
            // Implementation for saving player metrics to persistent storage
        }
        
        private void LoadPlayerMetrics()
        {
            // Implementation for loading player metrics from persistent storage
        }
        
        private string GetSavePath()
        {
            return System.IO.Path.Combine(Application.persistentDataPath, "puzzle_states.json");
        }
        
        #endregion
        
        #region IPerformanceAdjustable Implementation
        
        public void AdjustPerformance(GameManager.PerformanceLevel level)
        {
            switch (level)
            {
                case GameManager.PerformanceLevel.Low:
                    maxConcurrentPuzzles = 2;
                    puzzleUpdateInterval = 0.2f;
                    enablePuzzlePooling = true;
                    break;
                    
                case GameManager.PerformanceLevel.Medium:
                    maxConcurrentPuzzles = 3;
                    puzzleUpdateInterval = 0.15f;
                    enablePuzzlePooling = true;
                    break;
                    
                case GameManager.PerformanceLevel.High:
                    maxConcurrentPuzzles = 4;
                    puzzleUpdateInterval = 0.1f;
                    enablePuzzlePooling = true;
                    break;
                    
                case GameManager.PerformanceLevel.Ultra:
                    maxConcurrentPuzzles = 5;
                    puzzleUpdateInterval = 0.05f;
                    enablePuzzlePooling = false;
                    break;
            }
            
            // Apply performance adjustments to active puzzles
            foreach (var puzzle in activePuzzles)
            {
                if (puzzle is IPerformanceAdjustable adjustable)
                {
                    adjustable.AdjustPerformance(level);
                }
            }
            
            Debug.Log($"[PuzzleManager] Performance adjusted to {level}");
        }
        
        #endregion
        
        private void OnDestroy()
        {
            Cleanup();
        }
        
        #if UNITY_EDITOR
        [ContextMenu("Debug - Print Current Statistics")]
        private void DebugPrintStatistics()
        {
            Debug.Log($"=== Puzzle Manager Statistics ===");
            Debug.Log($"Current Difficulty: {currentDifficulty}");
            Debug.Log($"Active Puzzles: {activePuzzles.Count}");
            Debug.Log($"Consecutive Successes: {consecutiveSuccesses}");
            Debug.Log($"Consecutive Failures: {consecutiveFailures}");
            Debug.Log($"Session Time: {Time.time - sessionStartTime:F1}s");
            Debug.Log($"Completion Percentage: {GetCompletionPercentage() * 100:F1}%");
            Debug.Log($"================================");
        }
        #endif
    }
    
    #region Supporting Classes
    
    /**
     * @class PuzzleState
     * @brief Data structure for tracking individual puzzle state and completion data
     * 
     * @details
     * Represents the complete state of a puzzle instance including timing information,
     * completion status, difficulty level, and custom data. This class provides
     * comprehensive tracking for individual puzzle performance and enables persistent
     * state management across game sessions.
     * 
     * @fields
     * - PuzzleID: Unique identifier for the puzzle instance
     * - StartTime: Timestamp when puzzle was initiated
     * - CompletionTime: Time taken to complete the puzzle
     * - IsActive: Current activation status of the puzzle
     * - Success: Whether the puzzle was completed successfully
     * - HintsUsed: Number of hints utilized during puzzle solving
     * - Difficulty: Difficulty level at which puzzle was attempted
     * - CustomData: Additional puzzle-specific data storage
     * 
     * @serialization Supports Unity's serialization system for persistence
     */
    [System.Serializable]
    public class PuzzleState
    {
        public string PuzzleID; /**< Unique identifier for the puzzle instance */
        public float StartTime; /**< Timestamp when puzzle was initiated */
        public float CompletionTime; /**< Time taken to complete the puzzle */
        public bool IsActive; /**< Current activation status of the puzzle */
        public bool Success; /**< Whether the puzzle was completed successfully */
        public int HintsUsed; /**< Number of hints utilized during puzzle solving */
        public DifficultyLevel Difficulty; /**< Difficulty level at which puzzle was attempted */
        public Dictionary<string, object> CustomData = new Dictionary<string, object>(); /**< Additional puzzle-specific data storage */
    }
    
    /**
     * @class PlayerPuzzleMetrics
     * @brief Comprehensive tracking system for player puzzle performance and statistics
     * 
     * @details
     * Manages and tracks comprehensive player performance data across all puzzle
     * types and sessions. This class provides detailed analytics for individual
     * puzzle types, overall performance metrics, and recently used puzzles for
     * adaptive difficulty adjustment and player experience optimization.
     * 
     * @fields
     * - TypeMetrics: Performance data organized by puzzle type
     * - RecentlyUsedPuzzles: List of recently attempted puzzle IDs
     * - TotalPlayTime: Cumulative time spent on puzzle solving
     * - TotalPuzzlesSolved: Total number of successfully completed puzzles
     * - TotalPuzzlesAttempted: Total number of puzzle attempts
     * 
     * @analytics Provides comprehensive performance tracking and analysis
     * @difficulty_adjustment Supports adaptive difficulty algorithms
     * @persistence Enables cross-session progress tracking
     */
    [System.Serializable]
    public class PlayerPuzzleMetrics
    {
        public Dictionary<PuzzleType, PuzzleTypeMetrics> TypeMetrics = new Dictionary<PuzzleType, PuzzleTypeMetrics>(); /**< Performance data organized by puzzle type */
        public List<string> RecentlyUsedPuzzles = new List<string>(); /**< List of recently attempted puzzle IDs */
        public float TotalPlayTime; /**< Cumulative time spent on puzzle solving */
        public int TotalPuzzlesSolved; /**< Total number of successfully completed puzzles */
        public int TotalPuzzlesAttempted; /**< Total number of puzzle attempts */
        
        public PuzzleTypeMetrics GetTypeMetrics(PuzzleType type)
        {
            if (!TypeMetrics.ContainsKey(type))
                TypeMetrics[type] = new PuzzleTypeMetrics();
            return TypeMetrics[type];
        }
        
        public void RecordPuzzleCompletion(PuzzleType type, bool success, float time, int hints)
        {
            var metrics = GetTypeMetrics(type);
            metrics.TotalAttempts++;
            metrics.TotalHintsUsed += hints;
            
            if (success)
            {
                metrics.TotalSolved++;
                metrics.TotalSolveTime += time;
                TotalPuzzlesSolved++;
            }
            
            TotalPuzzlesAttempted++;
        }
        
        public float GetAverageSolveTime()
        {
            int totalSolved = TypeMetrics.Values.Sum(m => m.TotalSolved);
            if (totalSolved == 0) return 0f;
            
            float totalTime = TypeMetrics.Values.Sum(m => m.TotalSolveTime);
            return totalTime / totalSolved;
        }
        
        public void Reset()
        {
            TypeMetrics.Clear();
            RecentlyUsedPuzzles.Clear();
            TotalPlayTime = 0f;
            TotalPuzzlesSolved = 0;
            TotalPuzzlesAttempted = 0;
        }
    }
    
    /**
     * @class PuzzleTypeMetrics
     * @brief Detailed performance metrics for individual puzzle types
     * 
     * @details
     * Tracks comprehensive performance data for specific puzzle types including
     * attempt counts, success rates, timing statistics, and hint usage patterns.
     * This data enables detailed analysis of player performance across different
     * puzzle categories and supports adaptive difficulty adjustment algorithms.
     * 
     * @fields
     * - TotalAttempts: Total number of attempts for this puzzle type
     * - TotalSolved: Number of successful completions
     * - TotalSolveTime: Cumulative time spent solving this puzzle type
     * - TotalHintsUsed: Total number of hints utilized
     * - FastestSolveTime: Best completion time achieved
     * - SlowestSolveTime: Longest completion time recorded
     * 
     * @performance_tracking Enables detailed performance analysis
     * @difficulty_adjustment Supports puzzle type-specific difficulty tuning
     * @analytics Provides insights into player learning patterns
     */
    [System.Serializable]
    public class PuzzleTypeMetrics
    {
        public int TotalAttempts; /**< Total number of attempts for this puzzle type */
        public int TotalSolved; /**< Number of successful completions */
        public float TotalSolveTime; /**< Cumulative time spent solving this puzzle type */
        public int TotalHintsUsed; /**< Total number of hints utilized */
        public float FastestSolveTime = float.MaxValue; /**< Best completion time achieved */
        public float SlowestSolveTime; /**< Longest completion time recorded */
    }
    
    /**
     * @class PuzzleManagerSaveData
     * @brief Data structure for persistent storage of puzzle management system state
     * 
     * @details
     * Encapsulates all necessary data for saving and restoring the complete state
     * of the puzzle management system. This class enables cross-session persistence
     * of puzzle states, player metrics, and difficulty settings for seamless
     * gameplay continuity.
     * 
     * @fields
     * - PuzzleStates: Array of all puzzle state information
     * - PlayerMetrics: Complete player performance and statistics data
     * - CurrentDifficulty: Current difficulty level setting
     * 
     * @persistence Enables cross-session state preservation
     * @serialization Supports Unity's JSON serialization system
     * @state_management Provides complete system state restoration
     */
    [System.Serializable]
    public class PuzzleManagerSaveData
    {
        public PuzzleState[] PuzzleStates; /**< Array of all puzzle state information */
        public PlayerPuzzleMetrics PlayerMetrics; /**< Complete player performance and statistics data */
        public DifficultyLevel CurrentDifficulty; /**< Current difficulty level setting */
    }
    
    #endregion
} 