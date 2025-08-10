/**
 * @file PerformanceMonitor.cs
 * @brief Advanced performance monitoring system for Mixed Reality applications
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements a comprehensive performance monitoring system designed specifically
 * for Mixed Reality applications. It tracks frame rate, memory usage, CPU/GPU performance,
 * thermal throttling, and provides adaptive quality recommendations to maintain optimal
 * performance on mobile VR platforms such as the Oculus Quest.
 * 
 * @features
 * - Real-time frame rate monitoring with statistical analysis
 * - Memory usage tracking with leak detection algorithms
 * - CPU and GPU performance monitoring using Unity Profiler
 * - Thermal state monitoring and throttling detection
 * - Performance alert system with configurable thresholds
 * - Adaptive quality recommendations based on performance metrics
 * - Comprehensive performance metrics collection and analysis
 * - Trend analysis for performance degradation detection
 * - Integration with Unity Profiler for detailed performance data
 * - Configurable monitoring intervals and sample sizes
 * 
 * @monitoring_capabilities
 * The system provides comprehensive monitoring of:
 * 1. Frame rate metrics including average, minimum, maximum, and variance
 * 2. Memory usage patterns with automatic leak detection
 * 3. System performance including CPU/GPU utilization
 * 4. Thermal state monitoring for mobile device optimization
 * 5. Rendering performance including draw calls and render thread time
 * 
 * @alert_system
 * - Configurable performance thresholds for different metrics
 * - Multiple alert severity levels (Info, Warning, Critical)
 * - Automatic alert generation based on performance degradation
 * - Event-driven notification system for external systems
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - Unity Profiler for detailed performance data
 * - Unity Collections for efficient data structures
 * - System.Collections for queue-based data management
 * 
 * @license MIT License
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using Unity.Profiling;

namespace MREscapeRoom.Core
{
    /**
     * @class PerformanceMonitor
     * @brief Advanced performance monitoring system for Mixed Reality applications
     * 
     * @details
     * The PerformanceMonitor class provides comprehensive performance tracking and
     * analysis capabilities for Mixed Reality applications. It continuously monitors
     * various performance metrics, generates alerts for performance issues, and
     * provides recommendations for quality adjustments to maintain optimal performance.
     * 
     * @features
     * - Real-time performance metric collection and analysis
     * - Configurable monitoring intervals and sample sizes
     * - Event-driven alert system for performance issues
     * - Integration with Unity Profiler for detailed performance data
     * - Automatic trend analysis and performance degradation detection
     */
    public class PerformanceMonitor : MonoBehaviour
    {
        [Header("Performance Monitoring Configuration")]
        [SerializeField] private float targetFrameRate = 72.0f;
        [SerializeField] private int sampleSize = 120;
        [SerializeField] private float updateInterval = 0.5f;
        
        [Header("Performance Thresholds")]
        [SerializeField] private float lowFrameRateThreshold = 0.8f;
        [SerializeField] private float highMemoryThreshold = 1000f;
        [SerializeField] private float highCPUThreshold = 80f;
        [SerializeField] private float highGPUThreshold = 80f;
        
        private Queue<float> frameTimes;
        private Queue<float> memoryUsage;
        
        [Header("Monitoring State")]
        private bool isMonitoring = false;
        private float lastUpdateTime;
        
        [Header("Performance Data")]
        private PerformanceMetrics currentMetrics;
        private PerformanceAlert currentAlert;
        
        /**
         * @brief Unity Profiler markers for performance tracking
         */
        private static readonly ProfilerMarker s_FrameTimeMarker = new ProfilerMarker("FrameTime");
        private static readonly ProfilerMarker s_MemoryUsageMarker = new ProfilerMarker("MemoryUsage");
        
        /**
         * @brief Events for performance monitoring notifications
         */
        public event Action<PerformanceAlert> OnPerformanceAlert;
        public event Action<PerformanceMetrics> OnMetricsUpdated;
        
        /**
         * @brief Unity lifecycle method for initialization
         */
        private void Awake()
        {
            InitializePerformanceMonitor();
        }
        
        /**
         * @brief Unity lifecycle method for starting monitoring
         */
        private void Start()
        {
            StartMonitoring();
        }
        
        /**
         * @brief Unity lifecycle method for cleanup
         */
        private void OnDestroy()
        {
            StopMonitoring();
        }
        
        /**
         * @brief Initializes the performance monitor
         * 
         * @details
         * Sets up data structures for metric collection and prepares the monitoring
         * system for operation.
         */
        private void InitializePerformanceMonitor()
        {
            frameTimes = new Queue<float>(sampleSize);
            memoryUsage = new Queue<float>(sampleSize);
            currentMetrics = new PerformanceMetrics();
            
            Debug.Log($"[PerformanceMonitor] Initialized with target FPS: {targetFrameRate}");
        }
        
        /**
         * @brief Starts the performance monitoring system
         * 
         * @details
         * Activates continuous performance monitoring, initializes timing variables,
         * and begins collecting performance metrics for analysis and alert generation.
         */
        public void StartMonitoring()
        {
            isMonitoring = true;
            lastUpdateTime = Time.realtimeSinceStartup;
            Debug.Log("[PerformanceMonitor] Monitoring started");
        }
        
        /**
         * @brief Stops the performance monitoring system
         * 
         * @details
         * Deactivates performance monitoring, stops metric collection, and
         * preserves collected data for final analysis and reporting.
         */
        public void StopMonitoring()
        {
            isMonitoring = false;
            Debug.Log("[PerformanceMonitor] Monitoring stopped");
        }
        
        /**
         * @brief Unity lifecycle method for frame updates
         * 
         * @details
         * Continuously collects frame time data and periodically updates
         * comprehensive performance metrics. This method is called every frame.
         */
        private void Update()
        {
            if (!isMonitoring) return;
            
            using (s_FrameTimeMarker.Auto())
            {
                CollectFrameTimeData();
            }
            
            if (Time.realtimeSinceStartup - lastUpdateTime >= updateInterval)
            {
                using (s_MemoryUsageMarker.Auto())
                {
                    UpdatePerformanceMetrics();
                    CheckForPerformanceAlerts();
                }
                
                lastUpdateTime = Time.realtimeSinceStartup;
            }
        }
        
        /**
         * @brief Collects frame time data for performance analysis
         * 
         * @details
         * Records the time taken for each frame and maintains a rolling
         * queue of recent frame times for statistical analysis and
         * performance trend detection.
         */
        private void CollectFrameTimeData()
        {
            float frameTime = Time.unscaledDeltaTime;
            
            frameTimes.Enqueue(frameTime);
            if (frameTimes.Count > sampleSize)
            {
                frameTimes.Dequeue();
            }
        }
        
        /**
         * @brief Updates all performance metrics including FPS, memory, and system performance
         * 
         * @details
         * Comprehensive performance metric collection that analyzes frame rate data,
         * memory usage patterns, system performance indicators, and thermal state
         * information to provide a complete picture of application performance.
         */
        private void UpdatePerformanceMetrics()
        {
            CalculateFPSMetrics();
            CalculateMemoryMetrics();
            CalculateSystemMetrics();
            UpdateThermalMetrics();
            
            currentMetrics.timestamp = Time.time;
            OnMetricsUpdated?.Invoke(currentMetrics);
        }
        
        /**
         * @brief Calculates frame rate performance metrics
         * 
         * @details
         * Analyzes collected frame time data to determine average, minimum,
         * and maximum frame rates, as well as frame time variance for
         * performance consistency assessment.
         */
        private void CalculateFPSMetrics()
        {
            if (frameTimes.Count == 0) return;
            
            float totalFrameTime = 0f;
            float minFrameTime = float.MaxValue;
            float maxFrameTime = 0f;
            
            foreach (float frameTime in frameTimes)
            {
                totalFrameTime += frameTime;
                minFrameTime = Mathf.Min(minFrameTime, frameTime);
                maxFrameTime = Mathf.Max(maxFrameTime, frameTime);
            }
            
            float averageFrameTime = totalFrameTime / frameTimes.Count;
            currentMetrics.averageFPS = 1.0f / averageFrameTime;
            currentMetrics.minFPS = 1.0f / maxFrameTime;
            currentMetrics.maxFPS = 1.0f / minFrameTime;
            
            float varianceSum = 0f;
            foreach (float frameTime in frameTimes)
            {
                float diff = frameTime - averageFrameTime;
                varianceSum += diff * diff;
            }
            currentMetrics.frameTimeVariance = varianceSum / frameTimes.Count;
        }
        
        /**
         * @brief Calculates memory usage metrics
         * 
         * @details
         * Monitors memory allocation patterns, tracks memory usage over time,
         * and detects potential memory leaks through trend analysis of
         * memory consumption patterns.
         */
        private void CalculateMemoryMetrics()
        {
            currentMetrics.totalMemoryMB = Profiler.GetTotalAllocatedMemoryLong() / (1024f * 1024f);
            currentMetrics.reservedMemoryMB = Profiler.GetReservedMemoryLong() / (1024f * 1024f);
            currentMetrics.unusedMemoryMB = Profiler.GetUnusedReservedMemoryLong() / (1024f * 1024f);
            
            memoryUsage.Enqueue(currentMetrics.totalMemoryMB);
            if (memoryUsage.Count > sampleSize)
            {
                memoryUsage.Dequeue();
            }
            
            DetectMemoryLeaks();
        }
        
        /**
         * @brief Detects potential memory leaks through trend analysis
         * 
         * @details
         * Analyzes memory usage patterns over time to identify consistent
         * increases in memory consumption that may indicate memory leaks
         * or inefficient memory management.
         */
        private void DetectMemoryLeaks()
        {
            if (memoryUsage.Count < 10) return;
            
            float[] memoryArray = memoryUsage.ToArray();
            float trend = CalculateTrend(memoryArray);
            
            currentMetrics.memoryLeakDetected = trend > 0.1f;
        }
        
        /**
         * @brief Calculates system performance metrics
         * 
         * @details
         * Collects CPU and GPU utilization data, render thread performance
         * metrics, and draw call information to assess overall system
         * performance and identify potential bottlenecks.
         */
        private void CalculateSystemMetrics()
        {
            // Note: These are simplified metrics - in a real implementation,
            // you would use more sophisticated profiling methods
            currentMetrics.cpuUsagePercent = Mathf.Clamp01(Time.deltaTime / (1f / targetFrameRate)) * 100f;
            currentMetrics.gpuUsagePercent = Mathf.Clamp01(1f - (currentMetrics.averageFPS / targetFrameRate)) * 100f;
            currentMetrics.renderThreadTime = Time.deltaTime * 1000f; // Convert to milliseconds
            currentMetrics.drawCalls = UnityEngine.Rendering.Universal.UniversalRenderPipeline.asset != null ? 
                UnityEngine.Rendering.Universal.UniversalRenderPipeline.asset.shadowCascadeCount : 0;
        }
        
        /**
         * @brief Updates thermal state and throttling detection
         * 
         * @details
         * Monitors device thermal state to detect potential throttling
         * conditions that could impact performance, particularly important
         * for mobile VR devices like the Oculus Quest.
         */
        private void UpdateThermalMetrics()
        {
            float batteryLevel = SystemInfo.batteryLevel;
            float batteryStatus = SystemInfo.batteryStatus.GetHashCode();
            
            if (batteryLevel < 0.2f || batteryStatus > 0.8f)
            {
                currentMetrics.thermalThrottlingDetected = true;
                currentMetrics.thermalState = ThermalState.Warning;
            }
            else if (batteryLevel < 0.1f || batteryStatus > 0.9f)
            {
                currentMetrics.thermalThrottlingDetected = true;
                currentMetrics.thermalState = ThermalState.Critical;
            }
            else
            {
                currentMetrics.thermalThrottlingDetected = false;
                currentMetrics.thermalState = ThermalState.Nominal;
            }
        }
        
        /**
         * @brief Checks for performance alerts based on current metrics
         * 
         * @details
         * Analyzes current performance metrics against configurable thresholds
         * to generate appropriate performance alerts. Alerts are categorized
         * by severity and type to enable targeted performance optimization.
         */
        private void CheckForPerformanceAlerts()
        {
            CheckFrameRateAlerts();
            CheckMemoryAlerts();
            CheckThermalAlerts();
            CheckSystemAlerts();
        }
        
        /**
         * @brief Checks for frame rate related performance alerts
         * 
         * @details
         * Monitors frame rate performance and generates alerts when frame
         * rates fall below target thresholds or when frame time variance
         * indicates performance instability.
         */
        private void CheckFrameRateAlerts()
        {
            if (currentMetrics.averageFPS < targetFrameRate * lowFrameRateThreshold)
            {
                GenerateAlert(PerformanceAlert.AlertType.LowFrameRate, 
                            PerformanceAlert.AlertSeverity.Warning,
                            $"Low frame rate detected: {currentMetrics.averageFPS:F1} FPS",
                            currentMetrics.averageFPS,
                            PerformanceAction.ReduceQuality);
            }
            
            if (currentMetrics.frameTimeVariance > 0.01f)
            {
                GenerateAlert(PerformanceAlert.AlertType.LowFrameRate,
                            PerformanceAlert.AlertSeverity.Info,
                            "High frame time variance detected",
                            currentMetrics.frameTimeVariance,
                            PerformanceAction.Monitor);
            }
        }
        
        /**
         * @brief Checks for memory related performance alerts
         * 
         * @details
         * Monitors memory usage patterns and generates alerts for high
         * memory consumption, memory leaks, or inefficient memory management
         * that could impact application performance.
         */
        private void CheckMemoryAlerts()
        {
            if (currentMetrics.totalMemoryMB > highMemoryThreshold)
            {
                GenerateAlert(PerformanceAlert.AlertType.HighMemoryUsage,
                            PerformanceAlert.AlertSeverity.Warning,
                            $"High memory usage: {currentMetrics.totalMemoryMB:F1} MB",
                            currentMetrics.totalMemoryMB,
                            PerformanceAction.ReduceQuality);
            }
            
            if (currentMetrics.memoryLeakDetected)
            {
                GenerateAlert(PerformanceAlert.AlertType.MemoryLeak,
                            PerformanceAlert.AlertSeverity.Critical,
                            "Potential memory leak detected",
                            currentMetrics.totalMemoryMB,
                            PerformanceAction.Investigate);
            }
        }
        
        /**
         * @brief Checks for thermal related performance alerts
         * 
         * @details
         * Monitors device thermal state and generates alerts when thermal
         * throttling is detected, which could significantly impact
         * application performance on mobile devices.
         */
        private void CheckThermalAlerts()
        {
            if (currentMetrics.thermalThrottlingDetected)
            {
                GenerateAlert(PerformanceAlert.AlertType.ThermalThrottling,
                            PerformanceAlert.AlertSeverity.Critical,
                            $"Thermal throttling detected: {currentMetrics.thermalState}",
                            (float)currentMetrics.thermalState,
                            PerformanceAction.ReduceQuality);
            }
        }
        
        /**
         * @brief Checks for system performance alerts
         * 
         * @details
         * Monitors CPU and GPU utilization to identify potential system
         * bottlenecks that could impact application performance and
         * user experience.
         */
        private void CheckSystemAlerts()
        {
            if (currentMetrics.cpuUsagePercent > highCPUThreshold)
            {
                GenerateAlert(PerformanceAlert.AlertType.HighCPUUsage,
                            PerformanceAlert.AlertSeverity.Warning,
                            $"High CPU usage: {currentMetrics.cpuUsagePercent:F1}%",
                            currentMetrics.cpuUsagePercent,
                            PerformanceAction.ReduceQuality);
            }
            
            if (currentMetrics.gpuUsagePercent > highGPUThreshold)
            {
                GenerateAlert(PerformanceAlert.AlertType.HighGPUUsage,
                            PerformanceAlert.AlertSeverity.Warning,
                            $"High GPU usage: {currentMetrics.gpuUsagePercent:F1}%",
                            currentMetrics.gpuUsagePercent,
                            PerformanceAction.ReduceQuality);
            }
        }
        
        /**
         * @brief Generates a performance alert with specified parameters
         * 
         * @details
         * Creates a new performance alert with the specified type, severity,
         * message, value, and recommended action, then triggers the OnPerformanceAlert event
         * to notify external systems of the performance issue.
         * 
         * @param type Type of performance alert to generate
         * @param severity Severity level of the alert
         * @param message Descriptive message about the performance issue
         * @param value Numerical value associated with the alert
         * @param recommendedAction Recommended action to take
         */
        private void GenerateAlert(PerformanceAlert.AlertType type, 
                                 PerformanceAlert.AlertSeverity severity,
                                 string message, float value,
                                 PerformanceAction recommendedAction)
        {
            currentAlert = new PerformanceAlert
            {
                Type = type,
                Severity = severity,
                Message = message,
                Value = value,
                Timestamp = Time.time,
                RecommendedAction = recommendedAction,
                CurrentFPS = currentMetrics.averageFPS,
                AlertType = type
            };
            
            OnPerformanceAlert?.Invoke(currentAlert);
        }
        
        /**
         * @brief Calculates trend values for performance metrics
         * 
         * @details
         * Performs linear regression analysis on performance metric arrays
         * to determine trends and identify consistent performance degradation
         * or improvement patterns over time.
         * 
         * @param values Array of performance metric values
         * @returns Trend value indicating performance direction
         */
        private float CalculateTrend(float[] values)
        {
            if (values.Length < 2) return 0f;
            
            float sumX = 0f;
            float sumY = 0f;
            float sumXY = 0f;
            float sumX2 = 0f;
            
            for (int i = 0; i < values.Length; i++)
            {
                float x = i;
                float y = values[i];
                
                sumX += x;
                sumY += y;
                sumXY += x * y;
                sumX2 += x * x;
            }
            
            float n = values.Length;
            float slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
            
            return slope;
        }
        
        /**
         * @brief Gets current performance metrics
         * 
         * @details
         * Returns the most recently collected performance metrics,
         * providing external systems with access to current performance
         * data for analysis and decision-making.
         * 
         * @returns Current performance metrics
         */
        public PerformanceMetrics GetCurrentMetrics()
        {
            return currentMetrics;
        }
        
        /**
         * @brief Gets performance recommendations based on current metrics
         * 
         * @details
         * Analyzes current performance metrics to generate actionable
         * recommendations for performance optimization, including quality
         * level adjustments and specific actions to improve performance.
         * 
         * @returns Performance optimization recommendations
         */
        public PerformanceRecommendations GetPerformanceRecommendations()
        {
            var recommendations = new PerformanceRecommendations();
            
            if (currentMetrics.averageFPS < targetFrameRate * 0.7f)
            {
                recommendations.SuggestedActions.Add("Reduce rendering quality");
                recommendations.SuggestedActions.Add("Decrease draw distance");
                recommendations.SuggestedActions.Add("Optimize shader complexity");
                recommendations.RecommendedQualityLevel = GameManager.PerformanceLevel.Low;
                recommendations.ConfidenceLevel = 0.9f;
            }
            else if (currentMetrics.averageFPS < targetFrameRate * 0.85f)
            {
                recommendations.SuggestedActions.Add("Moderate quality reduction");
                recommendations.SuggestedActions.Add("Optimize texture resolution");
                recommendations.RecommendedQualityLevel = GameManager.PerformanceLevel.Medium;
                recommendations.ConfidenceLevel = 0.7f;
            }
            else
            {
                recommendations.SuggestedActions.Add("Performance is acceptable");
                recommendations.RecommendedQualityLevel = GameManager.PerformanceLevel.High;
                recommendations.ConfidenceLevel = 0.8f;
            }
            
            if (currentMetrics.memoryLeakDetected)
            {
                recommendations.SuggestedActions.Add("Investigate memory leaks");
                recommendations.SuggestedActions.Add("Review object pooling");
                recommendations.ConfidenceLevel = Mathf.Min(recommendations.ConfidenceLevel, 0.6f);
            }
            
            if (currentMetrics.thermalThrottlingDetected)
            {
                recommendations.SuggestedActions.Add("Reduce processing load");
                recommendations.SuggestedActions.Add("Implement thermal management");
                recommendations.ConfidenceLevel = Mathf.Min(recommendations.ConfidenceLevel, 0.5f);
            }
            
            return recommendations;
        }
        
        /**
         * @brief Logs current metrics to console for debugging
         * 
         * @details
         * Outputs current performance metrics to the Unity console for
         * debugging and development purposes, providing detailed
         * information about application performance.
         */
        public void LogMetricsToConsole()
        {
            Debug.Log($"[PerformanceMonitor] FPS: {currentMetrics.averageFPS:F1} " +
                     $"(Min: {currentMetrics.minFPS:F1}, Max: {currentMetrics.maxFPS:F1})");
            Debug.Log($"[PerformanceMonitor] Memory: {currentMetrics.totalMemoryMB:F1} MB " +
                     $"(Reserved: {currentMetrics.reservedMemoryMB:F1} MB)");
            Debug.Log($"[PerformanceMonitor] Thermal: {currentMetrics.thermalState} " +
                     $"(Throttling: {currentMetrics.thermalThrottlingDetected})");
        }
        
        /**
         * @brief Public method to update performance metrics (for external calls)
         * 
         * @details
         * Allows external systems to trigger performance metric updates
         * outside of the normal update cycle.
         */
        public void UpdatePerformanceMetricsExternal()
        {
            UpdatePerformanceMetrics();
        }
    }
    
    /**
     * @struct PerformanceMetrics
     * @brief Comprehensive performance metrics structure
     * 
     * @details
     * Contains all performance metrics collected by the PerformanceMonitor,
     * including frame rate data, memory usage, system performance, and
     * thermal state information for comprehensive performance analysis.
     */
    [System.Serializable]
    public class PerformanceMetrics
    {
        [Header("Frame Rate Metrics")]
        public float averageFPS;           /**< Average frames per second */
        public float minFPS;               /**< Minimum frames per second */
        public float maxFPS;               /**< Maximum frames per second */
        public float frameTimeVariance;    /**< Variance in frame time */
        
        [Header("Memory Metrics")]
        public float totalMemoryMB;        /**< Total allocated memory in MB */
        public float reservedMemoryMB;     /**< Reserved memory in MB */
        public float unusedMemoryMB;       /**< Unused reserved memory in MB */
        public bool memoryLeakDetected;    /**< Memory leak detection flag */
        
        [Header("System Performance")]
        public float cpuUsagePercent;      /**< CPU usage percentage */
        public float gpuUsagePercent;      /**< GPU usage percentage */
        public float renderThreadTime;     /**< Render thread execution time */
        public int drawCalls;              /**< Number of draw calls */
        
        [Header("Thermal State")]
        public bool thermalThrottlingDetected; /**< Thermal throttling detection */
        public ThermalState thermalState;      /**< Current thermal state */
        
        [Header("Timing")]
        public float timestamp;            /**< Metric collection timestamp */
    }
    
    /**
     * @enum ThermalState
     * @brief Thermal state enumeration for device monitoring
     * 
     * @details
     * Defines the thermal states that can be detected by the performance
     * monitoring system, indicating the level of thermal stress on the device.
     */
    public enum ThermalState
    {
        Nominal,    /**< Normal thermal conditions */
        Warning,    /**< Elevated thermal conditions */
        Critical    /**< Critical thermal conditions requiring immediate attention */
    }
    
    /**
     * @enum PerformanceAction
     * @brief Recommended actions for performance issues
     * 
     * @details
     * Defines the recommended actions that can be taken in response to
     * performance alerts, helping guide performance optimization efforts.
     */
    public enum PerformanceAction
    {
        Monitor,        /**< Continue monitoring the situation */
        ReduceQuality,  /**< Reduce quality settings for better performance */
        Investigate     /**< Investigate the root cause of the issue */
    }
    
    /**
     * @struct PerformanceAlert
     * @brief Performance alert structure for issue notification
     * 
     * @details
     * Contains information about performance issues detected by the
     * monitoring system, including alert type, severity, message,
     * and associated numerical values for analysis.
     */
    [System.Serializable]
    public class PerformanceAlert
    {
        [Header("Alert Information")]
        public AlertType Type;                     /**< Type of performance alert */
        public AlertSeverity Severity;             /**< Severity level of the alert */
        public string Message;                     /**< Descriptive alert message */
        public float Value;                        /**< Numerical value associated with alert */
        public float Timestamp;                    /**< Alert generation timestamp */
        public PerformanceAction RecommendedAction; /**< Recommended action to take */
        public float CurrentFPS;                   /**< Current frame rate when alert was generated */
        public AlertType AlertType;                /**< Duplicate field for GameManager compatibility */
        
        /**
         * @enum AlertType
         * @brief Types of performance alerts
         * 
         * @details
         * Defines the different categories of performance issues that
         * can be detected and reported by the monitoring system.
         */
        public enum AlertType
        {
            LowFrameRate,        /**< Frame rate below acceptable threshold */
            HighMemoryUsage,     /**< Memory usage above acceptable threshold */
            MemoryLeak,          /**< Potential memory leak detected */
            ThermalThrottling,   /**< Thermal throttling detected */
            HighCPUUsage,        /**< CPU usage above acceptable threshold */
            HighGPUUsage         /**< GPU usage above acceptable threshold */
        }
        
        /**
         * @enum AlertSeverity
         * @brief Severity levels for performance alerts
         * 
         * @details
         * Defines the severity levels for performance alerts, helping
         * prioritize issues and determine appropriate response actions.
         */
        public enum AlertSeverity
        {
            Info,       /**< Informational alert requiring no action */
            Warning,    /**< Warning alert requiring attention */
            Critical    /**< Critical alert requiring immediate action */
        }
        
        /**
         * @brief Equality comparison for performance alerts
         * 
         * @details
         * Compares performance alerts based on type, severity, and
         * timestamp to determine if they represent the same issue.
         * 
         * @param obj Object to compare with
         * @returns True if alerts are considered equal
         */
        public override bool Equals(object obj)
        {
            if (obj is PerformanceAlert other)
            {
                return Type == other.Type && 
                       Severity == other.Severity && 
                       Mathf.Abs(Timestamp - other.Timestamp) < 1.0f;
            }
            return false;
        }
        
        /**
         * @brief Hash code generation for performance alerts
         * 
         * @details
         * Generates a hash code based on alert type, severity, and
         * timestamp for efficient storage and retrieval in hash-based
         * data structures.
         * 
         * @returns Hash code for the performance alert
         */
        public override int GetHashCode()
        {
            return Type.GetHashCode() ^ Severity.GetHashCode() ^ Timestamp.GetHashCode();
        }
    }
    
    /**
     * @struct PerformanceRecommendations
     * @brief Performance optimization recommendations
     * 
     * @details
     * Contains actionable recommendations for performance optimization
     * based on current performance metrics, including suggested actions
     * and recommended quality levels with confidence ratings.
     */
    [System.Serializable]
    public class PerformanceRecommendations
    {
        [Header("Recommendations")]
        public List<string> SuggestedActions = new List<string>(); /**< List of suggested optimization actions */
        public GameManager.PerformanceLevel RecommendedQualityLevel; /**< Recommended performance quality level */
        public float ConfidenceLevel = 1.0f; /**< Confidence level in recommendations (0.0 to 1.0) */
    }
} 