/**
 * @file PerformanceMonitorIntegration.cs
 * @brief Unity-Python backend integration for enhanced performance monitoring
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class provides real-time integration between Unity's PerformanceMonitor and
 * Python backend systems, enabling cross-platform performance monitoring, automated
 * optimization recommendations, and adaptive quality adjustments based on system
 * performance across both platforms.
 * 
 * @features
 * - WebSocket communication with Python backend
 * - Real-time performance data streaming
 * - Automated performance optimization recommendations
 * - Cross-platform performance analysis
 * - Backend performance monitoring
 * - Quality level synchronization
 * - Emergency performance optimization triggers
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - PerformanceMonitor.cs
 * - WebSocket client library (NativeWebSocket or similar)
 * - JSON serialization
 * 
 * @license MIT License
 */

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Text;
using System.Threading.Tasks;

namespace MREscapeRoom.Core
{
    /**
     * @class PerformanceMonitorIntegration
     * @brief Unity-Python backend integration for enhanced performance monitoring
     * 
     * @details
     * The PerformanceMonitorIntegration class establishes real-time communication
     * between Unity's PerformanceMonitor and Python backend systems, enabling
     * comprehensive cross-platform performance monitoring and optimization.
     */
    public class PerformanceMonitorIntegration : MonoBehaviour
    {
        [Header("Backend Integration Configuration")]
        [SerializeField] private string backendWebSocketUrl = "ws://localhost:8889";
        [SerializeField] private string backendApiUrl = "http://localhost:8000";
        [SerializeField] private bool enableBackendIntegration = true;
        [SerializeField] private float reconnectInterval = 5.0f;
        [SerializeField] private float metricsSendInterval = 1.0f;
        
        [Header("Performance Thresholds")]
        [SerializeField] private float criticalFPSThreshold = 45.0f;
        [SerializeField] private float warningFPSThreshold = 60.0f;
        [SerializeField] private float criticalMemoryThreshold = 2000.0f;
        [SerializeField] private float warningMemoryThreshold = 1500.0f;
        
        [Header("Integration State")]
        [SerializeField] private bool isConnected = false;
        [SerializeField] private bool isStreaming = false;
        [SerializeField] private int reconnectAttempts = 0;
        [SerializeField] private int maxReconnectAttempts = 10;
        
        private PerformanceMonitor performanceMonitor;
        private WebSocket webSocket;
        private Coroutine reconnectCoroutine;
        private Coroutine metricsStreamingCoroutine;
        private Queue<PerformanceMetrics> metricsQueue;
        private Queue<PerformanceAlert> alertsQueue;
        
        // Events for external systems
        public event Action<bool> OnBackendConnectionChanged;
        public event Action<List<PerformanceRecommendation>> OnBackendRecommendationsReceived;
        public event Action<BackendMetrics> OnBackendMetricsReceived;
        
        /**
         * @brief Unity lifecycle method for initialization
         */
        private void Awake()
        {
            InitializeIntegration();
        }
        
        /**
         * @brief Unity lifecycle method for starting integration
         */
        private void Start()
        {
            if (enableBackendIntegration)
            {
                StartBackendIntegration();
            }
        }
        
        /**
         * @brief Unity lifecycle method for cleanup
         */
        private void OnDestroy()
        {
            StopBackendIntegration();
        }
        
        /**
         * @brief Initializes the integration system
         */
        private void InitializeIntegration()
        {
            metricsQueue = new Queue<PerformanceMetrics>();
            alertsQueue = new Queue<PerformanceAlert>();
            
            // Get reference to PerformanceMonitor
            performanceMonitor = FindObjectOfType<PerformanceMonitor>();
            if (performanceMonitor == null)
            {
                Debug.LogWarning("[PerformanceMonitorIntegration] PerformanceMonitor not found in scene!");
                return;
            }
            
            // Subscribe to PerformanceMonitor events
            performanceMonitor.OnMetricsUpdated += OnPerformanceMetricsUpdated;
            performanceMonitor.OnPerformanceAlert += OnPerformanceAlertReceived;
            
            Debug.Log("[PerformanceMonitorIntegration] Integration initialized");
        }
        
        /**
         * @brief Starts the backend integration system
         */
        public void StartBackendIntegration()
        {
            if (!enableBackendIntegration) return;
            
            Debug.Log("[PerformanceMonitorIntegration] Starting backend integration...");
            
            // Start WebSocket connection
            ConnectToBackend();
            
            // Start metrics streaming
            if (metricsStreamingCoroutine != null)
            {
                StopCoroutine(metricsStreamingCoroutine);
            }
            metricsStreamingCoroutine = StartCoroutine(StreamMetricsToBackend());
        }
        
        /**
         * @brief Stops the backend integration system
         */
        public void StopBackendIntegration()
        {
            Debug.Log("[PerformanceMonitorIntegration] Stopping backend integration...");
            
            // Stop coroutines
            if (reconnectCoroutine != null)
            {
                StopCoroutine(reconnectCoroutine);
                reconnectCoroutine = null;
            }
            
            if (metricsStreamingCoroutine != null)
            {
                StopCoroutine(metricsStreamingCoroutine);
                metricsStreamingCoroutine = null;
            }
            
            // Close WebSocket connection
            if (webSocket != null)
            {
                webSocket.Close();
                webSocket = null;
            }
            
            // Unsubscribe from PerformanceMonitor events
            if (performanceMonitor != null)
            {
                performanceMonitor.OnMetricsUpdated -= OnPerformanceMetricsUpdated;
                performanceMonitor.OnPerformanceAlert -= OnPerformanceAlertReceived;
            }
            
            isConnected = false;
            isStreaming = false;
            OnBackendConnectionChanged?.Invoke(false);
        }
        
        /**
         * @brief Connects to the Python backend via WebSocket
         */
        private async void ConnectToBackend()
        {
            try
            {
                Debug.Log($"[PerformanceMonitorIntegration] Connecting to backend at {backendWebSocketUrl}");
                
                // Create WebSocket connection
                webSocket = new WebSocket(backendWebSocketUrl);
                
                // Set up event handlers
                webSocket.OnOpen += OnWebSocketOpen;
                webSocket.OnMessage += OnWebSocketMessage;
                webSocket.OnClose += OnWebSocketClose;
                webSocket.OnError += OnWebSocketError;
                
                // Connect
                await webSocket.Connect();
                
            }
            catch (Exception e)
            {
                Debug.LogError($"[PerformanceMonitorIntegration] Failed to connect to backend: {e.Message}");
                ScheduleReconnect();
            }
        }
        
        /**
         * @brief Handles WebSocket connection open event
         */
        private void OnWebSocketOpen()
        {
            Debug.Log("[PerformanceMonitorIntegration] Connected to backend!");
            isConnected = true;
            reconnectAttempts = 0;
            OnBackendConnectionChanged?.Invoke(true);
            
            // Send initial configuration
            SendConfigurationToBackend();
        }
        
        /**
         * @brief Handles WebSocket message events
         */
        private void OnWebSocketMessage(byte[] data)
        {
            try
            {
                string message = Encoding.UTF8.GetString(data);
                var messageData = JsonUtility.FromJson<BackendMessage>(message);
                
                ProcessBackendMessage(messageData);
            }
            catch (Exception e)
            {
                Debug.LogError($"[PerformanceMonitorIntegration] Error processing backend message: {e.Message}");
            }
        }
        
        /**
         * @brief Handles WebSocket connection close event
         */
        private void OnWebSocketClose(WebSocketCloseCode closeCode)
        {
            Debug.Log($"[PerformanceMonitorIntegration] Backend connection closed: {closeCode}");
            isConnected = false;
            OnBackendConnectionChanged?.Invoke(false);
            
            if (enableBackendIntegration)
            {
                ScheduleReconnect();
            }
        }
        
        /**
         * @brief Handles WebSocket error events
         */
        private void OnWebSocketError(string errorMessage)
        {
            Debug.LogError($"[PerformanceMonitorIntegration] WebSocket error: {errorMessage}");
            isConnected = false;
            OnBackendConnectionChanged?.Invoke(false);
        }
        
        /**
         * @brief Schedules a reconnection attempt
         */
        private void ScheduleReconnect()
        {
            if (reconnectAttempts >= maxReconnectAttempts)
            {
                Debug.LogWarning("[PerformanceMonitorIntegration] Max reconnection attempts reached. Stopping integration.");
                return;
            }
            
            reconnectAttempts++;
            Debug.Log($"[PerformanceMonitorIntegration] Scheduling reconnection attempt {reconnectAttempts}/{maxReconnectAttempts} in {reconnectInterval} seconds");
            
            if (reconnectCoroutine != null)
            {
                StopCoroutine(reconnectCoroutine);
            }
            reconnectCoroutine = StartCoroutine(ReconnectCoroutine());
        }
        
        /**
         * @brief Coroutine for handling reconnection attempts
         */
        private IEnumerator ReconnectCoroutine()
        {
            yield return new WaitForSeconds(reconnectInterval);
            
            if (enableBackendIntegration && !isConnected)
            {
                ConnectToBackend();
            }
            
            reconnectCoroutine = null;
        }
        
        /**
         * @brief Sends initial configuration to backend
         */
        private void SendConfigurationToBackend()
        {
            if (!isConnected) return;
            
            var config = new UnityConfiguration
            {
                type = "unity_config",
                data = new UnityConfigData
                {
                    unityVersion = Application.unityVersion,
                    platform = Application.platform.ToString(),
                    targetFrameRate = Application.targetFrameRate,
                    enableAdaptiveQuality = true
                }
            };
            
            SendMessageToBackend(config);
        }
        
        /**
         * @brief Sends a message to the backend
         */
        private void SendMessageToBackend(object message)
        {
            if (!isConnected || webSocket == null) return;
            
            try
            {
                string json = JsonUtility.ToJson(message);
                byte[] data = Encoding.UTF8.GetBytes(json);
                webSocket.Send(data);
            }
            catch (Exception e)
            {
                Debug.LogError($"[PerformanceMonitorIntegration] Error sending message to backend: {e.Message}");
            }
        }
        
        /**
         * @brief Processes messages received from the backend
         */
        private void ProcessBackendMessage(BackendMessage message)
        {
            switch (message.type)
            {
                case "performance_recommendations":
                    ProcessPerformanceRecommendations(message.data);
                    break;
                    
                case "backend_metrics":
                    ProcessBackendMetrics(message.data);
                    break;
                    
                case "quality_change_request":
                    ProcessQualityChangeRequest(message.data);
                    break;
                    
                case "emergency_optimization":
                    ProcessEmergencyOptimization(message.data);
                    break;
                    
                default:
                    Debug.LogWarning($"[PerformanceMonitorIntegration] Unknown message type: {message.type}");
                    break;
            }
        }
        
        /**
         * @brief Processes performance recommendations from backend
         */
        private void ProcessPerformanceRecommendations(string data)
        {
            try
            {
                var recommendations = JsonUtility.FromJson<BackendRecommendations>(data);
                OnBackendRecommendationsReceived?.Invoke(recommendations.recommendations);
                
                Debug.Log($"[PerformanceMonitorIntegration] Received {recommendations.recommendations.Count} performance recommendations from backend");
                
                // Apply critical recommendations immediately
                foreach (var recommendation in recommendations.recommendations)
                {
                    if (recommendation.priority == "critical")
                    {
                        ApplyCriticalRecommendation(recommendation);
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[PerformanceMonitorIntegration] Error processing recommendations: {e.Message}");
            }
        }
        
        /**
         * @brief Processes backend performance metrics
         */
        private void ProcessBackendMetrics(string data)
        {
            try
            {
                var metrics = JsonUtility.FromJson<BackendMetrics>(data);
                OnBackendMetricsReceived?.Invoke(metrics);
                
                // Log backend performance
                Debug.Log($"[PerformanceMonitorIntegration] Backend: CPU {metrics.cpuUsagePercent:F1}%, "
                         $"Memory {metrics.memoryUsageMB:F1} MB, GPU {metrics.gpuUsagePercent:F1}%");
            }
            catch (Exception e)
            {
                Debug.LogError($"[PerformanceMonitorIntegration] Error processing backend metrics: {e.Message}");
            }
        }
        
        /**
         * @brief Processes quality change requests from backend
         */
        private void ProcessQualityChangeRequest(string data)
        {
            try
            {
                var request = JsonUtility.FromJson<QualityChangeRequest>(data);
                Debug.Log($"[PerformanceMonitorIntegration] Backend requested quality change to: {request.quality_level}");
                
                // Notify GameManager of quality change request
                if (GameManager.Instance != null)
                {
                    // Convert string quality level to PerformanceLevel enum
                    if (Enum.TryParse<GameManager.PerformanceLevel>(request.quality_level, true, out var qualityLevel))
                    {
                        GameManager.Instance.SetPerformanceLevel(qualityLevel);
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[PerformanceMonitorIntegration] Error processing quality change request: {e.Message}");
            }
        }
        
        /**
         * @brief Processes emergency optimization requests from backend
         */
        private void ProcessEmergencyOptimization(string data)
        {
            try
            {
                var emergency = JsonUtility.FromJson<EmergencyOptimization>(data);
                Debug.LogWarning($"[PerformanceMonitorIntegration] Emergency optimization triggered: {emergency.reason}");
                
                // Apply emergency optimizations
                ApplyEmergencyOptimizations();
            }
            catch (Exception e)
            {
                Debug.LogError($"[PerformanceMonitorIntegration] Error processing emergency optimization: {e.Message}");
            }
        }
        
        /**
         * @brief Applies critical performance recommendations
         */
        private void ApplyCriticalRecommendation(PerformanceRecommendation recommendation)
        {
            Debug.LogWarning($"[PerformanceMonitorIntegration] Applying critical recommendation: {recommendation.action}");
            
            // Implement critical recommendation logic
            switch (recommendation.action.ToLower())
            {
                case "reduce rendering quality to minimum":
                    if (GameManager.Instance != null)
                    {
                        GameManager.Instance.SetPerformanceLevel(GameManager.PerformanceLevel.Low);
                    }
                    break;
                    
                case "emergency performance mode":
                    ApplyEmergencyOptimizations();
                    break;
                    
                default:
                    Debug.LogWarning($"[PerformanceMonitorIntegration] Unknown critical action: {recommendation.action}");
                    break;
            }
        }
        
        /**
         * @brief Applies emergency performance optimizations
         */
        private void ApplyEmergencyOptimizations()
        {
            Debug.LogWarning("[PerformanceMonitorIntegration] Applying emergency performance optimizations!");
            
            // Set minimum quality level
            if (GameManager.Instance != null)
            {
                GameManager.Instance.SetPerformanceLevel(GameManager.PerformanceLevel.Low);
            }
            
            // Disable non-essential features
            // This would integrate with your existing systems
            // Example: Disable complex shaders, reduce physics complexity, etc.
        }
        
        /**
         * @brief Handles performance metrics updates from PerformanceMonitor
         */
        private void OnPerformanceMetricsUpdated(PerformanceMetrics metrics)
        {
            if (isConnected && isStreaming)
            {
                // Queue metrics for streaming
                metricsQueue.Enqueue(metrics);
                
                // Limit queue size
                if (metricsQueue.Count > 100)
                {
                    metricsQueue.Dequeue();
                }
            }
        }
        
        /**
         * @brief Handles performance alerts from PerformanceMonitor
         */
        private void OnPerformanceAlertReceived(PerformanceAlert alert)
        {
            if (isConnected)
            {
                // Send alert immediately to backend
                var alertMessage = new UnityAlertMessage
                {
                    type = "performance_alert",
                    data = new UnityAlertData
                    {
                        alertType = alert.Type.ToString(),
                        severity = alert.Severity.ToString(),
                        message = alert.Message,
                        currentFPS = alert.CurrentFPS,
                        recommendedAction = alert.RecommendedAction.ToString()
                    }
                };
                
                SendMessageToBackend(alertMessage);
            }
        }
        
        /**
         * @brief Coroutine for streaming metrics to backend
         */
        private IEnumerator StreamMetricsToBackend()
        {
            isStreaming = true;
            
            while (isStreaming)
            {
                if (isConnected && metricsQueue.Count > 0)
                {
                    // Send all queued metrics
                    while (metricsQueue.Count > 0)
                    {
                        var metrics = metricsQueue.Dequeue();
                        var metricsMessage = new UnityMetricsMessage
                        {
                            type = "performance_metrics",
                            data = metrics
                        };
                        
                        SendMessageToBackend(metricsMessage);
                    }
                }
                
                yield return new WaitForSeconds(metricsSendInterval);
            }
        }
        
        /**
         * @brief Manually triggers emergency optimization
         */
        public void TriggerEmergencyOptimization()
        {
            Debug.LogWarning("[PerformanceMonitorIntegration] Manual emergency optimization triggered!");
            ApplyEmergencyOptimizations();
        }
        
        /**
         * @brief Gets current connection status
         */
        public bool IsConnectedToBackend()
        {
            return isConnected;
        }
        
        /**
         * @brief Gets current streaming status
         */
        public bool IsStreamingMetrics()
        {
            return isStreaming;
        }
        
        /**
         * @brief Manually reconnects to backend
         */
        public void ReconnectToBackend()
        {
            if (reconnectCoroutine != null)
            {
                StopCoroutine(reconnectCoroutine);
                reconnectCoroutine = null;
            }
            
            ConnectToBackend();
        }
    }
    
    // Data structures for backend communication
    
    [System.Serializable]
    public class BackendMessage
    {
        public string type;
        public string data;
    }
    
    [System.Serializable]
    public class UnityConfiguration
    {
        public string type;
        public UnityConfigData data;
    }
    
    [System.Serializable]
    public class UnityConfigData
    {
        public string unityVersion;
        public string platform;
        public int targetFrameRate;
        public bool enableAdaptiveQuality;
    }
    
    [System.Serializable]
    public class UnityMetricsMessage
    {
        public string type;
        public PerformanceMetrics data;
    }
    
    [System.Serializable]
    public class UnityAlertMessage
    {
        public string type;
        public UnityAlertData data;
    }
    
    [System.Serializable]
    public class UnityAlertData
    {
        public string alertType;
        public string severity;
        public string message;
        public float currentFPS;
        public string recommendedAction;
    }
    
    [System.Serializable]
    public class BackendRecommendations
    {
        public List<PerformanceRecommendation> recommendations;
    }
    
    [System.Serializable]
    public class BackendMetrics
    {
        public float cpuUsagePercent;
        public float memoryUsageMB;
        public float gpuUsagePercent;
        public float gpuMemoryMB;
        public float networkLatencyMs;
        public float processingTimeMs;
        public int queueSize;
        public int activeThreads;
        public string timestamp;
    }
    
    [System.Serializable]
    public class QualityChangeRequest
    {
        public string quality_level;
    }
    
    [System.Serializable]
    public class EmergencyOptimization
    {
        public string reason;
        public string action;
    }
    
    [System.Serializable]
    public class PerformanceRecommendation
    {
        public string action;
        public string priority;
        public float confidence;
        public float expectedImprovement;
        public string implementation;
        public string timestamp;
    }
    
    // Simple WebSocket implementation (you may want to use a proper WebSocket library)
    public class WebSocket
    {
        public event Action OnOpen;
        public event Action<byte[]> OnMessage;
        public event Action<WebSocketCloseCode> OnClose;
        public event Action<string> OnError;
        
        private string url;
        private bool isConnected = false;
        
        public WebSocket(string url)
        {
            this.url = url;
        }
        
        public async Task Connect()
        {
            // This is a placeholder - you'll need to implement actual WebSocket connection
            // Consider using NativeWebSocket, WebSocketSharp, or similar library
            await Task.Delay(100);
            isConnected = true;
            OnOpen?.Invoke();
        }
        
        public void Send(byte[] data)
        {
            if (!isConnected) return;
            // Implement actual WebSocket send
        }
        
        public void Close()
        {
            if (!isConnected) return;
            isConnected = false;
            OnClose?.Invoke(WebSocketCloseCode.Normal);
        }
    }
    
    public enum WebSocketCloseCode
    {
        Normal = 1000,
        GoingAway = 1001,
        ProtocolError = 1002,
        UnsupportedData = 1003,
        NoStatusReceived = 1005,
        AbnormalClosure = 1006,
        InvalidFramePayloadData = 1007,
        PolicyViolation = 1008,
        MessageTooBig = 1009,
        InternalError = 1011
    }
}
