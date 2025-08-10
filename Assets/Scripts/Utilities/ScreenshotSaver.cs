/**
 * @file ScreenshotSaver.cs
 * @brief Advanced screenshot capture and gallery integration system for Mixed Reality applications
 * @author Mixed Reality Escape Room Development Team
 * @date December 2024
 * @version 1.0
 * 
 * @description
 * This class implements a comprehensive screenshot capture and gallery integration system
 * designed specifically for Mixed Reality applications on Android platforms. It provides
 * high-quality screenshot capture, automatic file management, and seamless integration
 * with the device's media gallery. The system supports both local storage and gallery
 * integration for enhanced user experience and content sharing capabilities.
 * 
 * @features
 * - High-quality screenshot capture using Unity's ScreenCapture system
 * - Automatic directory creation and file management
 * - Timestamp-based file naming for unique screenshot identification
 * - Android media scanner integration for gallery visibility
 * - Coroutine-based screenshot processing for non-blocking operation
 * - End-of-frame rendering synchronization for complete visual capture
 * - Persistent data path utilization for reliable storage
 * - Debug logging for development and troubleshooting
 * - Cross-platform compatibility with Android-specific optimizations
 * 
 * @screenshot_process
 * The screenshot capture system operates through the following workflow:
 * 1. User initiates screenshot capture request
 * 2. System waits for end-of-frame rendering completion
 * 3. Screenshot is captured using Unity's ScreenCapture API
 * 4. File is saved to persistent data directory with timestamp naming
 * 5. Android media scanner is invoked for gallery integration
 * 6. Screenshot becomes visible in device's photo gallery
 * 
 * @file_management
 * - Automatic screenshots directory creation in persistent data path
 * - Timestamp-based file naming (yyyyMMddHHmmss format)
 * - PNG format for high-quality image preservation
 * - Unique file identification to prevent overwrites
 * - Persistent storage for reliable file retention
 * 
 * @android_integration
 * - Media scanner integration for automatic gallery detection
 * - Native Android Java class utilization for platform-specific features
 * - Context-aware media scanning for proper file indexing
 * - Gallery visibility without manual file management
 * 
 * @dependencies
 * - Unity Engine 2022.3 LTS or newer
 * - Android platform for media scanner integration
 * - System.IO for file system operations
 * - Unity.Android for platform-specific functionality
 * - Coroutine system for asynchronous processing
 * 
 * @license MIT License
 */

using UnityEngine;
using System.Collections;
using System.IO;
using UnityEngine.Android;

namespace MREscapeRoom.Utilities
{
    /**
     * @class ScreenshotSaver
     * @brief Advanced screenshot capture and gallery integration system
     * 
     * @details
     * The ScreenshotSaver class provides comprehensive functionality for capturing
     * high-quality screenshots in Mixed Reality applications and integrating them
     * with the device's media gallery. It implements automatic file management,
     * timestamp-based naming, and Android-specific media scanner integration
     * for seamless user experience.
     * 
     * @implements MonoBehaviour - Unity component lifecycle management
     * @features Coroutine-based screenshot processing
     * @features Android media scanner integration
     * @features Automatic file management and organization
     * @features High-quality image capture and storage
     */
    public class ScreenshotSaver : MonoBehaviour
    {
        [Header("Screenshot Configuration")]
        private string screenshotsDirectory = Path.Combine(Application.persistentDataPath, "Screenshots"); /**< Directory path for screenshot storage */
        
        /**
         * @brief Unity lifecycle method for component initialization
         * 
         * @details
         * Initializes the screenshot saver component by ensuring the screenshots
         * directory exists. This method creates the directory structure if it
         * doesn't exist, ensuring proper file organization and storage capability.
         */
        private void Awake()
        {
            // Create screenshots directory if it doesn't exist
            if (!Directory.Exists(screenshotsDirectory))
            {
                Directory.CreateDirectory(screenshotsDirectory);
            }
        }

        /**
         * @brief Public interface for initiating screenshot capture
         * 
         * @details
         * Initiates the screenshot capture process by starting the coroutine-based
         * screenshot workflow. This method provides a public interface for external
         * systems to trigger screenshot capture without blocking the main thread.
         */
        public void SaveScreenshotToGallery()
        {
            StartCoroutine(TakeScreenshotAndSave());
        }

        /**
         * @brief Coroutine-based screenshot capture and save workflow
         * 
         * @details
         * Implements the complete screenshot capture workflow using coroutines
         * for non-blocking operation. This method ensures proper rendering
         * completion, generates unique filenames, and coordinates the entire
         * screenshot process from capture to gallery integration.
         * 
         * @workflow
         * 1. Wait for end-of-frame rendering completion
         * 2. Generate timestamp-based filename
         * 3. Capture screenshot using Unity's ScreenCapture API
         * 4. Wait for file system write completion
         * 5. Integrate with Android media gallery
         * 
         * @returns IEnumerator for coroutine execution
         */
        private IEnumerator TakeScreenshotAndSave()
        {
            // Wait for end of frame to ensure all rendering is complete
            yield return new WaitForEndOfFrame();

            // Generate unique filename with timestamp
            string fileName = $"Screenshot_{System.DateTime.Now:yyyyMMddHHmmss}.png";
            string filePath = Path.Combine(screenshotsDirectory, fileName);

            // Capture screenshot using Unity's ScreenCapture system
            ScreenCapture.CaptureScreenshot(fileName);

            // Wait for screenshot file to be written to disk
            yield return new WaitForSeconds(1f);

            // Integrate captured image with Android media gallery
            AddImageToGallery(filePath);
        }

        /**
         * @brief Integrates captured image with Android media gallery
         * 
         * @details
         * Utilizes Android's native media scanner to integrate the captured
         * screenshot with the device's media gallery. This method employs
         * Android Java classes to ensure proper file indexing and gallery
         * visibility without requiring manual file management.
         * 
         * @param filePath Full path to the captured screenshot file
         * @process
         * 1. Initialize Android media scanner connection
         * 2. Retrieve Unity player and current activity context
         * 3. Invoke media scanner for file indexing
         * 4. Log successful gallery integration
         */
        private void AddImageToGallery(string filePath)
        {
            // Initialize Android media scanner integration
            using (AndroidJavaClass mediaScannerConnection = new AndroidJavaClass("android.media.MediaScannerConnection"))
            using (AndroidJavaClass unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
            using (AndroidJavaObject currentActivity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity"))
            using (AndroidJavaObject context = currentActivity.Call<AndroidJavaObject>("getApplicationContext"))
            {
                // Invoke media scanner to index the captured image
                mediaScannerConnection.CallStatic("scanFile", context, new string[] { filePath }, null, null);
            }

            // Log successful gallery integration for debugging
            Debug.Log($"Image saved to gallery: {filePath}");
        }
    }
}
