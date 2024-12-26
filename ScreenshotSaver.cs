using UnityEngine;
using System.Collections;
using System.IO;
using UnityEngine.Android;

public class ScreenshotSaver : MonoBehaviour
{
    private string screenshotsDirectory = Path.Combine(Application.persistentDataPath, "Screenshots");

    private void Awake()
    {
        if (!Directory.Exists(screenshotsDirectory))
        {
            Directory.CreateDirectory(screenshotsDirectory);
        }
    }

    public void SaveScreenshotToGallery()
    {
        StartCoroutine(TakeScreenshotAndSave());
    }

    private IEnumerator TakeScreenshotAndSave()
    {
        yield return new WaitForEndOfFrame(); // Wait for the end of the frame to ensure all rendering is done

        string fileName = $"Screenshot_{System.DateTime.Now:yyyyMMddHHmmss}.png";
        string filePath = Path.Combine(screenshotsDirectory, fileName);

        ScreenCapture.CaptureScreenshot(fileName);

        // Wait for the screenshot to be saved
        yield return new WaitForSeconds(1f);

        // The screenshot has been saved to the local file system, now let's save it to the gallery
        AddImageToGallery(filePath);
    }

    private void AddImageToGallery(string filePath)
    {
        using (AndroidJavaClass mediaScannerConnection = new AndroidJavaClass("android.media.MediaScannerConnection"))
        using (AndroidJavaClass unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
        using (AndroidJavaObject currentActivity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity"))
        using (AndroidJavaObject context = currentActivity.Call<AndroidJavaObject>("getApplicationContext"))
        {
            mediaScannerConnection.CallStatic("scanFile", context, new string[] { filePath }, null, null);
        }

        Debug.Log($"Image saved to gallery: {filePath}");
    }
}
