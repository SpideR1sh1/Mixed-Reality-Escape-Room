using UnityEngine;
using UnityEngine.UI;
using System.Collections;


public class Timer : MonoBehaviour
{
    public TMPro.TextMeshProUGUI timerText; // For TextMeshPro
    private float startTime;
    private bool isRunning = false;

    void Start()
    {
        // Set the start time for 5 minutes (300 seconds)
        startTime = 300.0f;
        isRunning = true;
    }

    void Update()
    {
        if (isRunning)
        {
            startTime -= Time.deltaTime;
            string minutes = ((int)(startTime / 60)).ToString("00");
            string seconds = (startTime % 60).ToString("00");
            timerText.text = minutes + ":" + seconds;

            // Stop the timer if it reaches zero
            if (startTime <= 0)
            {
                timerText.text = "00:00";
                isRunning = false;
                // Add any additional actions when the time is up
                TimeUp();
            }
        }
    }

    void TimeUp()
    {
        // Actions to perform when the timer ends
        Debug.Log("Time's up!");
    }
}
