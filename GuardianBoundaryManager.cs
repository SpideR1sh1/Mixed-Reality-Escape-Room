using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Oculus;
public class GuardianBoundaryManager : MonoBehaviour
{
    public Material caveMaterial; // Assign a cave-like material in the inspector

    void Start()
    {
        if (OVRManager.boundary.GetConfigured())
        {
            Vector3 dimensions = OVRManager.boundary.GetDimensions(OVRBoundary.BoundaryType.PlayArea);
            CreateCaveWalls(dimensions);
        }
        else
        {
            Debug.LogWarning("Guardian boundary not configured!");
        }
    }

    void CreateCaveWalls(Vector3 dimensions)
    {
        // The dimensions provide the width (x) and depth (z). Height (y) can be set manually.
        float height = 3.0f; // Height of the walls

        // Create walls at the four corners of the play area
        Vector3[] corners = new Vector3[]
        {
            new Vector3(-dimensions.x / 2, 0, -dimensions.z / 2),
            new Vector3(dimensions.x / 2, 0, -dimensions.z / 2),
            new Vector3(dimensions.x / 2, 0, dimensions.z / 2),
            new Vector3(-dimensions.x / 2, 0, dimensions.z / 2)
        };

        for (int i = 0; i < corners.Length; i++)
        {
            GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            wall.transform.position = corners[i] + new Vector3(0, height / 2, 0); // Centering the wall
            wall.transform.localScale = new Vector3(0.1f, height, dimensions.z); // Assume thin walls with appropriate height and length
            wall.GetComponent<Renderer>().material = caveMaterial;

            // Rotate wall to line up properly except for the first and last
            if (i < corners.Length - 1)
                wall.transform.LookAt(corners[i + 1]);
            else
                wall.transform.LookAt(corners[0]);
        }
    }
}

