using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CaveWallGenerator : MonoBehaviour
{
    public GameObject[] rockPrefabs; // Assign your rock prefabs in the inspector
    public int numberOfRocks; // The number of rocks you want to place

    private Bounds bounds; // The bounds of the room model

    void Start()
    {
        // Assuming there's a MeshFilter component attached to the same GameObject as this script
        MeshFilter meshFilter = GetComponent<MeshFilter>();
        if (meshFilter)
        {
            bounds = meshFilter.mesh.bounds;
            bounds.size = Vector3.Scale(bounds.size, transform.localScale); // Adjust bounds to the scale of the gameObject
            GenerateCaveWalls();
        }
        else
        {
            Debug.LogError("MeshFilter not found on the Room Model.");
        }
    }

    void GenerateCaveWalls()
    {
        for (int i = 0; i < numberOfRocks; i++)
        {
            // Select a random prefab
            GameObject rockPrefab = rockPrefabs[Random.Range(0, rockPrefabs.Length)];
            // Calculate a random position within the wall bounds
            Vector3 position = new Vector3(
                Random.Range(-bounds.extents.x, bounds.extents.x),
                Random.Range(-bounds.extents.y, bounds.extents.y),
                Random.Range(-bounds.extents.z, bounds.extents.z)
            ) + bounds.center;

            position = transform.TransformPoint(position); // Convert local position to world position

            // Instantiate the rock at the random position
            GameObject rockInstance = Instantiate(rockPrefab, position, Random.rotation);
            rockInstance.transform.SetParent(this.transform); // Optional: Set the wall as the parent of the rock

            // Optional: Scale the rock randomly for more variety
            float scale = Random.Range(0.8f, 1.2f); // Random scale factor
            rockInstance.transform.localScale *= scale;
        }
    }
}
