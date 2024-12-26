using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class DepthToMeshConverter : MonoBehaviour
{
    MeshFilter meshFilter;

    void Start()
    {
        meshFilter = GetComponent<MeshFilter>();
        meshFilter.mesh = GenerateMeshFromDepthData(GetDepthData());
    }

    // Placeholder for your depth data acquisition method
    Vector3[] GetDepthData()
    {
        // Your method for getting depth data would go here.
        // This would typically return an array of Vector3 points.

        // Example of returning some dummy data to represent depth points
        return new Vector3[] {
            new Vector3(0, 0, 0),
            new Vector3(1, 0, 0),
            new Vector3(0, 1, 0),
            new Vector3(1, 1, 0)
        };
    }

    Mesh GenerateMeshFromDepthData(Vector3[] depthPoints)
    {
        // Create a new mesh
        Mesh mesh = new Mesh();

        // Assign vertices from the depth data
        mesh.vertices = depthPoints;

        // Automatically generate triangles - this is naive and for illustrative purposes only
        int[] triangles = new int[(depthPoints.Length - 1) * 3];
        for (int i = 0, t = 0; i < depthPoints.Length - 1; i++, t += 3)
        {
            triangles[t] = i;
            triangles[t + 1] = i + 1;
            triangles[t + 2] = i + 2;
        }

        mesh.triangles = triangles;

        // Recalculate the normals of the mesh
        mesh.RecalculateNormals();

        return mesh;
    }
}
