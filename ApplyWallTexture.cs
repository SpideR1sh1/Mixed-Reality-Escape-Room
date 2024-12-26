using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ApplyWallTexture : MonoBehaviour
{
    public Material wallMaterial;  // Assign this in the inspector

    void Start()
    {
        Renderer[] renderers = GetComponentsInChildren<Renderer>();
        foreach (Renderer rend in renderers)
        {
            rend.material = wallMaterial;
        }
    }
}

