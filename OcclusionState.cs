using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Meta.XR.Depth;


public class OcclusionState : MonoBehaviour
{
    public Renderer Renderer { get; private set; }
    public OcclusionType CurrentOcclusionType { get; private set; } = OcclusionType.NoOcclusion;

    private void Awake()
    {
        Renderer = GetComponent<Renderer>();
        if (!Renderer)
        {
            Debug.LogError("OcclusionState requires a Renderer component to function.", this);
        }
        else
        {
            // This creates a unique material instance for this object
            Renderer.material = new Material(Renderer.sharedMaterial);
            UpdateMaterialKeywords(); // Ensure occlusion is set on start.
        }
    }

    public void SetOcclusionType(OcclusionType type)
    {
        CurrentOcclusionType = type;
        UpdateMaterialKeywords();
    }

    private void UpdateMaterialKeywords()
    {
        if (!Renderer) return;

        Material mat = Renderer.material;

        // Disable all keywords initially
        mat.DisableKeyword(EnvironmentDepthOcclusionController.HardOcclusionKeyword);
        mat.DisableKeyword(EnvironmentDepthOcclusionController.SoftOcclusionKeyword);

        // Enable the keyword based on the current occlusion state
        switch (CurrentOcclusionType)
        {
            case OcclusionType.HardOcclusion:
                mat.EnableKeyword(EnvironmentDepthOcclusionController.HardOcclusionKeyword);
                break;
            case OcclusionType.SoftOcclusion:
                mat.EnableKeyword(EnvironmentDepthOcclusionController.SoftOcclusionKeyword);
                break;
        }

        // Apply the changes back to the renderer.
        Renderer.material = mat;
    }
}
