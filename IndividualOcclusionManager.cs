using UnityEngine;
using TMPro;
using System.Reflection;
using Meta.XR.Depth;
using DepthAPISample;

public class IndividualOcclusionManager : MonoBehaviour
{
    [SerializeField] private Transform rayOrigin;
    [SerializeField] private TextMeshProUGUI occlusionsModeText;
    [SerializeField] private OVRInput.RawButton occlusionChangeButton = OVRInput.RawButton.Y;

    private OcclusionController _currentOcclusionController;

    void Update()
    {
        if (OVRInput.GetDown(occlusionChangeButton))
        {
            RaycastHit hit;
            if (Physics.Raycast(rayOrigin.position, rayOrigin.forward, out hit))
            {
                _currentOcclusionController = hit.collider.GetComponent<OcclusionController>();
                if (_currentOcclusionController != null)
                {
                    // Use reflection to change the private _occlusionType field
                    FieldInfo occlusionTypeField = typeof(OcclusionController).GetField("_occlusionType", BindingFlags.NonPublic | BindingFlags.Instance);
                    if (occlusionTypeField != null)
                    {
                        OcclusionType currentType = (OcclusionType)occlusionTypeField.GetValue(_currentOcclusionController);
                        OcclusionType nextType = GetNextOcclusionType(currentType);
                        occlusionTypeField.SetValue(_currentOcclusionController, nextType);

                        // Invoke UpdateMaterialKeywords to refresh the material
                        MethodInfo updateKeywordsMethod = typeof(OcclusionController).GetMethod("UpdateMaterialKeywords", BindingFlags.NonPublic | BindingFlags.Instance);
                        updateKeywordsMethod?.Invoke(_currentOcclusionController, null);

                        // Update the UI text to display the new occlusion state
                        occlusionsModeText.text = $"Occlusion mode: {nextType}";
                    }
                }
            }
        }
    }

    private OcclusionType GetNextOcclusionType(OcclusionType currentType)
    {
        switch (currentType)
        {
            case OcclusionType.NoOcclusion:
                return OcclusionType.SoftOcclusion;
            case OcclusionType.SoftOcclusion:
                return OcclusionType.HardOcclusion;
            case OcclusionType.HardOcclusion:
                return OcclusionType.NoOcclusion;
            default:
                return OcclusionType.NoOcclusion;
        }
    }
}
