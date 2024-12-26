using System.Collections.Generic;
using UnityEngine;
using TMPro;
using Meta.XR.Depth;
using DepthAPISample;

namespace DepthAPISample
{

    [System.Serializable]
    public struct PlaceableObject
    {
        public GameObject prefab;
        public GameObject previewPrefab;
    }


    public class ObjPlacer : MonoBehaviour
    {
        [SerializeField] private Transform _rayOrigin;
        [SerializeField] private GameObject _cubePrefab;
        [SerializeField] private LineRenderer _lineRenderer;
        [SerializeField] private float _initialRayDistance = 10f;
        [SerializeField] private OVRInput.RawButton _objectPlacingButton = OVRInput.RawButton.RIndexTrigger;
        [SerializeField] private OVRInput.RawButton _toggleLaserButton = OVRInput.RawButton.B;
        [SerializeField] private GameObject _previewObjectPrefab;
        [SerializeField] private OVRInput.RawButton _changeObjectButton = OVRInput.RawButton.X; 
        [SerializeField] private PlaceableObject[] _placeableObjects;
        [SerializeField] private TextMeshProUGUI occlusionsModeText;
        [SerializeField] private OVRInput.RawButton occlusionChangeButton = OVRInput.RawButton.Y;


        [SerializeField] private Material highlightMaterial;
        private GameObject currentlyHighlighted;
        private Material originalMaterial;

        private List<GameObject> _placedObjects;
        private Vector3 _rayDirection;
        private bool _isLaserActive = true;
        private GameObject _previewObject;
        private float _currentRayDistance;
        private Vector2 _currentRotation;
        private int _currentPrefabIndex = 0;
        private int _currentObjectIndex;




        private void Awake()
        {
            _placedObjects = new List<GameObject>();
            _rayDirection = _rayOrigin.forward;
            _currentRayDistance = _initialRayDistance;
            _currentObjectIndex = 0;
            CreatePreviewObject();
        }

        private void Update()
        {
            HandleThumbstickInput();
            ToggleLaser();
            UpdateRayAndPreviewPosition();

            if (_isLaserActive)
            {
                if (OVRInput.GetDown(_changeObjectButton))
                {
                    CyclePreviewObject();
                }
                if (OVRInput.GetDown(_objectPlacingButton))
                {
                    AttemptObjectPlacement();
                }
                if (OVRInput.GetDown(occlusionChangeButton))
                {
                    AttemptToggleOcclusion();
                }
            }

        }


        private void AttemptToggleOcclusion()
        {
            RaycastHit hit;
            // Check for raycast hit
            if (Physics.Raycast(_rayOrigin.position, _rayOrigin.forward, out hit, _currentRayDistance))
            {

                if (hit.collider.gameObject.CompareTag("Interactable") && OVRInput.GetDown(occlusionChangeButton))
                {

                    OcclusionState occlusionState = hit.collider.GetComponent<OcclusionState>();
                    if (occlusionState != null)
                    {

                        OcclusionType nextOcclusionType = GetNextOcclusionType(occlusionState.CurrentOcclusionType);


                        occlusionState.SetOcclusionType(nextOcclusionType);
                    }
                }
            }
        }

        private void ToggleOcclusion(Renderer renderer)
        {
            OcclusionState occlusionState = renderer.GetComponent<OcclusionState>();
            if (occlusionState == null) return;


            OcclusionType nextOcclusionType = GetNextOcclusionType(occlusionState.CurrentOcclusionType);
            occlusionState.SetOcclusionType(nextOcclusionType);


            occlusionsModeText.text = $"Occlusion Mode: {nextOcclusionType}";
        }

        private OcclusionType GetNextOcclusionType(OcclusionType currentType)
        {
            switch (currentType)
            {
                case OcclusionType.NoOcclusion:
                    return OcclusionType.HardOcclusion;
                case OcclusionType.SoftOcclusion:
                    return OcclusionType.NoOcclusion;
                case OcclusionType.HardOcclusion:
                    return OcclusionType.SoftOcclusion;
                default:
                    return OcclusionType.NoOcclusion;
            }
        }

        private void CreatePreviewObject()
        {
            if (_placeableObjects.Length > 0)
            {
                _previewObject = Instantiate(_placeableObjects[_currentObjectIndex].previewPrefab);
                _previewObject.SetActive(false);
            }
        }


        private void HandleThumbstickInput()
        {

            Vector2 rotationInput = OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick);
            _currentRotation += new Vector2(rotationInput.x * 100f * Time.deltaTime, rotationInput.y * 100f * Time.deltaTime); 

            float lengthInput = OVRInput.Get(OVRInput.Axis2D.SecondaryThumbstick).y;
            _currentRayDistance += lengthInput * Time.deltaTime * 5f; 
            _currentRayDistance = Mathf.Clamp(_currentRayDistance, _initialRayDistance, 100f);
        }


        private void ToggleLaser()
        {
            if (OVRInput.GetDown(_toggleLaserButton))
            {
                _isLaserActive = !_isLaserActive;
                _lineRenderer.enabled = _isLaserActive;
                if (_previewObject != null)
                {
                    _previewObject.SetActive(_isLaserActive);
                }
            }
        }

        private void UpdateRayAndPreviewPosition()
        {
            Vector3 rayEndPoint = _rayOrigin.position + _rayOrigin.forward * _currentRayDistance;
            RaycastHit hit;

            // Perform the raycast
            if (Physics.Raycast(_rayOrigin.position, _rayOrigin.forward, out hit, _currentRayDistance))
            {

                if (hit.collider.gameObject.CompareTag("Interactable") && _isLaserActive)
                {

                    HighlightObject(hit.collider.gameObject);


                    if (_previewObject.activeSelf)
                    {
                        _previewObject.SetActive(false);
                    }


                    OcclusionState occlusionState = hit.collider.GetComponent<OcclusionState>();
                    if (occlusionState != null)
                    {
                        UpdateOcclusionModeText(occlusionState.CurrentOcclusionType);
                    }
                    else
                    {
                        occlusionsModeText.text = "Occlusion mode: N/A";
                    }
                }
                else
                {
                    // If the raycast hits a non-interactable object or nothing at all, clear the occlusion mode text
                    occlusionsModeText.text = "Occlusion mode: N/A";
                    PositionPreviewObject(rayEndPoint);
                    RemoveHighlight();
                }
            }
            else
            {

                occlusionsModeText.text = "Occlusion mode: N/A";
                PositionPreviewObject(rayEndPoint);
                RemoveHighlight();
            }


            _lineRenderer.SetPositions(new Vector3[] { _rayOrigin.position, rayEndPoint });
            _lineRenderer.startColor = _lineRenderer.endColor = Color.green;
        }


        private void UpdateOcclusionModeText(OcclusionType occlusionType)
        {

            occlusionsModeText.text = $"Occlusion Mode: {occlusionType}";
        }


        private void PositionPreviewObject(Vector3 position)
        {
            if (!_previewObject.activeSelf && _isLaserActive)
            {
                _previewObject.SetActive(true);
            }
            _previewObject.transform.position = position;
            _previewObject.transform.rotation = Quaternion.Euler(-_currentRotation.y, _currentRotation.x, 0);
        }


        private void HighlightObject(GameObject obj)
        {

            if (currentlyHighlighted == obj) return;


            RemoveHighlight();


            currentlyHighlighted = obj;
            var renderer = currentlyHighlighted.GetComponent<Renderer>();
            originalMaterial = renderer.material;
            renderer.material = highlightMaterial;
        }

        private void RemoveHighlight()
        {
            if (currentlyHighlighted != null)
            {

                var renderer = currentlyHighlighted.GetComponent<Renderer>();
                renderer.material = originalMaterial;
                currentlyHighlighted = null;
            }
        }




        private void CyclePreviewObject()
        {

            if (_previewObject != null)
                Destroy(_previewObject);


            _currentObjectIndex = (_currentObjectIndex + 1) % _placeableObjects.Length;
            _previewObject = Instantiate(_placeableObjects[_currentObjectIndex].previewPrefab);
            _previewObject.SetActive(_isLaserActive);
        }


        private void AttemptObjectPlacement()
        {
            if (_previewObject != null && _previewObject.activeSelf)
            {
                Vector3 placementPosition = _previewObject.transform.position;
                Quaternion placementRotation = Quaternion.Euler(-_currentRotation.y, _currentRotation.x, 0);


                GameObject placedObject = Instantiate(_placeableObjects[_currentObjectIndex].prefab, placementPosition, placementRotation);
                _placedObjects.Add(placedObject);
            }
        }


    }
}