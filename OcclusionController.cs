/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Licensed under the Oculus SDK License Agreement (the "License");
 * you may not use the Oculus SDK except in compliance with the License,
 * which is provided at the time of installation or download, or which
 * otherwise accompanies this software in either electronic or hard copy form.
 *
 * You may obtain a copy of the License at
 *
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Oculus SDK
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using Meta.XR.Depth;
using UnityEngine;

namespace DepthAPISample
{
    public class OcclusionController : MonoBehaviour
    {
        [SerializeField]
        private Renderer _renderer;

        // This is now public or provided via a method.
        public OcclusionType OcclusionType { get; private set; } = OcclusionType.NoOcclusion;

        private void Awake()
        {
            _renderer = GetComponent<Renderer>();
            UpdateMaterialKeywords();
        }

        public void SetOcclusionType(OcclusionType occlusionType)
        {
            this.OcclusionType = occlusionType;
            UpdateMaterialKeywords();
        }

        private void UpdateMaterialKeywords()
        {
            Material mat = _renderer.material;

            // Reset keywords
            mat.DisableKeyword(EnvironmentDepthOcclusionController.SoftOcclusionKeyword);
            mat.DisableKeyword(EnvironmentDepthOcclusionController.HardOcclusionKeyword);

            // Enable appropriate keyword
            switch (OcclusionType)
            {
                case OcclusionType.SoftOcclusion:
                    mat.EnableKeyword(EnvironmentDepthOcclusionController.SoftOcclusionKeyword);
                    break;
                case OcclusionType.HardOcclusion:
                    mat.EnableKeyword(EnvironmentDepthOcclusionController.HardOcclusionKeyword);
                    break;
            }
        }
    }
}

