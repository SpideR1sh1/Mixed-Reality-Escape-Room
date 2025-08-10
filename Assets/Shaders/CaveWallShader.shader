Shader "Custom/CaveWallShader"
{
    Properties
    {
        _MainTex("Albedo (RGB)", 2D) = "white" {}
        _BumpMap("Normal Map", 2D) = "bump" {}
        _Displacement("Displacement Amount", Float) = 0.5
        _Color("Color", Color) = (1,1,1,1)
        _Metallic("Metallic", Range(0,1)) = 0.5
        _Glossiness("Smoothness", Range(0,1)) = 0.5
    }
        SubShader
        {
            Tags { "RenderType" = "Opaque" }
            LOD 200

            CGPROGRAM
            #pragma surface surf Standard fullforwardshadows
            #pragma target 4.0
            #pragma vertex vert

            #include "UnityCG.cginc"

            struct Input
            {
                float2 uv_MainTex;
            };

            sampler2D _MainTex;
            sampler2D _BumpMap;
            float _Displacement;
            fixed4 _Color;
            float _Metallic;
            float _Glossiness;

            // Vertex displacement function
            void vert(inout appdata_full v)
            {
                // Generate a random value based on the vertex position
                float rnd = frac(sin(dot(v.vertex.xyz ,float3(12.9898,78.233,45.5432))) * 43758.5453);

                // Displace the vertex along its normal
                v.vertex.xyz += v.normal * rnd * _Displacement;
            }

            void surf(Input IN, inout SurfaceOutputStandard o)
            {
                // Albedo comes from a texture tinted by color
                fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * _Color;
                o.Albedo = c.rgb;

                // Normal map
                fixed3 normalTex = UnpackNormal(tex2D(_BumpMap, IN.uv_MainTex));
                o.Normal = normalTex;

                // Metallic and smoothness come from declared properties
                o.Metallic = _Metallic;
                o.Smoothness = _Glossiness;
                o.Alpha = c.a;
            }
            ENDCG
        }
            FallBack "Standard"
}
