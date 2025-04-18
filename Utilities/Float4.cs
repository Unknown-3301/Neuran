using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Utilities
{
    public struct Float4
    {
        public float float1, float2, float3, float4;

        public static int Size { get => sizeof(float) * 4; }
    }
}
