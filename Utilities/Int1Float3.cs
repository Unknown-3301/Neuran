using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Utilities
{
    public struct Int1Float3
    {
        public int int1;
        public float float1, float2, float3;

        public static int Size { get => sizeof(int) + sizeof(float) * 3; }
    }
}
