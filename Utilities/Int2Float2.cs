using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Utilities
{
    public struct Int2Float2
    {
        public int int1, int2;
        public float float1, float2;

        public static int Size { get => sizeof(int) * 2 + sizeof(float) * 2; }
    }
}
