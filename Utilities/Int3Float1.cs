using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Utilities
{
    public struct Int3Float1
    {
        public int int1, int2, int3;
        public float float1;

        public static int Size { get => sizeof(int) * 3 + sizeof(float); }
    }
}
