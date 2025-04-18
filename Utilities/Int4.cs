using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Utilities
{
    public struct Int4
    {
        public int int1, int2, int3, int4;

        public static int Size { get => sizeof(int) * 4; }
    }
}
