﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Utilities
{
    public struct Int8
    {
        public int int1, int2, int3, int4, int5, int6, int7, int8;

        public static int Size { get => sizeof(int) * 8; }
    }
}
