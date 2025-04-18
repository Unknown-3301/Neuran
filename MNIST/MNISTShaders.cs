﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.MNIST
{
    internal class MNISTShaders
    {
        /*
        #pragma kernel CSMain

        RWTexture3D<unorm float> Input : register(u0);
        RWTexture3D<float> Output : register(u1);

        [numthreads(8, 8, 1)]
        void CSMain(uint3 id : SV_DispatchThreadID)
        {
            uint width, height, depth;
            Input.GetDimensions(width, height, depth);

            if (id.x >= width || id.y >= height || id.z >= depth)
            {
                return;
            }

            Output[id] = Input[id];
        }
        */
        public static byte[] ByteConverter = new byte[]
        {
68, 88, 66, 67, 5, 32, 232, 29, 66, 159, 99, 69, 96, 139, 110, 45, 1, 252, 230, 156, 1, 0, 0, 0
            , 196, 2, 0, 0, 5, 0, 0, 0, 52, 0, 0, 0, 240, 0, 0, 0, 0, 1, 0, 0, 16, 1, 0
            , 0, 40, 2, 0, 0, 82, 68, 69, 70, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0
            , 0, 0, 60, 0, 0, 0, 0, 5, 83, 67, 0, 1, 0, 0, 137, 0, 0, 0, 82, 68, 49, 49, 60
            , 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 36, 0, 0, 0, 12, 0, 0, 0
            , 0, 0, 0, 0, 124, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 8, 0, 0, 0, 255, 255, 255
            , 255, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 130, 0, 0, 0, 4, 0, 0, 0, 5, 0
            , 0, 0, 8, 0, 0, 0, 255, 255, 255, 255, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 73
            , 110, 112, 117, 116, 0, 79, 117, 116, 112, 117, 116, 0, 77, 105, 99, 114, 111, 115, 111, 102, 116, 32, 40
            , 82, 41, 32, 72, 76, 83, 76, 32, 83, 104, 97, 100, 101, 114, 32, 67, 111, 109, 112, 105, 108, 101, 114
            , 32, 49, 48, 46, 49, 0, 171, 171, 171, 73, 83, 71, 78, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0
            , 0, 0, 79, 83, 71, 78, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 83, 72, 69, 88, 16
            , 1, 0, 0, 80, 0, 5, 0, 68, 0, 0, 0, 106, 8, 0, 1, 156, 40, 0, 4, 0, 224, 17, 0
            , 0, 0, 0, 0, 17, 17, 0, 0, 156, 40, 0, 4, 0, 224, 17, 0, 1, 0, 0, 0, 85, 85, 0
            , 0, 95, 0, 0, 2, 114, 0, 2, 0, 104, 0, 0, 2, 1, 0, 0, 0, 155, 0, 0, 4, 8, 0
            , 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 61, 16, 0, 137, 66, 1, 0, 128, 67, 68, 4, 0, 114
            , 0, 16, 0, 0, 0, 0, 0, 1, 64, 0, 0, 0, 0, 0, 0, 70, 238, 17, 0, 0, 0, 0, 0
            , 80, 0, 0, 6, 114, 0, 16, 0, 0, 0, 0, 0, 70, 2, 2, 0, 70, 2, 16, 0, 0, 0, 0
            , 0, 60, 0, 0, 7, 18, 0, 16, 0, 0, 0, 0, 0, 26, 0, 16, 0, 0, 0, 0, 0, 10, 0
            , 16, 0, 0, 0, 0, 0, 60, 0, 0, 7, 18, 0, 16, 0, 0, 0, 0, 0, 42, 0, 16, 0, 0
            , 0, 0, 0, 10, 0, 16, 0, 0, 0, 0, 0, 31, 0, 4, 3, 10, 0, 16, 0, 0, 0, 0, 0
            , 62, 0, 0, 1, 21, 0, 0, 1, 163, 0, 0, 136, 66, 1, 0, 128, 67, 68, 4, 0, 18, 0, 16
            , 0, 0, 0, 0, 0, 70, 10, 2, 0, 70, 238, 17, 0, 0, 0, 0, 0, 164, 0, 0, 6, 242, 224
            , 17, 0, 1, 0, 0, 0, 70, 10, 2, 0, 6, 0, 16, 0, 0, 0, 0, 0, 62, 0, 0, 1, 83
            , 84, 65, 84, 148, 0, 0, 0, 10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, };
    }
}
