using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Activations
{
    internal static class LeakyReLUShaders
    {
        /*
    #pragma kernel CSMain

    RWStructuredBuffer<float> Input : register(u0);
    RWStructuredBuffer<float> Output : register(u1);

    cbuffer Info : register(b0)
    {
        int Length;

        int dummy1;
        int dummy2;
        int dummy3;
    }

    [numthreads(16, 1, 1)]
    void CSMain(uint3 id : SV_DispatchThreadID)
    {
        if (id.x >= Length)
        {
            return;
        }

        float f = Output[id.x];

        if (Input[id.x] < 0)
        {
            Output[id.x] = f * 0.01; 
        }
        else
        {
            Output[id.x] = f;
        }
    }
    */
        public static byte[] DerLeakyReLU1 = new byte[]
        {
68, 88, 66, 67, 174, 170, 107, 225, 120, 231, 42, 207, 73, 190, 192, 49, 199, 72, 2, 124, 1, 0, 0, 0
            , 212, 4, 0, 0, 5, 0, 0, 0, 52, 0, 0, 0, 200, 2, 0, 0, 216, 2, 0, 0, 232, 2, 0
            , 0, 56, 4, 0, 0, 82, 68, 69, 70, 140, 2, 0, 0, 3, 0, 0, 0, 176, 0, 0, 0, 3, 0
            , 0, 0, 60, 0, 0, 0, 0, 5, 83, 67, 0, 1, 0, 0, 100, 2, 0, 0, 82, 68, 49, 49, 60
            , 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 36, 0, 0, 0, 12, 0, 0, 0
            , 0, 0, 0, 0, 156, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0
            , 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 162, 0, 0, 0, 6, 0, 0, 0, 6, 0
            , 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 169
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 1, 0, 0, 0, 1, 0, 0, 0, 73, 110, 112, 117, 116, 0, 79, 117, 116, 112, 117, 116, 0, 73, 110
            , 102, 111, 0, 171, 171, 169, 0, 0, 0, 4, 0, 0, 0, 248, 0, 0, 0, 16, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 156, 0, 0, 0, 1, 0, 0, 0, 224, 1, 0, 0, 4, 0, 0, 0, 0
            , 0, 0, 0, 3, 0, 0, 0, 162, 0, 0, 0, 1, 0, 0, 0, 60, 2, 0, 0, 4, 0, 0, 0
            , 0, 0, 0, 0, 3, 0, 0, 0, 152, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0
            , 0, 164, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0
            , 0, 0, 200, 1, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 164, 1, 0, 0, 0
            , 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 207, 1, 0, 0
            , 8, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 164, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255
            , 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 214, 1, 0, 0, 12, 0, 0, 0, 4, 0
            , 0, 0, 0, 0, 0, 0, 164, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255
            , 255, 255, 255, 0, 0, 0, 0, 76, 101, 110, 103, 116, 104, 0, 105, 110, 116, 0, 171, 0, 0, 2, 0
            , 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 159, 1, 0, 0, 100, 117, 109, 109, 121, 49, 0, 100, 117, 109, 109, 121, 50, 0
            , 100, 117, 109, 109, 121, 51, 0, 171, 171, 171, 8, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2
            , 0, 0, 0, 24, 2, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255
            , 0, 0, 0, 0, 36, 69, 108, 101, 109, 101, 110, 116, 0, 102, 108, 111, 97, 116, 0, 171, 0, 0, 3
            , 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 17, 2, 0, 0, 8, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2
            , 0, 0, 0, 24, 2, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255
            , 0, 0, 0, 0, 77, 105, 99, 114, 111, 115, 111, 102, 116, 32, 40, 82, 41, 32, 72, 76, 83, 76, 32
            , 83, 104, 97, 100, 101, 114, 32, 67, 111, 109, 112, 105, 108, 101, 114, 32, 49, 48, 46, 49, 0, 73, 83
            , 71, 78, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 79, 83, 71, 78, 8, 0, 0, 0, 0
            , 0, 0, 0, 8, 0, 0, 0, 83, 72, 69, 88, 72, 1, 0, 0, 80, 0, 5, 0, 82, 0, 0, 0
            , 106, 8, 0, 1, 89, 0, 0, 4, 70, 142, 32, 0, 0, 0, 0, 0, 1, 0, 0, 0, 158, 0, 0
            , 4, 0, 224, 17, 0, 0, 0, 0, 0, 4, 0, 0, 0, 158, 0, 0, 4, 0, 224, 17, 0, 1, 0
            , 0, 0, 4, 0, 0, 0, 95, 0, 0, 2, 18, 0, 2, 0, 104, 0, 0, 2, 1, 0, 0, 0, 155
            , 0, 0, 4, 16, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 80, 0, 0, 7, 18, 0, 16, 0
            , 0, 0, 0, 0, 10, 0, 2, 0, 10, 128, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 0, 4
            , 3, 10, 0, 16, 0, 0, 0, 0, 0, 62, 0, 0, 1, 21, 0, 0, 1, 167, 0, 0, 138, 2, 35
            , 0, 128, 131, 153, 25, 0, 18, 0, 16, 0, 0, 0, 0, 0, 10, 0, 2, 0, 1, 64, 0, 0, 0
            , 0, 0, 0, 6, 224, 17, 0, 1, 0, 0, 0, 167, 0, 0, 138, 2, 35, 0, 128, 131, 153, 25, 0
            , 34, 0, 16, 0, 0, 0, 0, 0, 10, 0, 2, 0, 1, 64, 0, 0, 0, 0, 0, 0, 6, 224, 17
            , 0, 0, 0, 0, 0, 49, 0, 0, 7, 34, 0, 16, 0, 0, 0, 0, 0, 26, 0, 16, 0, 0, 0
            , 0, 0, 1, 64, 0, 0, 0, 0, 0, 0, 31, 0, 4, 3, 26, 0, 16, 0, 0, 0, 0, 0, 56
            , 0, 0, 7, 18, 0, 16, 0, 0, 0, 0, 0, 10, 0, 16, 0, 0, 0, 0, 0, 1, 64, 0, 0
            , 10, 215, 35, 60, 168, 0, 0, 8, 18, 224, 17, 0, 1, 0, 0, 0, 10, 0, 2, 0, 1, 64, 0
            , 0, 0, 0, 0, 0, 10, 0, 16, 0, 0, 0, 0, 0, 21, 0, 0, 1, 62, 0, 0, 1, 83, 84
            , 65, 84, 148, 0, 0, 0, 12, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2
            , 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, };
        /*
        #pragma kernel CSMain

        RWTexture2D<float> Input : register(u0);
        RWTexture2D<float> Output : register(u1);

        cbuffer Info : register(b0)
        {
            int Width;
            int Height;

            int dummy1;
            int dummy2;
        }

        [numthreads(8, 8, 1)]
        void CSMain(uint3 id : SV_DispatchThreadID)
        {
            if (id.x >= Width || id.y >= Height)
            {
                return;
            }

            float f = Output[id.xy];

            if (Input[id.xy] < 0)
            {
                Output[id.xy] = f * 0.01;
            }
            else
            {
                Output[id.xy] = f;
            }
        }
        */
        public static byte[] DerLeakyReLU2 = new byte[]
        {
68, 88, 66, 67, 35, 2, 103, 247, 7, 27, 190, 136, 122, 32, 204, 208, 184, 206, 72, 200, 1, 0, 0, 0
            , 36, 4, 0, 0, 5, 0, 0, 0, 52, 0, 0, 0, 20, 2, 0, 0, 36, 2, 0, 0, 52, 2, 0
            , 0, 136, 3, 0, 0, 82, 68, 69, 70, 216, 1, 0, 0, 1, 0, 0, 0, 176, 0, 0, 0, 3, 0
            , 0, 0, 60, 0, 0, 0, 0, 5, 83, 67, 0, 1, 0, 0, 173, 1, 0, 0, 82, 68, 49, 49, 60
            , 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 36, 0, 0, 0, 12, 0, 0, 0
            , 0, 0, 0, 0, 156, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 255, 255, 255
            , 255, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 162, 0, 0, 0, 4, 0, 0, 0, 5, 0
            , 0, 0, 4, 0, 0, 0, 255, 255, 255, 255, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 169
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 1, 0, 0, 0, 1, 0, 0, 0, 73, 110, 112, 117, 116, 0, 79, 117, 116, 112, 117, 116, 0, 73, 110
            , 102, 111, 0, 171, 171, 169, 0, 0, 0, 4, 0, 0, 0, 200, 0, 0, 0, 16, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 104, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 116
            , 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0
            , 152, 1, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 116, 1, 0, 0, 0, 0, 0
            , 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 159, 1, 0, 0, 8, 0
            , 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 116, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0
            , 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 166, 1, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0
            , 0, 0, 0, 0, 116, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255
            , 255, 0, 0, 0, 0, 87, 105, 100, 116, 104, 0, 105, 110, 116, 0, 171, 171, 0, 0, 2, 0, 1, 0
            , 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 110, 1, 0, 0, 72, 101, 105, 103, 104, 116, 0, 100, 117, 109, 109, 121, 49, 0, 100, 117
            , 109, 109, 121, 50, 0, 77, 105, 99, 114, 111, 115, 111, 102, 116, 32, 40, 82, 41, 32, 72, 76, 83, 76
            , 32, 83, 104, 97, 100, 101, 114, 32, 67, 111, 109, 112, 105, 108, 101, 114, 32, 49, 48, 46, 49, 0, 171
            , 171, 171, 73, 83, 71, 78, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 79, 83, 71, 78, 8
            , 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 83, 72, 69, 88, 76, 1, 0, 0, 80, 0, 5, 0
            , 83, 0, 0, 0, 106, 8, 0, 1, 89, 0, 0, 4, 70, 142, 32, 0, 0, 0, 0, 0, 1, 0, 0
            , 0, 156, 24, 0, 4, 0, 224, 17, 0, 0, 0, 0, 0, 85, 85, 0, 0, 156, 24, 0, 4, 0, 224
            , 17, 0, 1, 0, 0, 0, 85, 85, 0, 0, 95, 0, 0, 2, 50, 0, 2, 0, 104, 0, 0, 2, 1
            , 0, 0, 0, 155, 0, 0, 4, 8, 0, 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 80, 0, 0, 7
            , 50, 0, 16, 0, 0, 0, 0, 0, 70, 0, 2, 0, 70, 128, 32, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 60, 0, 0, 7, 18, 0, 16, 0, 0, 0, 0, 0, 26, 0, 16, 0, 0, 0, 0, 0, 10, 0
            , 16, 0, 0, 0, 0, 0, 31, 0, 4, 3, 10, 0, 16, 0, 0, 0, 0, 0, 62, 0, 0, 1, 21
            , 0, 0, 1, 163, 0, 0, 136, 194, 0, 0, 128, 67, 85, 21, 0, 18, 0, 16, 0, 0, 0, 0, 0
            , 70, 5, 2, 0, 70, 238, 17, 0, 1, 0, 0, 0, 163, 0, 0, 136, 194, 0, 0, 128, 67, 85, 21
            , 0, 34, 0, 16, 0, 0, 0, 0, 0, 70, 5, 2, 0, 22, 238, 17, 0, 0, 0, 0, 0, 49, 0
            , 0, 7, 34, 0, 16, 0, 0, 0, 0, 0, 26, 0, 16, 0, 0, 0, 0, 0, 1, 64, 0, 0, 0
            , 0, 0, 0, 31, 0, 4, 3, 26, 0, 16, 0, 0, 0, 0, 0, 56, 0, 0, 7, 18, 0, 16, 0
            , 0, 0, 0, 0, 10, 0, 16, 0, 0, 0, 0, 0, 1, 64, 0, 0, 10, 215, 35, 60, 164, 0, 0
            , 6, 242, 224, 17, 0, 1, 0, 0, 0, 70, 5, 2, 0, 6, 0, 16, 0, 0, 0, 0, 0, 21, 0
            , 0, 1, 62, 0, 0, 1, 83, 84, 65, 84, 148, 0, 0, 0, 13, 0, 0, 0, 1, 0, 0, 0, 0
            , 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0
            , 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0
            , 0, };
        /*
        #pragma kernel CSMain

        RWTexture3D<float> Input : register(u0);
        RWTexture3D<float> Output : register(u1);

        cbuffer Info : register(b0)
        {
            int Width;
            int Height;
            int Depth;

            int dummy1;
        }

        [numthreads(8, 8, 1)]
        void CSMain(uint3 id : SV_DispatchThreadID)
        {
            if (id.x >= Width || id.y >= Height || id.z >= Depth)
            {
                return;
            }

            float f = Output[id];

            if (Input[id] < 0)
            {
                Output[id] = f * 0.01;
            }
            else
            {
                Output[id] = f;
            }
        }
        */
        public static byte[] DerLeakyReLU3 = new byte[]
        {
68, 88, 66, 67, 210, 79, 123, 236, 176, 163, 29, 60, 59, 86, 150, 251, 215, 6, 251, 166, 1, 0, 0, 0
            , 60, 4, 0, 0, 5, 0, 0, 0, 52, 0, 0, 0, 16, 2, 0, 0, 32, 2, 0, 0, 48, 2, 0
            , 0, 160, 3, 0, 0, 82, 68, 69, 70, 212, 1, 0, 0, 1, 0, 0, 0, 176, 0, 0, 0, 3, 0
            , 0, 0, 60, 0, 0, 0, 0, 5, 83, 67, 0, 1, 0, 0, 172, 1, 0, 0, 82, 68, 49, 49, 60
            , 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 36, 0, 0, 0, 12, 0, 0, 0
            , 0, 0, 0, 0, 156, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 255, 255, 255
            , 255, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 162, 0, 0, 0, 4, 0, 0, 0, 5, 0
            , 0, 0, 8, 0, 0, 0, 255, 255, 255, 255, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 169
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 1, 0, 0, 0, 1, 0, 0, 0, 73, 110, 112, 117, 116, 0, 79, 117, 116, 112, 117, 116, 0, 73, 110
            , 102, 111, 0, 171, 171, 169, 0, 0, 0, 4, 0, 0, 0, 200, 0, 0, 0, 16, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 104, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 116
            , 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0
            , 152, 1, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 116, 1, 0, 0, 0, 0, 0
            , 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 159, 1, 0, 0, 8, 0
            , 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 116, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0
            , 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 165, 1, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0
            , 0, 0, 0, 0, 116, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255
            , 255, 0, 0, 0, 0, 87, 105, 100, 116, 104, 0, 105, 110, 116, 0, 171, 171, 0, 0, 2, 0, 1, 0
            , 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 110, 1, 0, 0, 72, 101, 105, 103, 104, 116, 0, 68, 101, 112, 116, 104, 0, 100, 117, 109
            , 109, 121, 49, 0, 77, 105, 99, 114, 111, 115, 111, 102, 116, 32, 40, 82, 41, 32, 72, 76, 83, 76, 32
            , 83, 104, 97, 100, 101, 114, 32, 67, 111, 109, 112, 105, 108, 101, 114, 32, 49, 48, 46, 49, 0, 73, 83
            , 71, 78, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 79, 83, 71, 78, 8, 0, 0, 0, 0
            , 0, 0, 0, 8, 0, 0, 0, 83, 72, 69, 88, 104, 1, 0, 0, 80, 0, 5, 0, 90, 0, 0, 0
            , 106, 8, 0, 1, 89, 0, 0, 4, 70, 142, 32, 0, 0, 0, 0, 0, 1, 0, 0, 0, 156, 40, 0
            , 4, 0, 224, 17, 0, 0, 0, 0, 0, 85, 85, 0, 0, 156, 40, 0, 4, 0, 224, 17, 0, 1, 0
            , 0, 0, 85, 85, 0, 0, 95, 0, 0, 2, 114, 0, 2, 0, 104, 0, 0, 2, 1, 0, 0, 0, 155
            , 0, 0, 4, 8, 0, 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 80, 0, 0, 7, 114, 0, 16, 0
            , 0, 0, 0, 0, 70, 2, 2, 0, 70, 130, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0
            , 7, 18, 0, 16, 0, 0, 0, 0, 0, 26, 0, 16, 0, 0, 0, 0, 0, 10, 0, 16, 0, 0, 0
            , 0, 0, 60, 0, 0, 7, 18, 0, 16, 0, 0, 0, 0, 0, 42, 0, 16, 0, 0, 0, 0, 0, 10
            , 0, 16, 0, 0, 0, 0, 0, 31, 0, 4, 3, 10, 0, 16, 0, 0, 0, 0, 0, 62, 0, 0, 1
            , 21, 0, 0, 1, 163, 0, 0, 136, 66, 1, 0, 128, 67, 85, 21, 0, 18, 0, 16, 0, 0, 0, 0
            , 0, 70, 10, 2, 0, 70, 238, 17, 0, 1, 0, 0, 0, 163, 0, 0, 136, 66, 1, 0, 128, 67, 85
            , 21, 0, 34, 0, 16, 0, 0, 0, 0, 0, 70, 10, 2, 0, 22, 238, 17, 0, 0, 0, 0, 0, 49
            , 0, 0, 7, 34, 0, 16, 0, 0, 0, 0, 0, 26, 0, 16, 0, 0, 0, 0, 0, 1, 64, 0, 0
            , 0, 0, 0, 0, 31, 0, 4, 3, 26, 0, 16, 0, 0, 0, 0, 0, 56, 0, 0, 7, 18, 0, 16
            , 0, 0, 0, 0, 0, 10, 0, 16, 0, 0, 0, 0, 0, 1, 64, 0, 0, 10, 215, 35, 60, 164, 0
            , 0, 6, 242, 224, 17, 0, 1, 0, 0, 0, 70, 10, 2, 0, 6, 0, 16, 0, 0, 0, 0, 0, 21
            , 0, 0, 1, 62, 0, 0, 1, 83, 84, 65, 84, 148, 0, 0, 0, 14, 0, 0, 0, 1, 0, 0, 0
            , 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0
            , 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
            , 0, 0, };
        /*
        #pragma kernel CSMain

        RWStructuredBuffer<float> Input : register(u0);
        RWStructuredBuffer<float> Output : register(u1);

        cbuffer Info : register(b0)
        {
            int Length;

            int dummy1;
            int dummy2;
            int dummy3;
        }

        [numthreads(16, 1, 1)]
        void CSMain(uint3 id : SV_DispatchThreadID)
        {
            if (id.x >= Length)
            {
                return;
            }

            float f = Input[id.x];

            Output[id.x] = max(f, 0.01 * f);
        }
        */
        public static byte[] LeakyReLU1 = new byte[]
    { 
    68, 88, 66, 67, 194, 181, 62, 34, 45, 31, 202, 18, 90, 150, 53, 188, 113, 243, 61, 64, 1, 0, 0, 0
                , 156, 4, 0, 0, 5, 0, 0, 0, 52, 0, 0, 0, 200, 2, 0, 0, 216, 2, 0, 0, 232, 2, 0
                , 0, 0, 4, 0, 0, 82, 68, 69, 70, 140, 2, 0, 0, 3, 0, 0, 0, 176, 0, 0, 0, 3, 0
                , 0, 0, 60, 0, 0, 0, 0, 5, 83, 67, 0, 1, 0, 0, 100, 2, 0, 0, 82, 68, 49, 49, 60
                , 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 36, 0, 0, 0, 12, 0, 0, 0
                , 0, 0, 0, 0, 156, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0
                , 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 162, 0, 0, 0, 6, 0, 0, 0, 6, 0
                , 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 169
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 1, 0, 0, 0, 1, 0, 0, 0, 73, 110, 112, 117, 116, 0, 79, 117, 116, 112, 117, 116, 0, 73, 110
                , 102, 111, 0, 171, 171, 169, 0, 0, 0, 4, 0, 0, 0, 248, 0, 0, 0, 16, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 156, 0, 0, 0, 1, 0, 0, 0, 224, 1, 0, 0, 4, 0, 0, 0, 0
                , 0, 0, 0, 3, 0, 0, 0, 162, 0, 0, 0, 1, 0, 0, 0, 60, 2, 0, 0, 4, 0, 0, 0
                , 0, 0, 0, 0, 3, 0, 0, 0, 152, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0
                , 0, 164, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0
                , 0, 0, 200, 1, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 164, 1, 0, 0, 0
                , 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 207, 1, 0, 0
                , 8, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 164, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255
                , 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 214, 1, 0, 0, 12, 0, 0, 0, 4, 0
                , 0, 0, 0, 0, 0, 0, 164, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255
                , 255, 255, 255, 0, 0, 0, 0, 76, 101, 110, 103, 116, 104, 0, 105, 110, 116, 0, 171, 0, 0, 2, 0
                , 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 159, 1, 0, 0, 100, 117, 109, 109, 121, 49, 0, 100, 117, 109, 109, 121, 50, 0
                , 100, 117, 109, 109, 121, 51, 0, 171, 171, 171, 8, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2
                , 0, 0, 0, 24, 2, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255
                , 0, 0, 0, 0, 36, 69, 108, 101, 109, 101, 110, 116, 0, 102, 108, 111, 97, 116, 0, 171, 0, 0, 3
                , 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 17, 2, 0, 0, 8, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2
                , 0, 0, 0, 24, 2, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255
                , 0, 0, 0, 0, 77, 105, 99, 114, 111, 115, 111, 102, 116, 32, 40, 82, 41, 32, 72, 76, 83, 76, 32
                , 83, 104, 97, 100, 101, 114, 32, 67, 111, 109, 112, 105, 108, 101, 114, 32, 49, 48, 46, 49, 0, 73, 83
                , 71, 78, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 79, 83, 71, 78, 8, 0, 0, 0, 0
                , 0, 0, 0, 8, 0, 0, 0, 83, 72, 69, 88, 16, 1, 0, 0, 80, 0, 5, 0, 68, 0, 0, 0
                , 106, 8, 0, 1, 89, 0, 0, 4, 70, 142, 32, 0, 0, 0, 0, 0, 1, 0, 0, 0, 158, 0, 0
                , 4, 0, 224, 17, 0, 0, 0, 0, 0, 4, 0, 0, 0, 158, 0, 0, 4, 0, 224, 17, 0, 1, 0
                , 0, 0, 4, 0, 0, 0, 95, 0, 0, 2, 18, 0, 2, 0, 104, 0, 0, 2, 1, 0, 0, 0, 155
                , 0, 0, 4, 16, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 80, 0, 0, 7, 18, 0, 16, 0
                , 0, 0, 0, 0, 10, 0, 2, 0, 10, 128, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 0, 4
                , 3, 10, 0, 16, 0, 0, 0, 0, 0, 62, 0, 0, 1, 21, 0, 0, 1, 167, 0, 0, 138, 2, 35
                , 0, 128, 131, 153, 25, 0, 18, 0, 16, 0, 0, 0, 0, 0, 10, 0, 2, 0, 1, 64, 0, 0, 0
                , 0, 0, 0, 6, 224, 17, 0, 0, 0, 0, 0, 56, 0, 0, 7, 34, 0, 16, 0, 0, 0, 0, 0
                , 10, 0, 16, 0, 0, 0, 0, 0, 1, 64, 0, 0, 10, 215, 35, 60, 52, 0, 0, 7, 18, 0, 16
                , 0, 0, 0, 0, 0, 26, 0, 16, 0, 0, 0, 0, 0, 10, 0, 16, 0, 0, 0, 0, 0, 168, 0
                , 0, 8, 18, 224, 17, 0, 1, 0, 0, 0, 10, 0, 2, 0, 1, 64, 0, 0, 0, 0, 0, 0, 10
                , 0, 16, 0, 0, 0, 0, 0, 62, 0, 0, 1, 83, 84, 65, 84, 148, 0, 0, 0, 9, 0, 0, 0
                , 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0
                , 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 1, 0, 0, 0, };
    /*
    #pragma kernel CSMain

    RWTexture2D<float> Input : register(u0);
    RWTexture2D<float> Output : register(u1);

    cbuffer Info : register(b0)
    {
        int Width;
        int Height;
    
        int dummy1;
        int dummy2;
    }

    [numthreads(8, 8, 1)]
    void CSMain(uint3 id : SV_DispatchThreadID)
    {
        if (id.x >= Width || id.y >= Height)
        {
            return;
        }
    
        float f = Input[id.xy];
    
        Output[id.xy] = max(f, 0.01 * f);
    }
    */
    public static byte[] LeakyReLU2 = new byte[]
    { 
    68, 88, 66, 67, 96, 153, 163, 234, 98, 226, 173, 176, 205, 96, 254, 210, 93, 218, 163, 224, 1, 0, 0, 0
                , 244, 3, 0, 0, 5, 0, 0, 0, 52, 0, 0, 0, 20, 2, 0, 0, 36, 2, 0, 0, 52, 2, 0
                , 0, 88, 3, 0, 0, 82, 68, 69, 70, 216, 1, 0, 0, 1, 0, 0, 0, 176, 0, 0, 0, 3, 0
                , 0, 0, 60, 0, 0, 0, 0, 5, 83, 67, 0, 1, 0, 0, 173, 1, 0, 0, 82, 68, 49, 49, 60
                , 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 36, 0, 0, 0, 12, 0, 0, 0
                , 0, 0, 0, 0, 156, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 4, 0, 0, 0, 255, 255, 255
                , 255, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 162, 0, 0, 0, 4, 0, 0, 0, 5, 0
                , 0, 0, 4, 0, 0, 0, 255, 255, 255, 255, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 169
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 1, 0, 0, 0, 1, 0, 0, 0, 73, 110, 112, 117, 116, 0, 79, 117, 116, 112, 117, 116, 0, 73, 110
                , 102, 111, 0, 171, 171, 169, 0, 0, 0, 4, 0, 0, 0, 200, 0, 0, 0, 16, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 104, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 116
                , 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0
                , 152, 1, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 116, 1, 0, 0, 0, 0, 0
                , 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 159, 1, 0, 0, 8, 0
                , 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 116, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0
                , 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 166, 1, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0
                , 0, 0, 0, 0, 116, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255
                , 255, 0, 0, 0, 0, 87, 105, 100, 116, 104, 0, 105, 110, 116, 0, 171, 171, 0, 0, 2, 0, 1, 0
                , 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 110, 1, 0, 0, 72, 101, 105, 103, 104, 116, 0, 100, 117, 109, 109, 121, 49, 0, 100, 117
                , 109, 109, 121, 50, 0, 77, 105, 99, 114, 111, 115, 111, 102, 116, 32, 40, 82, 41, 32, 72, 76, 83, 76
                , 32, 83, 104, 97, 100, 101, 114, 32, 67, 111, 109, 112, 105, 108, 101, 114, 32, 49, 48, 46, 49, 0, 171
                , 171, 171, 73, 83, 71, 78, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 79, 83, 71, 78, 8
                , 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 83, 72, 69, 88, 28, 1, 0, 0, 80, 0, 5, 0
                , 71, 0, 0, 0, 106, 8, 0, 1, 89, 0, 0, 4, 70, 142, 32, 0, 0, 0, 0, 0, 1, 0, 0
                , 0, 156, 24, 0, 4, 0, 224, 17, 0, 0, 0, 0, 0, 85, 85, 0, 0, 156, 24, 0, 4, 0, 224
                , 17, 0, 1, 0, 0, 0, 85, 85, 0, 0, 95, 0, 0, 2, 50, 0, 2, 0, 104, 0, 0, 2, 1
                , 0, 0, 0, 155, 0, 0, 4, 8, 0, 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 80, 0, 0, 7
                , 50, 0, 16, 0, 0, 0, 0, 0, 70, 0, 2, 0, 70, 128, 32, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 60, 0, 0, 7, 18, 0, 16, 0, 0, 0, 0, 0, 26, 0, 16, 0, 0, 0, 0, 0, 10, 0
                , 16, 0, 0, 0, 0, 0, 31, 0, 4, 3, 10, 0, 16, 0, 0, 0, 0, 0, 62, 0, 0, 1, 21
                , 0, 0, 1, 163, 0, 0, 136, 194, 0, 0, 128, 67, 85, 21, 0, 18, 0, 16, 0, 0, 0, 0, 0
                , 70, 5, 2, 0, 70, 238, 17, 0, 0, 0, 0, 0, 56, 0, 0, 7, 34, 0, 16, 0, 0, 0, 0
                , 0, 10, 0, 16, 0, 0, 0, 0, 0, 1, 64, 0, 0, 10, 215, 35, 60, 52, 0, 0, 7, 18, 0
                , 16, 0, 0, 0, 0, 0, 26, 0, 16, 0, 0, 0, 0, 0, 10, 0, 16, 0, 0, 0, 0, 0, 164
                , 0, 0, 6, 242, 224, 17, 0, 1, 0, 0, 0, 70, 5, 2, 0, 6, 0, 16, 0, 0, 0, 0, 0
                , 62, 0, 0, 1, 83, 84, 65, 84, 148, 0, 0, 0, 10, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0
                , 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, };
    /*
    #pragma kernel CSMain

    RWTexture3D<float> Input : register(u0);
    RWTexture3D<float> Output : register(u1);

    cbuffer Info : register(b0)
    {
        int Width;
        int Height;
        int Depth;
    
        int dummy1;
    }

    [numthreads(8, 8, 1)]
    void CSMain(uint3 id : SV_DispatchThreadID)
    {
        if (id.x >= Width || id.y >= Height || id.z >= Depth)
        {
            return;
        }
    
        float f = Input[id];
    
        Output[id] = max(f, 0.01 * f);
    }
    */
    public static byte[] LeakyReLU3 = new byte[]
    { 
    68, 88, 66, 67, 144, 60, 106, 54, 146, 72, 26, 54, 177, 233, 184, 166, 0, 137, 158, 122, 1, 0, 0, 0
                , 12, 4, 0, 0, 5, 0, 0, 0, 52, 0, 0, 0, 16, 2, 0, 0, 32, 2, 0, 0, 48, 2, 0
                , 0, 112, 3, 0, 0, 82, 68, 69, 70, 212, 1, 0, 0, 1, 0, 0, 0, 176, 0, 0, 0, 3, 0
                , 0, 0, 60, 0, 0, 0, 0, 5, 83, 67, 0, 1, 0, 0, 172, 1, 0, 0, 82, 68, 49, 49, 60
                , 0, 0, 0, 24, 0, 0, 0, 32, 0, 0, 0, 40, 0, 0, 0, 36, 0, 0, 0, 12, 0, 0, 0
                , 0, 0, 0, 0, 156, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 255, 255, 255
                , 255, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 162, 0, 0, 0, 4, 0, 0, 0, 5, 0
                , 0, 0, 8, 0, 0, 0, 255, 255, 255, 255, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 169
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 1, 0, 0, 0, 1, 0, 0, 0, 73, 110, 112, 117, 116, 0, 79, 117, 116, 112, 117, 116, 0, 73, 110
                , 102, 111, 0, 171, 171, 169, 0, 0, 0, 4, 0, 0, 0, 200, 0, 0, 0, 16, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 104, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 116
                , 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0
                , 152, 1, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 116, 1, 0, 0, 0, 0, 0
                , 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 159, 1, 0, 0, 8, 0
                , 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 116, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0
                , 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 165, 1, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0
                , 0, 0, 0, 0, 116, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255
                , 255, 0, 0, 0, 0, 87, 105, 100, 116, 104, 0, 105, 110, 116, 0, 171, 171, 0, 0, 2, 0, 1, 0
                , 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 110, 1, 0, 0, 72, 101, 105, 103, 104, 116, 0, 68, 101, 112, 116, 104, 0, 100, 117, 109
                , 109, 121, 49, 0, 77, 105, 99, 114, 111, 115, 111, 102, 116, 32, 40, 82, 41, 32, 72, 76, 83, 76, 32
                , 83, 104, 97, 100, 101, 114, 32, 67, 111, 109, 112, 105, 108, 101, 114, 32, 49, 48, 46, 49, 0, 73, 83
                , 71, 78, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 79, 83, 71, 78, 8, 0, 0, 0, 0
                , 0, 0, 0, 8, 0, 0, 0, 83, 72, 69, 88, 56, 1, 0, 0, 80, 0, 5, 0, 78, 0, 0, 0
                , 106, 8, 0, 1, 89, 0, 0, 4, 70, 142, 32, 0, 0, 0, 0, 0, 1, 0, 0, 0, 156, 40, 0
                , 4, 0, 224, 17, 0, 0, 0, 0, 0, 85, 85, 0, 0, 156, 40, 0, 4, 0, 224, 17, 0, 1, 0
                , 0, 0, 85, 85, 0, 0, 95, 0, 0, 2, 114, 0, 2, 0, 104, 0, 0, 2, 1, 0, 0, 0, 155
                , 0, 0, 4, 8, 0, 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 80, 0, 0, 7, 114, 0, 16, 0
                , 0, 0, 0, 0, 70, 2, 2, 0, 70, 130, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 0, 0
                , 7, 18, 0, 16, 0, 0, 0, 0, 0, 26, 0, 16, 0, 0, 0, 0, 0, 10, 0, 16, 0, 0, 0
                , 0, 0, 60, 0, 0, 7, 18, 0, 16, 0, 0, 0, 0, 0, 42, 0, 16, 0, 0, 0, 0, 0, 10
                , 0, 16, 0, 0, 0, 0, 0, 31, 0, 4, 3, 10, 0, 16, 0, 0, 0, 0, 0, 62, 0, 0, 1
                , 21, 0, 0, 1, 163, 0, 0, 136, 66, 1, 0, 128, 67, 85, 21, 0, 18, 0, 16, 0, 0, 0, 0
                , 0, 70, 10, 2, 0, 70, 238, 17, 0, 0, 0, 0, 0, 56, 0, 0, 7, 34, 0, 16, 0, 0, 0
                , 0, 0, 10, 0, 16, 0, 0, 0, 0, 0, 1, 64, 0, 0, 10, 215, 35, 60, 52, 0, 0, 7, 18
                , 0, 16, 0, 0, 0, 0, 0, 26, 0, 16, 0, 0, 0, 0, 0, 10, 0, 16, 0, 0, 0, 0, 0
                , 164, 0, 0, 6, 242, 224, 17, 0, 1, 0, 0, 0, 70, 10, 2, 0, 6, 0, 16, 0, 0, 0, 0
                , 0, 62, 0, 0, 1, 83, 84, 65, 84, 148, 0, 0, 0, 11, 0, 0, 0, 1, 0, 0, 0, 0, 0
                , 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 1
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0
                , };
    }
}
