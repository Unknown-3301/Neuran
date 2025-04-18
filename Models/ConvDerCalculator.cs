using ComputeShaders;
using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Security;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Models
{
    internal class ConvDerCalculator : IDisposable
    {
        private CSDevice device;

        private ComputeShader preDerShader;
        private CSCBuffer<Int12> preInfo;

        private TensorGPUSummation summation;
        private ComputeShader filtersDerShader;
        private FInfo finfo;
        private CSCBuffer<FInfo> filtersInfo;

        public int BatchSize { get; private set; }
        public int GroupSize { get; private set; }

        public ConvDerCalculator(Tensor output, Tensor input, Tensor filtersAndBias, int filterSize, int stride, int padding, int groupSize = 8, int batchSize = 10)
        {
            device = output.device;

            GroupSize = groupSize;
            BatchSize = batchSize;

            preDerShader = device.CreateComputeShader(ConvolutionShaders.ConvolutionPreDer);
            preInfo = device.CreateBuffer(new Int12()
            {
                int1 = input.Dimensions[0],
                int2 = input.Dimensions[1],
                int3 = input.Dimensions[2],

                int4 = output.Dimensions[0],
                int5 = output.Dimensions[1],
                int6 = output.Dimensions[2],

                int7 = filterSize,
                int8 = stride,
                int9 = padding,

                int10 = filtersAndBias.TensorLength,
            }, Int12.Size);

            summation = new TensorGPUSummation(device, output.TensorLength, batchSize);
            filtersDerShader = device.CreateComputeShader(ConvolutionShaders.FiltersDerCopy);

            finfo = new FInfo()
            {
                InputWidth = input.Dimensions[0],
                InputHeight = input.Dimensions[1],
                InputDepth = input.Dimensions[2],

                OutputWidth = output.Dimensions[0],
                OutputHeight = output.Dimensions[1],
                OutputDepth = output.Dimensions[2],
                Padding = padding,

                FilterSize = filterSize,
                SumGroupSideSize = groupSize,
                Stride = stride,
                BatchSize = batchSize,

                GroupedWidth = (int)Math.Ceiling(output.Dimensions[0] / (float)groupSize),
                GroupedHeight = (int)Math.Ceiling(output.Dimensions[1] / (float)groupSize),

            };

            filtersInfo = device.CreateBuffer(finfo, FInfo.Size);
        }
        public void CalcPreLayer(Tensor beforeActivationDer, Tensor filtersAndBias, Tensor preLayerDer)
        {
            preLayerDer.SetUAV(0);
            beforeActivationDer.SetUAV(1);
            filtersAndBias.SetUAV(2);

            device.SetBuffer(preInfo, 0);

            device.SetComputeShader(preDerShader);

            device.Dispatch((int)Math.Ceiling(preLayerDer.Dimensions[0] / 8f), (int)Math.Ceiling(preLayerDer.Dimensions[1] / 8f), preLayerDer.Dimensions[2]);
            
            //float[] d1 = preLayerDer.GetData(); //DEBUG
            //float[] d2 = beforeActivationDer.GetData(); //DEBUG
            //float[] d3 = filtersAndBias.GetData(); //DEBUG
        }

        public void CalcFiltersDer(Tensor input, Tensor beforeActivationDer, Tensor filtersAndBias)
        {
            summation.Sum(batchSize: 1, result: filtersAndBias.Gradient, resultOffset: filtersAndBias.TensorLength - 1, inputs: beforeActivationDer);

            int fullRuns = (filtersAndBias.TensorLength - 1) / BatchSize; // -1 to not count bias
            int remain = (filtersAndBias.TensorLength - 1) % BatchSize;

            for (int i = 0; i < fullRuns; i++)
            {
                device.SetComputeShader(filtersDerShader);

                input.SetUAV(0);
                beforeActivationDer.SetUAV(1);
                summation.Input.SetUAV(2);

                device.SetBuffer(filtersInfo, 0);

                finfo.StartFilter = i * BatchSize;
                filtersInfo.UpdateBuffer(finfo);

                device.Dispatch((int)Math.Ceiling(finfo.GroupedWidth / 8f), (int)Math.Ceiling(finfo.GroupedHeight / 8f), beforeActivationDer.Dimensions[2]);

                //float[] d0 = summation.Input.GetData(); //DEBUG

                summation.RunFromInput(finfo.GroupedWidth * finfo.GroupedHeight, BatchSize, filtersAndBias.Gradient, i * BatchSize, GroupSize);

                //float[] d1 = input.GetData(); //DEBUG
                //float[] d2 = beforeActivationDer.GetData(); //DEBUG
                //float[] d3 = filtersAndBias.Gradient.GetData(); //DEBUG
            }

            if (remain != 0)
            {
                device.SetComputeShader(filtersDerShader);

                input.SetUAV(0);
                beforeActivationDer.SetUAV(1);
                summation.Input.SetUAV(2);

                device.SetBuffer(filtersInfo, 0);

                finfo.StartFilter = fullRuns * BatchSize;
                filtersInfo.UpdateBuffer(finfo);

                device.Dispatch((int)Math.Ceiling(finfo.GroupedWidth / 8f), (int)Math.Ceiling(finfo.GroupedHeight / 8f), beforeActivationDer.Dimensions[2]);

                summation.RunFromInput(finfo.GroupedWidth * finfo.GroupedHeight, remain, filtersAndBias.Gradient, fullRuns * BatchSize, GroupSize);
            }
        }
        public void Dispose()
        {
            preDerShader.Dispose();
            preInfo.Dispose();
            summation.Dispose();
            filtersDerShader.Dispose();
            filtersInfo.Dispose();
        }
    }

    internal struct FInfo
    {
        public int InputWidth;
        public int InputHeight;
        public int InputDepth;

        public int OutputWidth;
        public int OutputHeight;
        public int OutputDepth; //Number of filters too
        public int Padding;

        public int FilterSize;
        public int SumGroupSideSize; //The side length of a sum group (square) to sum together
        public int Stride;
        public int BatchSize; //Number of filter pixels in each batch

        public int StartFilter; //The index (from FilterElements) of the first filter pixel in the batch.
        public int StartZ; //The start depth for BeforeActivationDer (inclusive)
        public int EndZ; //The end depth for BeforeActivationDer (inclusive)

        //The dimensions of the output texture after grouping every SumGroupSideSize * SumGroupSideSize
        //pixels to one pixel (summing them up).
        public int GroupedWidth;
        public int GroupedHeight;

        public static int Size { get => sizeof(int) * 16; }
    }
}
