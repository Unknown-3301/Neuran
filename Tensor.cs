using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ComputeShaders;

namespace Neuran
{
    /// <summary>
    /// A class that stores data in a n-dimensional space.
    /// <br>Note that if it is a gpu tensor, it is able to store up to 3d data only:</br>
    /// <br>1d tensors are stored as StructuredBuffer in hlsl shaders.</br>
    /// <br>2d tensors are stored as Texture2D (float) in hlsl shaders.</br>
    /// <br>1d tensors are stored as Texture2DArray (float) in hlsl shaders.</br>
    /// </summary>
    public class Tensor : IDisposable
    {
        /// <summary>
        /// What processor does the tensor operate/store data on.
        /// </summary>
        public ProcessorType ProcessorType { get; }

        /// <summary>
        /// If true, when copying to a 2D/3D GPU Tensor, instead of using <see cref="ShaderResource{T}.UpdateSubresource(IntPtr)"/> it will dispose the currect shader resource and create a new one with the data to copy to (usually faster).
        /// </summary>
        public bool CreateNewTexture { get; set; } = true;

        /// <summary>
        /// The dimensions of the tensor.
        /// </summary>
        public int[] Dimensions { get; }

        /// <summary>
        /// The total number of elements in the tensor.
        /// </summary>
        public int TensorLength { get; }

        /// <summary>
        /// A tensor to store tensor gradient.
        /// </summary>
        public Tensor Gradient { get; private set; }


        //CPU stuff
        float[] data;
        int[] dimensionsProd;

        //GPU stuff
        internal CSDevice device;
        CSStructuredBuffer<float> fData;
        CSTexture2D sData;
        CSTexture3D tData;

        public int GetDEBUGSize()
        {
            if (ProcessorType == ProcessorType.CPU)
                return 0;

            switch (Dimensions.Length)
            {
                case 1: return fData.Length * sizeof(float);
                case 2: return sData.Width * sData.Height * sizeof(float);
                case 3: return tData.Width * tData.Height * tData.Depth * sizeof(float);
            }

            return 0;
        }

        /// <summary>
        /// Creates a new tensor.
        /// </summary>
        /// <param name="device">Direct3D11 device. If this is a CPU tensor pass null.</param>
        /// <param name="dimensions">The dimensions of the tensor. Note for GPU tensors its length should not exceed 3.</param>
        public Tensor(CSDevice device, params int[] dimensions)
        {
            ProcessorType = device == null ? ProcessorType.CPU : ProcessorType.GPU;
            this.device = device;

            Dimensions = new int[dimensions.Length];

            TensorLength = device == null ? InitCPU(dimensions) : InitGPU(dimensions);
        }

        /// <summary>
        /// Creates a new 1D GPU tensor storing data from <paramref name="order1_data"/>.
        /// </summary>
        /// <param name="device">Direct3D11 device.</param>
        /// <param name="order1_data">The data.</param>
        public Tensor(CSDevice device, float[] order1_data)
        {
            ProcessorType = ProcessorType.GPU;
            this.device = device;

            Dimensions = new int[] { order1_data.Length };

            fData = device.CreateStructuredBuffer(order1_data, sizeof(float));

            dimensionsProd = new int[1];
            dimensionsProd[0] = 1;

            TensorLength = order1_data.Length;
        }

        private int InitCPU(int[] dimensions)
        {
            dimensionsProd = new int[dimensions.Length];
            int prod = 1;
            for (int i = 0; i < dimensions.Length; i++)
            {
                Dimensions[i] = dimensions[i];
                dimensionsProd[i] = prod;
                prod *= dimensions[i];
            }

            data = new float[prod];

            return prod;
        }
        private int InitGPU(int[] dimensions)
        {
            dimensionsProd = new int[dimensions.Length];
            int prod = 1;
            for (int i = 0; i < dimensions.Length; i++)
            {
                Dimensions[i] = dimensions[i];
                dimensionsProd[i] = prod;
                prod *= dimensions[i];
            }

            switch (Dimensions.Length)
            {
                case 1: fData = device.CreateStructuredBuffer(new float[dimensions[0]], sizeof(float)); break;
                case 2: sData = device.CreateTexture2D(dimensions[0], dimensions[1], TextureFormat.R32_Float); break;
                case 3: tData = device.CreateTexture3D(dimensions[0], dimensions[1], dimensions[2], TextureFormat.R32_Float); break;
            }

            return prod;
        }

        /// <summary>
        /// Returns the data stored in the Tensor.
        /// <br>Note: the returned array is the exact reference data stored on the tensor if it was a cpu tensor.
        /// On the gpu tensor however, it is a copy.</br>
        /// </summary>
        /// <returns></returns>
        public float[] GetData()
        {
            if (ProcessorType == ProcessorType.CPU)
                return data;

            switch (Dimensions.Length)
            {
                case 1: return GetData1();
                case 2: return GetData2();
                case 3: return GetData3();
                default: return null;
            }
        }

        #region CPU Functions
        private int FlatIndex(params int[] index)
        {
            int fIndex = 0;

            for (int i = 0; i < Math.Min(index.Length, Dimensions.Length); i++)
            {
                fIndex += index[i] * dimensionsProd[i];
            }

            return fIndex;
        }
        /// <summary>
        /// Access the data stored. CPU ONLY
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public float this[params int[] index] { get => data[FlatIndex(index)]; set => data[FlatIndex(index)] = value; }
        /// <summary>
        /// Access the data stored as a flat array. CPU ONLY
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public float this[int index] { get => data[index]; set => data[index] = value; }
        #endregion

        #region GPU Functions
        /// <summary>
        /// Gives access to the raw data. GPU ONLY
        /// </summary>
        /// <param name="accessMode"></param>
        /// <param name="accessAction"></param>
        public void AccessRawData(CPUAccessMode accessMode, Action<TextureDataBox> accessAction)
        {
            switch (Dimensions.Length)
            {
                case 1: fData.AccessRawData(accessAction, accessMode); break;
                case 2: sData.AccessRawData(accessAction, accessMode); break;
                case 3: tData.AccessRawData(accessAction, accessMode); break;
            }
        }

        /// <summary>
        /// Returns the element data pointer.
        /// </summary>
        /// <param name="box"></param>
        /// <param name="index">Element index (in a flat array).</param>
        /// <returns></returns>
        public IntPtr ElementPosition(TextureDataBox box, int index)
        {
            switch (Dimensions.Length)
            {
                case 1: return IntPtr.Add(box.DataPointer, index * sizeof(float));
                case 2: return IntPtr.Add(box.DataPointer, index * sizeof(float) + (box.RowPitch - sData.Width * sizeof(float)) * (index / sData.Width));
                case 3: return IntPtr.Add(box.DataPointer, index * sizeof(float) + (box.RowPitch - tData.Width * sizeof(float)) * (index / tData.Width) + (box.SlicePitch - box.RowPitch * tData.Height) * (index / (tData.Width * tData.Height)));
            }

            return box.DataPointer;
        }
        //public IntPtr ElementPositionOld(TextureDataBox box, int index)
        //{
        //    switch (Dimensions.Length)
        //    {
        //        case 1: return IntPtr.Add(box.DataPointer, index * sizeof(float));
        //        case 2: return IntPtr.Add(box.DataPointer, index * box.RowPitch + index % sData.Width * sizeof(float));
        //        case 3: return IntPtr.Add(box.DataPointer, index * box.SlicePitch + index / tData.Width % tData.Height * box.RowPitch % tData.Width + index * sizeof(float));
        //    }

        //    return box.DataPointer;
        //}
        private float[] GetData1()
        {
            fData.EnableCPU_Raw_ReadWrite();
            float[] data = new float[fData.Length];
            fData.GetData(ref data);
            return data;
        }
        private unsafe float[] GetData2()
        {
            sData.EnableCPU_Raw_ReadWrite();
            float[] data = new float[sData.Width * sData.Height];

            fixed (float* f = &data[0])
            {
                IntPtr dst = (IntPtr)f;

                sData.AccessRawData(box =>
                {
                    IntPtr scr = box.DataPointer;

                    for (int i = 0; i < sData.Height; i++)
                    {
                        ComputeShaders.Utilities.CopyMemory(dst, scr, (uint)(sizeof(float) * sData.Width));

                        scr = IntPtr.Add(scr, box.RowPitch);
                        dst = IntPtr.Add(dst, sizeof(float) * sData.Width);
                    }

                }, CPUAccessMode.Read);
            }

            return data;
        }
        private unsafe float[] GetData3()
        {
            tData.EnableCPU_Raw_ReadWrite();
            float[] data = new float[tData.Width * tData.Height * tData.Depth];

            fixed (float* f = &data[0])
            {
                IntPtr dst = (IntPtr)f;

                tData.AccessRawData(box =>
                {
                    IntPtr o_scr = box.DataPointer;
                    IntPtr scr = box.DataPointer;
                    int sliceDiff = box.SlicePitch - box.RowPitch * tData.Height;

                    for (int z = 0; z < tData.Depth; z++)
                    {
                        for (int i = 0; i < tData.Height; i++)
                        {
                            ComputeShaders.Utilities.CopyMemory(dst, scr, (uint)(sizeof(float) * tData.Width));

                            scr = IntPtr.Add(scr, box.RowPitch);
                            dst = IntPtr.Add(dst, sizeof(float) * tData.Width);
                        }

                        scr = IntPtr.Add(scr, sliceDiff);
                    }

                }, CPUAccessMode.Read);
            }

            return data;
        }
        /// <summary>
        /// Sets the resource uav (Unordered Access View).
        /// </summary>
        /// <param name="uav_index"></param>
        public void SetUAV(int uav_index)
        {
            switch (Dimensions.Length)
            {
                case 1: device.SetUnorderedAccessView(fData, uav_index); break;
                case 2: device.SetUnorderedAccessView(sData, uav_index); break;
                case 3: device.SetUnorderedAccessView(tData, uav_index); break;
            }
        }

        private unsafe void CTCCopy(Tensor destination)
        {
            fixed (float* fs = &data[0])
            {
                fixed (float* fd = &destination.data[0])
                {
                    IntPtr dst = (IntPtr)fd;
                    IntPtr scr = (IntPtr)fs;

                    ComputeShaders.Utilities.CopyMemory(dst, scr, (uint)(Math.Min(data.Length, destination.data.Length) * sizeof(float)));
                }
            }
        }
        private unsafe void CTCCopy(Tensor destination, TensorBox srcBox, int dstX, int dstY = 0, int dstZ = 0)
        {
            int flat_src_start = FlatIndex(srcBox.StartX, srcBox.StartY, srcBox.StartZ);
            int flat_dst_start = destination.FlatIndex(dstX, dstY, dstZ);

            fixed (float* fs = &data[0])
            {
                fixed (float* fd = &destination.data[0])
                {
                    IntPtr dst = IntPtr.Add((IntPtr)fd, flat_dst_start * sizeof(float));
                    IntPtr src = IntPtr.Add((IntPtr)fs, flat_src_start * sizeof(float));

                    for (int z = 0; z < srcBox.CountZ; z++)
                    {
                        int src_diff = 0;
                        int dst_diff = 0;

                        if (Dimensions.Length >= 2)
                        {
                            src_diff = Dimensions[0] * (Dimensions[1] - srcBox.CountY) * sizeof(float); //the number of bytes between the end of a slice copy and the next start copy position
                            dst_diff = destination.Dimensions[0] * (destination.Dimensions[1] - srcBox.CountY) * sizeof(float); //the number of bytes between the end of a slice copy and the next start copy position
                        }


                        for (int y = 0; y < srcBox.CountY; y++)
                        {
                            ComputeShaders.Utilities.CopyMemory(dst, src, (uint)srcBox.CountX * sizeof(float));

                            src = IntPtr.Add(src, Dimensions[0] * sizeof(float));
                            dst = IntPtr.Add(dst, destination.Dimensions[0] * sizeof(float));
                        }

                        src = IntPtr.Add(src, src_diff);
                        dst = IntPtr.Add(dst, dst_diff);
                        //if (srcBox.CountX == Dimensions[0])
                        //{
                        //    ComputeShaders.Utilities.CopyMemory(dst, src, (uint)(srcBox.CountX * srcBox.CountY) * sizeof(float));

                        //    src = IntPtr.Add(src, Dimensions[0] * (Dimensions.Length >= 2 ? Dimensions[1] : 1) * sizeof(float));
                        //    dst = IntPtr.Add(dst, destination.Dimensions[0] * (destination.Dimensions.Length >= 2 ? destination.Dimensions[1] : 1) * sizeof(float));
                        //}
                        //else
                        //{
                        //    int diff = 0;

                        //    if (Dimensions.Length >= 2)
                        //    {
                        //        diff = Dimensions[0] * (Dimensions[1] - srcBox.CountY) * sizeof(float); //the number of bytes between the end of a slice copy and the next start copy position
                        //    }


                        //    for (int y = 0; y < srcBox.CountY; y++)
                        //    {
                        //        ComputeShaders.Utilities.CopyMemory(dst, src, (uint)srcBox.CountX * sizeof(float));

                        //        src = IntPtr.Add(src, Dimensions[0] * sizeof(float));
                        //        dst = IntPtr.Add(dst, destination.Dimensions[0] * sizeof(float));
                        //    }

                        //    src = IntPtr.Add(src, diff); 
                        //    dst = IntPtr.Add(dst, diff);
                        //}
                    }
                }
            }
        }
        private unsafe void CTGCopy(Tensor destination)
        {
            float[] d = GetData();

            fixed (float* fs = d)
            {
                IntPtr scr = (IntPtr)fs;

                switch (destination.Dimensions.Length)
                {
                    case 1: destination.fData.UpdateSubresource(scr); break;
                    case 2:
                        if (CreateNewTexture)
                        {
                            int width = destination.sData.Width, height = destination.sData.Height;

                            destination.sData.Dispose();
                            destination.sData = destination.device.CreateTexture2D(width, height, TextureFormat.R32_Float, scr);
                        }
                        else
                            destination.sData.UpdateSubresource(scr);
                        break;
                    case 3:
                        if (CreateNewTexture)
                        {
                            int width = destination.tData.Width, height = destination.tData.Height, depth = destination.tData.Depth;

                            destination.tData.Dispose();
                            destination.tData = destination.device.CreateTexture3D(width, height, depth, TextureFormat.R32_Float, scr);
                        }
                        else
                            destination.sData.UpdateSubresource(scr);
                        break;
                }
            }
        }
        private unsafe void CTGCopy(Tensor destination, TensorBox srcBox, int dstX, int dstY = 0, int dstZ = 0)
        {
            float[] d = GetData();

            fixed (float* fs = d)
            {
                IntPtr scr = (IntPtr)fs;

                switch (destination.Dimensions.Length)
                {
                    case 1: destination.fData.EnableCPU_Raw_ReadWrite(); break;
                    case 2: destination.sData.EnableCPU_Raw_ReadWrite(); break;
                    case 3: destination.tData.EnableCPU_Raw_ReadWrite(); break;
                }

                int flat_src_start = FlatIndex(srcBox.StartX, srcBox.StartY, srcBox.StartZ);
                int src_diff = Dimensions[0] * ((Dimensions.Length >= 2 ? Dimensions[1] : 0) - srcBox.CountY) * sizeof(float);
                IntPtr src = IntPtr.Add((IntPtr)fs, flat_src_start * sizeof(float));

                destination.AccessRawData(CPUAccessMode.Write, dstbox =>
                {
                    int flat_dst_start = destination.FlatIndex(dstX, dstY, dstZ);
                    int dst_diff = dstbox.SlicePitch - dstbox.RowPitch * srcBox.CountY;
                    IntPtr dst = IntPtr.Add(dstbox.DataPointer, dstX * sizeof(float) + dstbox.RowPitch * dstY + dstbox.SlicePitch * dstZ);

                    for (int z = 0; z < srcBox.CountZ; z++)
                    {
                        for (int y = 0; y < srcBox.CountY; y++)
                        {
                            ComputeShaders.Utilities.CopyMemory(dst, src, (uint)srcBox.CountX * sizeof(float));

                            src = IntPtr.Add(src, Dimensions[0] * sizeof(float));
                            dst = IntPtr.Add(dst, dstbox.RowPitch);
                        }

                        dst = IntPtr.Add(dst, dst_diff);
                        src = IntPtr.Add(src, src_diff);
                    }
                });
            }
        }
        private unsafe void GTCCopy(Tensor destination)
        {
            float[] floats = GetData();

            fixed (float* fs = &floats[0])
            {
                fixed (float* fd = &destination.data[0])
                {
                    IntPtr dst = (IntPtr)fd;
                    IntPtr scr = (IntPtr)fs;

                    ComputeShaders.Utilities.CopyMemory(dst, scr, (uint)(Math.Min(floats.Length, destination.data.Length) * sizeof(float)));
                }
            }
        }
        private unsafe void GTCCopy(Tensor destination, TensorBox srcBox, int dstX, int dstY = 0, int dstZ = 0)
        {
            float[] floats = GetData();

            int flat_src_start = FlatIndex(srcBox.StartX, srcBox.StartY, srcBox.StartZ);
            int flat_dst_start = destination.FlatIndex(dstX, dstY, dstZ);

            fixed (float* fs = &floats[0])
            {
                fixed (float* fd = &destination.data[0])
                {
                    IntPtr dst = IntPtr.Add((IntPtr)fd, flat_dst_start * sizeof(float));
                    IntPtr src = IntPtr.Add((IntPtr)fs, flat_src_start * sizeof(float));

                    for (int z = 0; z < srcBox.CountZ; z++)
                    {
                        int src_diff = 0;
                        int dst_diff = 0;

                        if (Dimensions.Length >= 2)
                        {
                            src_diff = Dimensions[0] * (Dimensions[1] - srcBox.CountY) * sizeof(float); //the number of bytes between the end of a slice copy and the next start copy position
                            dst_diff = destination.Dimensions[0] * (destination.Dimensions[1] - srcBox.CountY) * sizeof(float); //the number of bytes between the end of a slice copy and the next start copy position
                        }


                        for (int y = 0; y < srcBox.CountY; y++)
                        {
                            ComputeShaders.Utilities.CopyMemory(dst, src, (uint)srcBox.CountX * sizeof(float));

                            src = IntPtr.Add(src, Dimensions[0] * sizeof(float));
                            dst = IntPtr.Add(dst, destination.Dimensions[0] * sizeof(float));
                        }

                        src = IntPtr.Add(src, src_diff);
                        dst = IntPtr.Add(dst, dst_diff);
                        //if (srcBox.CountX == Dimensions[0])
                        //{
                        //    ComputeShaders.Utilities.CopyMemory(dst, src, (uint)(Dimensions[0] * srcBox.CountY) * sizeof(float));

                        //    src = IntPtr.Add(src, Dimensions[0] * (Dimensions.Length >= 2 ? Dimensions[1] : 0) * sizeof(float));
                        //    dst = IntPtr.Add(dst, destination.Dimensions[0] * (destination.Dimensions.Length >= 2 ? destination.Dimensions[1] : 0) * sizeof(float));
                        //}
                        //else
                        //{
                        //    int diff = 0;

                        //    if (Dimensions.Length >= 2)
                        //    {
                        //        diff = Dimensions[0] * (Dimensions[1] - srcBox.CountY) * sizeof(float);
                        //    }


                        //    for (int y = 0; y < srcBox.CountY; y++)
                        //    {
                        //        ComputeShaders.Utilities.CopyMemory(dst, src, (uint)srcBox.CountX * sizeof(float));

                        //        src = IntPtr.Add(src, Dimensions[0] * sizeof(float));
                        //        dst = IntPtr.Add(dst, destination.Dimensions[0] * sizeof(float));
                        //    }

                        //    src = IntPtr.Add(src, diff);
                        //    dst = IntPtr.Add(dst, diff);
                        //}
                    }
                }
            }
        }
        private unsafe void GTGCopy(Tensor destination)
        {
            if (!device.SameNativeDevice(destination.device))
            {
                CTGCopy(destination);
                return;
            }

            switch (destination.Dimensions.Length)
            {
                case 1: fData.CopyResource(destination.fData); break;
                case 2: sData.CopyResource(destination.sData); break;
                case 3: tData.CopyResource(destination.tData); break;
            }
        }
        private unsafe void GTGCopy(Tensor destination, TensorBox srcBox, int dstX, int dstY = 0, int dstZ = 0)
        {
            if (!device.SameNativeDevice(destination.device))
            {
                CTGCopy(destination, srcBox, dstX, dstY, dstZ);
                return;
            }

            switch (destination.Dimensions.Length)
            {
                case 1: fData.CopyTo(destination.fData, srcBox.StartX, dstX, srcBox.CountX); break;
                case 2: sData.CopyTo(destination.sData, srcBox.StartX, srcBox.StartY, dstX, dstY, srcBox.CountX, srcBox.CountY); break;
                case 3: tData.CopyTo(destination.tData, srcBox.StartX, srcBox.StartY, srcBox.StartZ, dstX, dstY, dstZ, srcBox.CountX, srcBox.CountY, srcBox.CountZ); break;
            }
        }

        /// <summary>
        /// Copies the contents of this tensor to <paramref name="destination"/> regardless of the <see cref="ProcessorType"/>.
        /// </summary>
        /// <param name="destination"></param>
        /// <param name="srcBox">The source tensor data box to copy.</param>
        /// <param name="dstX">The x-coordinate of the pixel in destination to start copy to.</param>
        /// <param name="dstY">The y-coordinate of the pixel in destination to start copy to.</param>
        /// <param name="dstZ">The z-coordinate of the pixel in destination to start copy to.</param>
        public unsafe void CopyTo(Tensor destination, TensorBox srcBox, int dstX, int dstY = 0, int dstZ = 0)
        {
            if(ProcessorType == ProcessorType.CPU && destination.ProcessorType == ProcessorType.CPU)
                CTCCopy(destination, srcBox, dstX, dstY, dstZ);
            else if (ProcessorType == ProcessorType.CPU && destination.ProcessorType == ProcessorType.GPU)
                CTGCopy(destination, srcBox, dstX, dstY, dstZ);
            else if (ProcessorType == ProcessorType.GPU && destination.ProcessorType == ProcessorType.GPU)
                GTGCopy(destination, srcBox, dstX, dstY, dstZ);
            else
                GTCCopy(destination, srcBox, dstX, dstY, dstZ);
        }

        /// <summary>
        /// Copies the contents of this tensor to <paramref name="destination"/> regardless of the <see cref="ProcessorType"/>.
        /// <br>Note that both tensors must have the exact same dimensions if this tensor operates on the GPU, and the same total length of on CPU</br>
        /// </summary>
        /// <param name="destination"></param>
        public unsafe void CopyTo(Tensor destination)
        {
            if (ProcessorType == ProcessorType.CPU && destination.ProcessorType == ProcessorType.CPU)
                CTCCopy(destination);
            else if (ProcessorType == ProcessorType.CPU && destination.ProcessorType == ProcessorType.GPU)
                CTGCopy(destination);
            else if (ProcessorType == ProcessorType.GPU && destination.ProcessorType == ProcessorType.GPU)
                GTGCopy(destination);
            else
                GTCCopy(destination);
        }

        /// <summary>
        /// Updates the data stored in the tensor with new data from <paramref name="data"/>.
        /// </summary>
        /// <param name="data">The new data.</param>
        public unsafe void UpdateRawData(IntPtr data)
        {
            if (ProcessorType == ProcessorType.CPU)
            {
                fixed (float* f = &this.data[0])
                {
                    IntPtr dst = (IntPtr)f;

                    ComputeShaders.Utilities.CopyMemory(dst, data, (uint)(this.data.Length * sizeof(float)));
                }

                return;
            }

            switch (Dimensions.Length)
            {
                case 1: fData.UpdateSubresource(data); break;
                case 2:
                    if (CreateNewTexture)
                    {
                        int width = sData.Width, height = sData.Height;

                        sData.Dispose();
                        sData = device.CreateTexture2D(width, height, TextureFormat.R32_Float, data);
                    }
                    else
                        sData.UpdateSubresource(data);
                    break;
                case 3:
                    if (CreateNewTexture)
                    {
                        int width = tData.Width, height = tData.Height, depth = tData.Depth;

                        tData.Dispose();
                        tData = device.CreateTexture3D(width, height, depth, TextureFormat.R32_Float, data);
                    }
                    else
                        tData.UpdateSubresource(data);
                    break;
            }
        }

        /// <summary>
        /// Creates a new gradient tensor.
        /// <br>Note that this tensor is only used to store gradients not calculating them.</br>
        /// </summary>
        public void CreateGradient() => Gradient = EmptyClone();

        /// <summary>
        /// Disposes of the gradient tensor.
        /// </summary>
        public void DisposeGradient()
        {
            Gradient.Dispose();
            Gradient = null;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (ProcessorType == ProcessorType.CPU)
                return;

            switch (Dimensions.Length)
            {
                case 1: fData.Dispose(); break;
                case 2: sData.Dispose(); break;
                case 3: tData.Dispose(); break;
            }
        }
        #endregion

        /// <summary>
        /// Returns an empty clone of the tensor.
        /// </summary>
        /// <returns></returns>
        public Tensor EmptyClone() => new Tensor(device, Dimensions);
    }
}
