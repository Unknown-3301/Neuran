using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ComputeShaders;

namespace Neuran.IDX.CS
{
    /// <summary>
    /// An enum that contains the methods to read idx data (in IntPtr form) to <see cref="CSTexture2D"/> or <see cref="CSTexture2DArray"/>.
    /// </summary>
    public enum TextureReadingMethod
    {
        /// <summary>
        /// Disposes the texture and creates a new one using <see cref="CSDevice.CreateTexture2D(int, int, TextureFormat, IntPtr, bool)"/> or <see cref="CSDevice.CreateTexture2DArray(int, int, TextureFormat, bool, IntPtr[])"/> To read the idx data. This is recommended.
        /// </summary>
        CreateNewTexture,
        /// <summary>
        /// Uses <see cref="ShaderResource{T}.WriteToRawData(Action{TextureDataBox})"/> To read the idx data.
        /// </summary>
        WriteToRawData,
        /// <summary>
        /// Uses <see cref="CSTexture2D.UpdateSubresource(IntPtr)"/> or <see cref="CSTexture2DArray.UpdateSubresource(IntPtr)"/> To read the idx data.
        /// </summary>
        UpdateSubresource,
    }
}
