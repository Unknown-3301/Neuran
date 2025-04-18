using System.Collections;
using System.Collections.Generic;

namespace Neuran.IDX
{
    /// <summary>
    /// The types of data to store in idx files.
    /// </summary>
    public enum IDXDataTypes
    {
        /// <summary>
        /// A single byte that ranges from 0 to 255
        /// </summary>
        UnsignedByte,
        /// <summary>
        /// A single byte that ranges from -127 to 127
        /// </summary>
        SignedByte,
        /// <summary>
        /// Short (2 bytes)
        /// </summary>
        Short,
        /// <summary>
        /// Integer (4 bytes)
        /// </summary>
        Int,
        /// <summary>
        /// Float (4 bytes)
        /// </summary>
        Float,
        /// <summary>
        /// Double (8 bytes)
        /// </summary>
        Double,
    }
}
