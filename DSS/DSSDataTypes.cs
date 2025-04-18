using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.DSS
{
    public enum DSSDataTypes
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
