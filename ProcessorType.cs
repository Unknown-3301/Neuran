using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran
{
    /// <summary>
    /// An enum for the type of processor used to process information
    /// </summary>
    public enum ProcessorType
    {
        /// <summary>
        /// The object operates on the CPU.
        /// </summary>
        CPU,

        /// <summary>
        /// The object operates on the GPU.
        /// </summary>
        GPU
    }
}
