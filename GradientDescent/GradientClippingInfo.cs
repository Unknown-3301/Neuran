using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.GradientDescent
{
    /// <summary>
    /// Info for gradient clipping during training.
    /// </summary>
    public struct GradientClippingInfo
    {
        /// <summary>
        /// The maximum gradient allowed (as norm not value)
        /// </summary>
        public float Max { get; set; }
        /// <summary>
        /// The minimum gradient allowed (as norm not value)
        /// </summary>
        public float Min { get; set; }
    }
}
