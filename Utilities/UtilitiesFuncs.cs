using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Utilities
{
    public static class UtilitiesFuncs
    {
        /// <summary>
        /// Return normal distribution number
        /// </summary>
        /// <param name="Mean">The common number in range</param>
        /// <param name="StanDev">The positive of range</param>
        /// <param name="rand">Random seed</param>
        /// <returns></returns>
        public static double RandomGaussain(double Mean, double StanDev, Random rand)
        {
            double u1 = 1.0 - rand.NextDouble();
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            double randNormal = Mean + StanDev * randStdNormal;
            return randNormal;
        }

        public static Tensor[] EmptyCloneArray(this Tensor[] tensors)
        {
            Tensor[] clone = new Tensor[tensors.Length];
            for (int i = 0; i < tensors.Length; i++)
            {
                clone[i] = tensors[i].EmptyClone();
            }
            return clone;
        }
    }
}
