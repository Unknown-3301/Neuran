using Neuran.GradientDescent;
using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Models
{
    internal class MultiHeadLayerHelper : IGradientDescent
    {
        //not needed
        public Tensor[] PreLayerDer => throw new NotImplementedException();

        public IGradientDescent ConnectedFrom => throw new NotImplementedException();

        public Tensor[] Output => throw new NotImplementedException();

        public Tensor[] Input => throw new NotImplementedException();

        private IGradientDescent connectedTo;
        private int startIndex;
        private List<Tensor[]> pastPreLayers; //0 means the prelayer with past time 0
        private List<int> pastTimes;


        public MultiHeadLayerHelper(IGradientDescent connectedTo, List<Tensor[]> preLayers, List<int> pastTimes, int startIndex)
        {
            this.connectedTo = connectedTo;
            this.startIndex = startIndex;
            this.pastTimes = pastTimes;
            pastPreLayers = preLayers;

            connectedTo.Connect(this);
        }

        public void AddParameters(List<Tensor> parameters)
        {
            
        }

        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            for (int i = 0; i < lossDer.Length; i++)
            {
                pastPreLayers[pastTime][i + startIndex].Add(lossDer[i]);
            }

            if (!pastTimes.Contains(pastTime))
                pastTimes.Add(pastTime);
        }

        public void Connect(IGradientDescent model)
        {
            
        }

        public void Dispose()
        {
            
        }

        public void EndGD()
        {
            
        }

        public void LoadRandomState()
        {
            
        }

        public void PrepareGD(int maxTruncatedLength)
        {
            
        }

        public void Reset()
        {
            
        }

        public void ResetGradients()
        {
            
        }

        public Tensor[] Run(Tensor[] input)
        {
            throw new NotImplementedException();
        }

        public void SaveRandomState()
        {
            
        }
    }
}
