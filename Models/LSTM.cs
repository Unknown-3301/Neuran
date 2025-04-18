using ComputeShaders;
using Neuran.Activations;
using Neuran.GradientDescent;
using Neuran.Utilities;
using SharpDX.D3DCompiler;
using SharpDX.DXGI;
using System;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.Models
{
    public class LSTM : IGradientDescent
    {
        public Tensor[] Input { get; private set; }
        public Tensor[] Output { get; private set; }

        public Tensor[] PreLayerDer { get => preLayerDer; }

        public IGradientDescent ConnectedFrom { get; private set; }

        private Tensor cellState;
        private Tensor hiddenState; //the actual output

        private Tensor preCellState;
        private Tensor preHiddenState;

        private FullyConnectedLayer inputLayer; //not the actual input //its output is the input gate
        private FullyConnectedLayer forgetLayer; //its output is the forget gate
        private FullyConnectedLayer cellLayer; //its output is the cell gate
        private FullyConnectedLayer outputLayer; //not the actual output //its output is the output gate

        private Tanh cellAct;

        // Intermediate tensors
        private Tensor hiddenInput; //the concatenation of both the input and the previous hidden state (the previous output)
        private Tensor cXf; //the result of multiplying previous cell state with the forget gate
        private Tensor iXg; //the result of multiplying input gate with the cell gate
        private Tensor hyperCellState; //the result of applying Tanh to the cell state

        // Derivative Tensors
        private Tensor hyperCellStateNextDer; // The derivative of the loss w.r.t the hyperCellState (after Tanh)
        private Tensor hyperCellStatePreDer; // The derivative of the loss w.r.t the before activation hyperCellState (before Tanh)
        private Tensor outputLayerNextDer; // The derivative of the loss w.r.t the output of "outputLayer"
        private Tensor forgetLayerNextDer; // The derivative of the loss w.r.t the output of "forgetLayer"
        private Tensor inputLayerNextDer; // The derivative of the loss w.r.t the output of "inputLayer"
        private Tensor celltLayerNextDer; // The derivative of the loss w.r.t the output of "cellLayer"
        private Tensor preCellStateDer; // The derivative of the loss w.r.t the previous cell state
        private Tensor fullInputDer; // The tensor that holds the derivative of loss w.r.t the contatenated vector of both input x in hidden state ht-1
        private Tensor preHiddenStateDer; // The derivative of the loss w.r.t the previous hidden state.
        private Tensor[] preLayerDer;

        private Tensor[] inputLayerOutputs;
        private Tensor[] forgetLayerOutputs;
        private Tensor[] cellLayerOutputs;
        private Tensor[] outputLayerOutputs;
        private Tensor[] hyperCellStates;
        private Tensor[] cellStates;
        private int iteratedTime;

        private Tensor emptyPreCellStateDer;

        //DEBUG
        //public List<(int, List<float[]>)> DebugData = new List<(int, List<float[]>)>();


        /// <summary>
        /// Creates a new instance.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        public LSTM(int input, int output, Random random, CSDevice device = null)
        {
            forgetLayer = new FullyConnectedLayer(input + output, output, new Sigmoid(device), random, device);
            inputLayer = new FullyConnectedLayer(input + output, output, new Sigmoid(device), random, device);
            cellLayer = new FullyConnectedLayer(input + output, output, new Tanh(device), random, device);
            outputLayer = new FullyConnectedLayer(input + output, output, new Sigmoid(device), random, device);

            cellState = new Tensor(device, output);
            preCellState = new Tensor(device, output);
            hiddenState = new Tensor(device, output);
            preHiddenState = new Tensor(device, output);

            hiddenInput = new Tensor(device, input + output);
            cXf = new Tensor(device, output);
            iXg = new Tensor(device, output);
            hyperCellState = new Tensor(device, output);

            cellAct = new Tanh(device);

            Input = new Tensor[] { new Tensor(device, input) };
            Output = new Tensor[] { new Tensor(device, output) };
        }


        private Tensor tocpu(Tensor x)
        {
            Tensor r = new Tensor(null, x.Dimensions);
            x.CopyTo(r); return r;
        }
        private Tensor tocpu(IModel model) => tocpu(model.Output[0]);
        public List<Tensor> DEBUG()
        {
            return new List<Tensor>
            {
                tocpu(hiddenInput),
                tocpu(forgetLayer),
                tocpu(inputLayer),
                tocpu(cellLayer),
                tocpu(outputLayer),

                tocpu(cellState),
                tocpu(hiddenState),

                tocpu(cXf),
                tocpu(iXg),
                tocpu(hyperCellState)
            };
        }
        private float[] copy(Tensor t)
        {
            if (t.ProcessorType == ProcessorType.GPU)
                return t.GetData();

            float[] d = new float[t.TensorLength];
            for (int i = 0; i < d.Length; i++)
            {
                d[i] = t[i];
            }

            return d;
        }
        private float[] Multiply(float[] a, float[] b) //DEBUG
        {
            float[] c = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                c[i] = a[i] * b[i];
            }
            return c;
        }
        private float[] Add(float[] a, float[] b) //DEBUG
        {
            float[] c = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                c[i] = a[i] + b[i];
            }
            return c;
        }
        public Tensor[] Run(Tensor[] input)
        {
            Tensor[] hiddens = new Tensor[] { hiddenInput };

            input[0].CopyTo(Input[0]);
            hiddenState.CopyTo(preHiddenState);
            cellState.CopyTo(preCellState);

            //float[] d1 = Input[0].GetData();//DEBUG
            //float[] d2 = preHiddenState.GetData();//DEBUG
            //float[] d3 = preCellState.GetData();//DEBUG

            input[0].CopyTo(hiddenInput, new TensorBox(0, input[0].TensorLength), 0);
            preHiddenState.CopyTo(hiddenInput, new TensorBox(0, preHiddenState.TensorLength), input[0].TensorLength);

            //float[] d4 = hiddenInput.GetData();//DEBUG

            preCellState.CopyTo(cXf);

            //float[] d5 = cXf.GetData();//DEBUG

            cXf.Multiply(forgetLayer.Run(hiddens)[0]);

            //float[] d6 = cXf.GetData();//DEBUG
            //float[] d7 = forgetLayer.Output[0].GetData();//DEBUG
            //float[] d8 = Multiply(d5, d7); //DEBUG

            inputLayer.Run(hiddens)[0].CopyTo(iXg);

            //float[] d9 = iXg.GetData(); //DEBUG

            iXg.Multiply(cellLayer.Run(hiddens)[0]);

            //float[] d10 = iXg.GetData(); //DEBUG
            //float[] d11 = cellLayer.Output[0].GetData(); //DEBUG
            //float[] d12 = Multiply(d9, d11); //DEBUG


            cXf.CopyTo(cellState);
            cellState.Add(iXg);

            //float[] d13 = cellState.GetData(); //DEBUG
            //float[] d14 = Add(d10, d6); //DEBUG


            cellAct.Activate(cellState, hyperCellState);

            outputLayer.Run(hiddens)[0].CopyTo(hiddenState);
            hiddenState.Multiply(hyperCellState);

            hiddenState.CopyTo(Output[0]);

            iteratedTime++;
            if (inputLayerOutputs != null)
            {
                for (int i = Math.Min(iteratedTime, inputLayerOutputs.Length) - 2; i >= 0; i--)
                {
                    inputLayerOutputs[i].CopyTo(inputLayerOutputs[i + 1]);
                    forgetLayerOutputs[i].CopyTo(forgetLayerOutputs[i + 1]);
                    cellLayerOutputs[i].CopyTo(cellLayerOutputs[i + 1]);
                    outputLayerOutputs[i].CopyTo(outputLayerOutputs[i + 1]);
                    hyperCellStates[i].CopyTo(hyperCellStates[i + 1]);
                    cellStates[i].CopyTo(cellStates[i + 1]);
                }

                inputLayer.Output[0].CopyTo(inputLayerOutputs[0]);
                forgetLayer.Output[0].CopyTo(forgetLayerOutputs[0]);
                cellLayer.Output[0].CopyTo(cellLayerOutputs[0]);
                outputLayer.Output[0].CopyTo(outputLayerOutputs[0]);
                hyperCellState.CopyTo(hyperCellStates[0]);
                cellState.CopyTo(cellStates[0]);
            }

            return Output;
        }

        private void InnerBackpropagation(Tensor preCellSDer, Tensor preHiddenSDer, int pastTime)
        {
            //List<float[]> data = new List<float[]>(); //DEBUG

            preHiddenSDer.CopyTo(hyperCellStateNextDer);
            preHiddenSDer.CopyTo(outputLayerNextDer);

            //data.Add(copy(hyperCellStateNextDer)); //DEBUG

            // dL/d(x+h) 1st chain (chain to output layer)
            hyperCellStateNextDer.Multiply(outputLayerOutputs[pastTime]);
            //data.Add(copy(hyperCellStateNextDer)); //DEBUG

            outputLayerNextDer.Multiply(hyperCellStates[pastTime]);
            //data.Add(copy(outputLayerNextDer)); //DEBUG

            outputLayer.Backpropagate(new Tensor[] { outputLayerNextDer }, pastTime);
            Tensor outputLayerPreDer = outputLayer.PreLayerDer[0];
            //data.Add(copy(outputLayerPreDer)); //DEBUG

            hyperCellStateNextDer.CopyTo(hyperCellStatePreDer);

            cellAct.GetDerivative(cellStates[pastTime], hyperCellStates[pastTime], hyperCellStatePreDer);
            //data.Add(copy(hyperCellStatePreDer)); //DEBUG
            hyperCellStatePreDer.Add(preCellSDer); //******************************************
            //data.Add(copy(hyperCellStatePreDer)); //DEBUG

            // dL/d(Ct-1) derivative of loss w.r.t previous cell state
            hyperCellStatePreDer.CopyTo(preCellStateDer);
            preCellStateDer.Multiply(forgetLayerOutputs[pastTime]);
            //data.Add(copy(preCellStateDer)); //DEBUG

            // dL/d(x+h) 2nd chain (chain to forget layer)
            hyperCellStatePreDer.CopyTo(forgetLayerNextDer);

            if (pastTime + 1 < Math.Min(iteratedTime, inputLayerOutputs.Length))
                forgetLayerNextDer.Multiply(cellStates[pastTime + 1]);
            else
                forgetLayerNextDer.Zero();
            //data.Add(copy(forgetLayerNextDer)); //DEBUG

            forgetLayer.Backpropagate(new Tensor[] { forgetLayerNextDer }, pastTime);
            Tensor forgetLayerPreDer = forgetLayer.PreLayerDer[0];
            //data.Add(copy(forgetLayerPreDer)); //DEBUG

            // dL/d(x+h) 3rd chain (chain to cell layer)
            hyperCellStatePreDer.CopyTo(celltLayerNextDer);

            celltLayerNextDer.Multiply(inputLayerOutputs[pastTime]);
            //data.Add(copy(celltLayerNextDer)); //DEBUG
            cellLayer.Backpropagate(new Tensor[] { celltLayerNextDer }, pastTime);
            Tensor cellLayerPreDer = cellLayer.PreLayerDer[0];
            //data.Add(copy(cellLayerPreDer)); //DEBUG

            // dL/d(x+h) 4th chain (chain to input layer)
            hyperCellStatePreDer.CopyTo(inputLayerNextDer);
            inputLayerNextDer.Multiply(cellLayerOutputs[pastTime]);
            //data.Add(copy(inputLayerNextDer)); //DEBUG
            inputLayer.Backpropagate(new Tensor[] { inputLayerNextDer }, pastTime);
            Tensor inputLayerPreDer = inputLayer.PreLayerDer[0];
            //data.Add(copy(inputLayerPreDer)); //DEBUG

            // combines all chains
            fullInputDer.Zero();
            fullInputDer.Add(outputLayerPreDer);
            fullInputDer.Add(forgetLayerPreDer);
            fullInputDer.Add(cellLayerPreDer);
            fullInputDer.Add(inputLayerPreDer);
            //data.Add(copy(fullInputDer)); //DEBUG

            fullInputDer.CopyTo(preLayerDer[0], new TensorBox(0, preLayerDer[0].TensorLength), 0);
            fullInputDer.CopyTo(preHiddenStateDer, new TensorBox(preLayerDer[0].TensorLength, preHiddenStateDer.TensorLength), 0);
            //data.Add(copy(preLayerDer[0])); //DEBUG
            //data.Add(copy(preHiddenStateDer)); //DEBUG

            //DebugData.Add((pastTime, data));//DEBUG

            ConnectedFrom?.Backpropagate(preLayerDer, pastTime);

            if (pastTime + 1 < Math.Min(iteratedTime, inputLayerOutputs.Length))
                InnerBackpropagation(preCellStateDer, preHiddenStateDer, pastTime + 1);
        }
        public void Backpropagate(Tensor[] lossDer, int pastTime)
        {
            InnerBackpropagation(emptyPreCellStateDer, lossDer[0], pastTime);
        }


        /// <inheritdoc/>
        public void PrepareGD(int maxTruncatedLength)
        {
            CSDevice device = Input[0].device;
            int output = Output[0].TensorLength;
            int input = Input[0].TensorLength;

            hyperCellStateNextDer = new Tensor(device, output);
            hyperCellStatePreDer = new Tensor(device, output);
            outputLayerNextDer = new Tensor(device, output);
            forgetLayerNextDer = new Tensor(device, output);
            inputLayerNextDer = new Tensor(device, output);
            celltLayerNextDer = new Tensor(device, output);
            preCellStateDer = new Tensor(device, output);
            fullInputDer = new Tensor(device, input + output);
            preHiddenStateDer = new Tensor(device, output);
            preLayerDer = new Tensor[] { new Tensor(device, input) };
            emptyPreCellStateDer = cellState.EmptyClone();

            inputLayerOutputs = new Tensor[maxTruncatedLength];
            forgetLayerOutputs = new Tensor[maxTruncatedLength];
            cellLayerOutputs = new Tensor[maxTruncatedLength];
            outputLayerOutputs = new Tensor[maxTruncatedLength];
            hyperCellStates = new Tensor[maxTruncatedLength];
            cellStates = new Tensor[maxTruncatedLength];

            for (int i = 0; i < maxTruncatedLength; i++)
            {
                inputLayerOutputs[i] = new Tensor(device, output);
                forgetLayerOutputs[i] = new Tensor(device, output);
                cellLayerOutputs[i] = new Tensor(device, output);
                outputLayerOutputs[i] = new Tensor(device, output);
                hyperCellStates[i] = new Tensor(device, output);
                cellStates[i] = new Tensor(device, output);
            }

            inputLayer.PrepareGD(maxTruncatedLength);
            outputLayer.PrepareGD(maxTruncatedLength);
            forgetLayer.PrepareGD(maxTruncatedLength);
            cellLayer.PrepareGD(maxTruncatedLength);
        }
        /// <inheritdoc/>
        public void Dispose()
        {
            if (hyperCellStateNextDer != null)
                EndGD();

            Input[0].Dispose();
            Output[0].Dispose();
            cellState.Dispose();
            hiddenState.Dispose();
            preCellState.Dispose();
            preHiddenState.Dispose();

            inputLayer.Dispose();
            cellLayer.Dispose();
            outputLayer.Dispose();
            forgetLayer.Dispose();

            hiddenInput.Dispose();
            iXg.Dispose();
            cXf.Dispose();
            hyperCellState.Dispose();
            emptyPreCellStateDer.Dispose();
        }
        /// <inheritdoc/>
        public void EndGD()
        {
            hyperCellStateNextDer.Dispose();
            hyperCellStateNextDer = null;
            hyperCellStatePreDer.Dispose();
            outputLayerNextDer.Dispose();
            forgetLayerNextDer.Dispose();
            inputLayerNextDer.Dispose();
            celltLayerNextDer.Dispose();
            preCellStateDer.Dispose();
            fullInputDer.Dispose();
            preHiddenStateDer.Dispose();
            preLayerDer[0].Dispose();

            for (int i = 0; i < inputLayerOutputs.Length; i++)
            {
                inputLayerOutputs[i].Dispose();
                forgetLayerOutputs[i].Dispose();
                cellLayerOutputs[i].Dispose();
                outputLayerOutputs[i].Dispose();
                hyperCellStates[i].Dispose();
                cellStates[i].Dispose();
            }

            inputLayerOutputs = null;
            forgetLayerOutputs = null;
            cellLayerOutputs = null;
            outputLayerOutputs = null;
            hyperCellStates = null;
            cellStates = null;

            inputLayer.EndGD();
            outputLayer.EndGD();
            forgetLayer.EndGD();
            cellLayer.EndGD();
        }
        /// <inheritdoc/>
        public void AddParameters(List<Tensor> parameters)
        {
            inputLayer.AddParameters(parameters);
            outputLayer.AddParameters(parameters);
            forgetLayer.AddParameters(parameters);
            cellLayer.AddParameters(parameters);
        }
        /// <inheritdoc/>
        public void Connect(IGradientDescent model)
        {
            ConnectedFrom = model;
        }
        /// <inheritdoc/>
        public void ResetGradients()
        {
            inputLayer.ResetGradients();
            outputLayer.ResetGradients();
            forgetLayer.ResetGradients();
            cellLayer.ResetGradients();
        }
        /// <inheritdoc/>
        public void Reset()
        {
            inputLayer.Reset();
            outputLayer.Reset();
            forgetLayer.Reset();
            cellLayer.Reset();

            hiddenState.Zero();
            cellState.Zero();

            iteratedTime = 0;
        }
        /// <inheritdoc/>
        public void SaveRandomState()
        {

        }
        /// <inheritdoc/>
        public void LoadRandomState()
        {

        }


    }
}
