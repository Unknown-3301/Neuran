using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using ComputeShaders;
using Neuran.GradientDescent;
using Neuran.Loss;
using Neuran.Utilities;

namespace Neuran.GradientDescent
{
    public class GradientEstimator
    {
        /// <summary>
        /// The amount of change applied to every learnable parameter to estimate the gradients.
        /// </summary>
        public float StepSize { get; set; } = 0.001f;
        /// <summary>
        /// The loss function used.
        /// </summary>
        public ILoss[] LossFunctions { get; set; }
        /// <summary>
        /// The iterator used to get the sequence of input/output data.
        /// </summary>
        public IDataIterator DataIterator { get; set; }

        private unsafe void ChangeElement(Tensor t, float newValue, int index)
        {
            if (t.ProcessorType == ProcessorType.CPU)
            {
                t[index] = newValue;
                return;
            }

            IntPtr scr = (IntPtr)(byte*)&newValue;
            t.AccessRawData(CPUAccessMode.Write, box => ComputeShaders.Utilities.CopyMemory(t.ElementPosition(box, index), scr, sizeof(float)));
        }
        private float[] GetDataCopy(Tensor t)
        {
            float[] data = t.GetData();

            if (t.ProcessorType == ProcessorType.GPU)
                return data;

            float[] copy = new float[data.Length];
            for (int i = 0; i < copy.Length; i++)
            {
                copy[i] = data[i];
            }

            return copy;
        }

        private float TestModel(IModel model, Tensor[] emptyInput, int length)
        {
            model.Reset();
            model.LoadRandomState();

            float totalLoss = 0;

            for (int i = 0; i < length; i++)
            {
                (Tensor[] input, Tensor[] correctOutput) = DataIterator.GetNext();

                Tensor[] model_output = model.Run(input == null ? emptyInput : input);

                if (correctOutput == null || i != length - 1)
                    continue;

                float loss = 0;

                for (int j = 0; j < LossFunctions.Length; j++)
                {
                    loss += LossFunctions[j].GetLoss(model_output[j], correctOutput[j]);
                }

                totalLoss += loss;
            }

            DataIterator.ResetSequence();

            return totalLoss; // removed ( / length) for DEBUG
        }
        private float TestModel2(IModel model, List<(Tensor[] input, Tensor[] output)> data)
        {
            model.Reset();
            model.LoadRandomState();

            float totalLoss = 0;

            for (int i = 0; i < data.Count; i++)
            {
                (Tensor[] input, Tensor[] correctOutput) = data[i];

                Tensor[] model_output = model.Run(input);

                if (i != data.Count - 1)
                    continue;

                float loss = 0;

                for (int j = 0; j < LossFunctions.Length; j++)
                {
                    loss += LossFunctions[j].GetLoss(model_output[j], correctOutput[j]);
                }

                totalLoss += loss;
            }

            DataIterator.ResetSequence();

            return totalLoss; // removed ( / length) for DEBUG
        }
        /// <summary>
        /// Estimates the gradients in <paramref name="model"/> with the data from <see cref="DataIterator"/>.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="model"></param>
        /// <returns></returns>
        public List<Tensor> Estimate(IGradientDescent model)
        {
            DataIterator.LoopSequence = true;
            DataIterator.ResetSequence();
            model.SaveRandomState();

            List<Tensor> estimated = new List<Tensor>();
            List<Tensor> model_parameters = new List<Tensor>();

            model.AddParameters(model_parameters);

            Tensor[] emptyInput = model.Input.EmptyCloneArray();

            int sequenceLength = DataIterator.SequenceLength;

            for (int i = 0; i < model_parameters.Count; i++)
            {
                estimated.Add(new Tensor(null, model_parameters[i].Dimensions));
            }

            for (int seq = 0; seq < sequenceLength; seq++)
            {
                for (int i = 0; i < model_parameters.Count; i++)
                {
                    Tensor parameter = model_parameters[i];

                    float[] values = GetDataCopy(parameter);

                    for (int j = 0; j < parameter.TensorLength; j++)
                    {
                        ChangeElement(parameter, values[j] + StepSize, j);
                        float y2 = TestModel(model, emptyInput, seq + 1);

                        ChangeElement(parameter, values[j] - StepSize, j);
                        float y1 = TestModel(model, emptyInput, seq + 1);

                        ChangeElement(parameter, values[j], j);

                        estimated[i][j] += (y2 - y1) / (2 * StepSize);
                    }
                }
            }

            return estimated;

        }

        /// <summary>
        /// Estimates the gradients for a model and calculate its gradients to be compared.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="estimated"></param>
        /// <param name="calculated"></param>
        public void EstimateAndCalculate(IGradientDescent model, out List<Tensor> estimated, out List<Tensor> calculated)
        {
            estimated = Estimate(model);

            model.Reset();
            model.LoadRandomState();
            model.PrepareGD(DataIterator.SequenceLength);

            Tensor[] derivative = model.Output.EmptyCloneArray();
            Tensor[] derivativeSum = model.Output.EmptyCloneArray();

            Tensor[] emptyInput = model.Input.EmptyCloneArray();

            List<Tensor> param = new List<Tensor>();
            model.AddParameters(param);
            for (int i = 0; i < param.Count; i++)
            {
                param[i].CreateGradient();
            }

            int length = DataIterator.SequenceLength;

            for (int i = 0; i < length; i++)
            {
                (Tensor[] input, Tensor[] correct) = DataIterator.GetNext();

                Tensor[] predicted = model.Run(input == null ? emptyInput : input);

                //float[] d1 = input[0].GetData(); //DEBUG
                //float[] d2 = correct[0].GetData(); //DEBUG
                //float[] d3 = predicted[0].GetData(); //DEBUG

                if (correct == null)
                    continue;

                for (int j = 0; j < LossFunctions.Length; j++)
                {
                    LossFunctions[j].GetDerivative(predicted[j], correct[j], derivativeSum[j], true);
                    derivativeSum[j].CopyTo(derivative[j]);
                }
                //float[] d4 = derivativeSum[0].GetData(); //DEBUG

                //derivative.Divide(i + 1); removed for DEBUG

                model.Backpropagate(derivative, 0);
            }

            calculated = new List<Tensor>(param.Count);
            for (int i = 0; i < param.Count; i++)
            {
                calculated.Add(param[i].EmptyClone());
                param[i].Gradient.CopyTo(calculated[i]);
            }

            model.Reset();
            model.LoadRandomState();
            model.EndGD();
        }

        public List<Tensor[]> EstimatePreLayer(IGradientDescent model) //FOR NOW THIS ONLY WORK FOR NON-SEQUENTIAL MODELS (no LSTMS or Recurrsion)
        {
            DataIterator.LoopSequence = true;
            DataIterator.ResetSequence();
            model.SaveRandomState();

            List<Tensor[]> estimated = new List<Tensor[]>();

            int sequenceLength = DataIterator.SequenceLength;

            for (int i = 0; i < sequenceLength; i++)
            {
                Tensor[] c = new Tensor[model.Input.Length];

                for (int j = 0; j < model.Input.Length; j++)
                {
                    c[j] = new Tensor(null, model.Input[j].Dimensions);
                }

                estimated.Add(c);
            }

            for (int seq = 0; seq < sequenceLength; seq++)
            {
                (Tensor[] input, Tensor[] correct) = DataIterator.GetNext();

                if (input == null)
                    throw new Exception("Null input! When estimating PreLayer, no input from data iterator can be null!");

                float[][] inputData = new float[input.Length][];
                for (int i = 0; i < input.Length; i++)
                {
                    inputData[i] = GetDataCopy(input[i]);
                }

                for (int i = 0; i < input.Length; i++)
                {
                    for (int j = 0; j < input[i].TensorLength; j++)
                    {
                        ChangeElement(input[i], inputData[i][j] + StepSize, j);
                        float y2 = TestModel2(model, new List<(Tensor[] input, Tensor[] output)>() { (input, correct)}); //null because every input from data iterator must not be null (so no need to empty input)

                        ChangeElement(input[i], inputData[i][j] - StepSize, j);
                        float y1 = TestModel2(model, new List<(Tensor[] input, Tensor[] output)>() { (input, correct) }); //null because every input from data iterator must not be null (so no need to empty input)

                        ChangeElement(input[i], inputData[i][j], j);

                        estimated[seq][i][j] += (y2 - y1) / (2 * StepSize);
                    }
                }
            }

            return estimated;
        }
        public void EstimateAndCalculatePreLayer(IGradientDescent model, out List<Tensor[]> estimated, out List<Tensor[]> calculated)//FOR NOW THIS ONLY WORK FOR NON-SEQUENTIAL MODELS (no LSTMS or Recurrsion)
        {
            estimated = EstimatePreLayer(model);

            calculated = new List<Tensor[]>();
            model.Reset();
            model.LoadRandomState();
            model.PrepareGD(DataIterator.SequenceLength);

            Tensor[] derivative = model.Output.EmptyCloneArray();
            Tensor[] derivativeSum = model.Output.EmptyCloneArray();

            List<Tensor> param = new List<Tensor>();
            model.AddParameters(param);
            for (int i = 0; i < param.Count; i++)
            {
                param[i].CreateGradient();
            }

            int length = DataIterator.SequenceLength;

            for (int i = 0; i < length; i++)
            {
                (Tensor[] input, Tensor[] correct) = DataIterator.GetNext();

                Tensor[] predicted = model.Run(input);

                //float[] d1 = input[0].GetData(); //DEBUG
                //float[] d2 = correct[0].GetData(); //DEBUG
                //float[] d3 = predicted[0].GetData(); //DEBUG

                if (correct == null)
                    continue;

                for (int j = 0; j < LossFunctions.Length; j++)
                {
                    LossFunctions[j].GetDerivative(predicted[j], correct[j], derivativeSum[j], true);
                    derivativeSum[j].CopyTo(derivative[j]);
                }
                //float[] d4 = derivativeSum[0].GetData(); //DEBUG

                //derivative.Divide(i + 1); removed for DEBUG

                model.Backpropagate(derivative, 0);

                calculated.Add(model.PreLayerDer.EmptyCloneArray());
                for (int j = 0; j < model.PreLayerDer.Length; j++)
                {
                    model.PreLayerDer[j].CopyTo(calculated[i][j]);
                }
            }

            model.Reset();
            model.LoadRandomState();
            model.EndGD();
        }
    }
}
