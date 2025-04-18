using ComputeShaders;
using ComputeShaders.Windows;
using Neuran.Activations;
using Neuran.GradientDescent;
using Neuran.Loss;
using Neuran.Optimizers;
using Neuran.Utilities;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net.Http.Headers;
using System.Runtime.Remoting.Metadata.W3cXsd2001;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.GradientDescent
{
    public class GradientDescentTrainer
    {
        private int sequenceIndex, currentSequenceLength, increment;

        private IGradientDescent model;
        private ILoss[] lossFunctions;
        private IOptimizer optimizer;
        private IDataIterator trainingData;
        private bool shuffle;
        private int maxTruncatedLength;
        private int batchSize;
        private Random random;

        private Tensor[] derivative, emptyInput;

        private List<Tensor> parameters;

        
        /// <summary>
        /// Info for applying gradient clipping.
        /// </summary>
        public GradientClipper Clipping { get; set; }
        /// <summary>
        /// This is run before the algorithm applies the gradients to the model parameters. Usually, this is used for debugging. (int, int ) -> (current epoch, current sequence)
        /// </summary>
        public Action<int, int> beforeGradients { get; set; }
        /// <summary>
        /// The number of epochs finished
        /// </summary>
        public int FinishedEpochs { get; private set; }
        /// <summary>
        /// The number of sequences finished. This counter resets every epoch.
        /// </summary>
        public int FinishedSequences { get => sequenceIndex; }

        public GradientDescentTrainer(IGradientDescent model, ILoss[] lossFunctions, IOptimizer optimizer, IDataIterator trainingData, bool shuffle, int maxTruncatedLength, int batchSize, Random random, GradientClippingInfo? clipInfo = null)
        {
            this.model = model;
            this.lossFunctions = lossFunctions;
            this.optimizer = optimizer;
            this.trainingData = trainingData;
            this.shuffle = shuffle;
            this.maxTruncatedLength = maxTruncatedLength;
            this.batchSize = batchSize;
            this.random = random;

            currentSequenceLength = trainingData.SequenceLength;

            model.PrepareGD(maxTruncatedLength);

            parameters = new List<Tensor>();
            model.AddParameters(parameters);

            if (clipInfo != null)
                Clipping = new GradientClipper(parameters, clipInfo.Value);

            for (int i = 0; i < parameters.Count; i++)
            {
                optimizer.AddParameter(parameters[i]);
            }

            derivative = model.Output.EmptyCloneArray();
            emptyInput = model.Input.EmptyCloneArray();
        }

        //private void old()
        //{
        //    for (int i = 0; i < currentSequenceLength; i++)
        //    {
        //        (Tensor input, Tensor correctOutput) = trainingData.GetNext();
        //
        //        Tensor predictedOutput = model.Run(input == null ? emptyInput : input);
        //
        //        if (correctOutput == null)
        //            continue;
        //
        //        lossFunction.GetDerivative(predictedOutput, correctOutput, derivativeSum, i == 0);
        //
        //        derivativeSum.Divide(batchSize);
        //
        //        if (currentSequenceLength != 1)
        //        {
        //            derivativeSum.CopyTo(derivative);
        //
        //            if (i != 0)
        //                derivative.Divide(i + 1);
        //        }
        //
        //        model.Backpropagate(currentSequenceLength == 1 ? derivativeSum : derivative, 0);
        //
        //        increment++;
        //        if (increment % batchSize == 0)
        //        {
        //            increment = 0;
        //
        //            Clipping?.Clip();
        //
        //            beforeGradients?.Invoke(FinishedEpochs, sequenceIndex);
        //
        //            optimizer.ApplyAll();
        //        }
        //
        //    }
        //}
        private void SingleSequence()
        {
            for (int i = 0; i < currentSequenceLength; i++)
            {
                (Tensor[] input, Tensor[] correctOutput) = trainingData.GetNext();

                Tensor[] predictedOutput = model.Run(input == null ? emptyInput : input);

                if (correctOutput == null)
                    continue;

                for (int j = 0; j < lossFunctions.Length; j++)
                {
                    lossFunctions[j].GetDerivative(predictedOutput[j], correctOutput[j], derivative[j], true);
                    derivative[j].Divide(batchSize * trainingData.OutputLength);
                }

                model.Backpropagate(derivative, 0);

                increment++;
                if (increment % batchSize == 0)
                {
                    increment = 0;

                    Clipping?.Clip();

                    beforeGradients?.Invoke(FinishedEpochs, sequenceIndex);

                    optimizer.ApplyAll();
                }
            }
            
        }
        public void Iterate(int sequencesPerIteration)
        {
            for (int i = 0; i < sequencesPerIteration; i++)
            {
                SingleSequence();

                model.Reset();

                sequenceIndex++;
                currentSequenceLength = trainingData.SequenceLength;

                if (sequenceIndex >= trainingData.Length)
                {
                    sequenceIndex = 0;
                    FinishedEpochs++;

                    if (shuffle)
                        trainingData.Shuffle(random);
                }
            }
        }

        /// <summary>
        /// Trains the model using gradient descent.
        /// </summary>
        /// <param name="model">The model to train.</param>
        /// <param name="lossFunction"></param>
        /// <param name="optimizer"></param>
        /// <param name="trainingData"></param>
        /// <param name="epochs"></param>
        /// <param name="shuffle"></param>
        /// <param name="maxTruncatedLength"></param>
        /// <param name="batchSize"></param>
        /// <param name="random"></param>
        /// <param name="clipInfo"></param>
        /// <param name="forEverySequnce">A log action that runs after every sequence. (int, int) -> (epoch, sequenceIndex)</param>
        public static void Train(IGradientDescent model, ILoss[] lossFunctions, IOptimizer optimizer, IDataIterator trainingData, int epochs, bool shuffle, int maxTruncatedLength, int batchSize, Random random, Action<int, int> forEverySequnce = null, GradientClippingInfo? clipInfo = null)
        {
            model.PrepareGD(maxTruncatedLength);

            List<Tensor> parameters = new List<Tensor>();
            model.AddParameters(parameters);

            for (int i = 0; i < parameters.Count; i++)
            {
                optimizer.AddParameter(parameters[i]);
            }

            GradientClipper clipper = null;

            if (clipInfo != null)
                clipper = new GradientClipper(parameters, clipInfo.Value);

            Tensor[] derivative = model.Output.EmptyCloneArray();
            Tensor[] emptyInput = model.Input.EmptyCloneArray();

            int inc = 1;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int seq = 0; seq < trainingData.Length; seq++)
                {
                    int sequenceLength = trainingData.SequenceLength;

                    Tensor[] predictedOutput = null;
                    Tensor[] correctOutput = null;
                    Tensor[] input = null;

                    for (int i = 0; i < sequenceLength; i++)
                    {
                        (input, correctOutput) = trainingData.GetNext();

                        predictedOutput = model.Run(input ?? emptyInput);

                        if (correctOutput == null)
                            continue;

                        for (int j = 0; j < lossFunctions.Length; j++)
                        {
                            lossFunctions[j].GetDerivative(predictedOutput[j], correctOutput[j], derivative[j], true);
                            derivative[j].Divide(batchSize * trainingData.OutputLength);
                        }

                        model.Backpropagate(derivative, 0);
                    }

                    if (inc % batchSize != 0)
                        forEverySequnce?.Invoke(epoch, seq);

                    if (inc % batchSize == 0)
                    {
                        clipper?.Clip();

                        forEverySequnce?.Invoke(epoch, seq);

                        optimizer.ApplyAll();

                        inc = 0;
                    }
                    
                    inc++;

                    model.Reset(); //reset recurrsion
                }

                if (shuffle)
                    trainingData.Shuffle(random);
            }

            for (int i = 0; i < emptyInput.Length; i++)
            {
                emptyInput[i].Dispose();
            }
        }
    }
}
