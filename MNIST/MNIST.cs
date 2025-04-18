using ComputeShaders;
using System;
using System.Collections.Generic;
using System.IO.Compression;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace Neuran.MNIST
{
    public static class MNIST
    {
        private static Stream DownloadDecompress(string url, string path, WebClient client)
        {
            if (File.Exists(path))
                File.Delete(path);

            client.DownloadFile(url, path);
            Stream s = File.OpenRead(path);
            GZipStream stream = new GZipStream(s, CompressionMode.Decompress, false);

            string decompressedPath = Path.ChangeExtension(path, "(decompressed).idx");

            FileStream fs = File.Create(decompressedPath);

            stream.CopyTo(fs);
            fs.Close();

            return File.Open(decompressedPath, FileMode.Open); //not sure the best way (most likely not)
        }
        private static MNISTDataIterator DownloadData(string input_url, string label_url, string input_path, string label_path, WebClient c, CSDevice device)
        {
            Stream input_stream = DownloadDecompress(input_url, input_path, c);
            Stream label_stream = DownloadDecompress(label_url, label_path, c);

            return new MNISTDataIterator(input_stream, label_stream, SequenceType.ManyToMany, device);
        }
        /// <summary>
        /// Get the mnist data from <paramref name="saveDirectory"/> or download them and save the compressed and uncompressed data in the said directory.
        /// Note that the iterators returned store the input as a 3D Tensor.
        /// </summary>
        /// <param name="saveDirectory">The path to the folder to store the data to.</param>
        /// <param name="device">The d3d11 device for the input tensor in both iterators returned.</param>
        /// <returns></returns>
        public static (MNISTDataIterator training, MNISTDataIterator testing) GetOrDownloadData(string saveDirectory, CSDevice device)
        {
            WebClient c = new WebClient();
            MNISTDataIterator training = DownloadData("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", saveDirectory + "train-images-idx3-ubyte.gz", saveDirectory + "train-labels-idx1-ubyte.gz", c, device);
            MNISTDataIterator testing = DownloadData("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", saveDirectory + "t10k-images-idx3-ubyte.gz", saveDirectory + "t10k-labels-idx1-ubyte.gz", c, device);
            c.Dispose();

            return (training, testing);
        }
    }
}
