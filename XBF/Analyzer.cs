using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;
namespace XBF
{
    public class Analyzer
    {
        public List<Rectangle> getFaceRegions(Mat img, String ssdProtoFile, String ssdFile, int imgDim = 300)
        {
            Emgu.CV.Dnn.Net net = DnnInvoke.ReadNetFromCaffe(ssdProtoFile, ssdFile);
            MCvScalar meanVal = new MCvScalar(104, 177, 123);
            Mat inputBlob = DnnInvoke.BlobFromImage(img, 1.0, new Size(imgDim, imgDim), meanVal, false, false);
            net.SetInput(inputBlob, "data");
            Mat detection = net.Forward("detection_out");

            float confidenceThreshold = 0.5f;

            List<Rectangle> faceRegions = new List<Rectangle>();

            int[] dim = detection.SizeOfDimemsion;
            int step = dim[3] * sizeof(float);
            IntPtr start = detection.DataPointer;
            for (int i = 0; i < dim[2]; i++)
            {
                float[] values = new float[dim[3]];
                Marshal.Copy(new IntPtr(start.ToInt64() + step * i), values, 0, dim[3]);
                float confident = values[2];

                if (confident > confidenceThreshold)
                {
                    float xLeftBottom = values[3] * img.Cols;
                    float yLeftBottom = values[4] * img.Rows;
                    float xRightTop = values[5] * img.Cols;
                    float yRightTop = values[6] * img.Rows;
                    RectangleF objectRegion = new RectangleF(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);
                    Rectangle faceRegion = Rectangle.Round(objectRegion);
                    faceRegions.Add(faceRegion);

                }
            }
            return faceRegions;
        }
    }
}
