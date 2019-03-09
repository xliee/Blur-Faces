using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;

namespace Blur
{
    class Detector
    {
        public List<Rectangle> getFaceRegions (Mat img, String ssdProtoFile, String ssdFile, int imgDim = 300)
        {
            Emgu.CV.Dnn.Net net = DnnInvoke.ReadNetFromCaffe(ssdProtoFile, ssdFile);

            //Mat img = img2;

            //int imgDim = 300;
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
        public List<VectorOfVectorOfPointF> getLandmarks(Mat img, List<Rectangle> faceRegions, String facemarkFileName)
        {
            List<VectorOfVectorOfPointF> landmarks = new List<VectorOfVectorOfPointF>(faceRegions.Count);
            for(int i = 0; i <= faceRegions.Count-1; i++)
            {
                landmarks.Add( new VectorOfVectorOfPointF());
            }
            FacemarkLBFParams facemarkParam = new Emgu.CV.Face.FacemarkLBFParams();
            FacemarkLBF facemark = new Emgu.CV.Face.FacemarkLBF(facemarkParam);
            for (int j = 0; j <= faceRegions.Count-1; j++)
            {
                Rectangle[] faces = { faceRegions[j] };
                VectorOfRect vr = new VectorOfRect(faces);
                try
                {
                    facemark.LoadModel(facemarkFileName);
                    facemark.Fit(img, vr, landmarks[j]);
                }
                catch (Emgu.CV.Util.CvException ex)
                {
                    Console.WriteLine(ex.ToString());
                }

            }
            return landmarks;
        }
        public Mat drawFaceMark(Mat img)
        {
            //Mat img = img2;

            int imgDim = 300;
            MCvScalar meanVal = new MCvScalar(104, 177, 123);
            String ssdFile = "res10_300x300_ssd_iter_140000.caffemodel";
            String ssdProtoFile = "deploy.prototxt";
            String facemarkFileName = "lbfmodel.yaml";

            //CheckAndDownloadFile(ssdFile, "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/");
            //CheckAndDownloadFile(ssdProtoFile, "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/");
            //CheckAndDownloadFile(facemarkFileName, "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/");

            List<Rectangle> faceRegions = getFaceRegions(img, ssdProtoFile, ssdFile, imgDim);
            List<VectorOfVectorOfPointF> landmarks = getLandmarks(img, faceRegions, facemarkFileName);



            int numFaces = faceRegions.Count;
            for (int j = 0; j <= faceRegions.Count-1; j++)
            {
                CvInvoke.Rectangle(img, faceRegions[j], new MCvScalar(0, 255, 0));

                for (int i = 0; i < landmarks[i].Size; i++)
                {
                    using (VectorOfPointF vpf = landmarks[j][i])
                        try
                        {
                            FaceInvoke.DrawFacemarks(img, vpf, new MCvScalar(255, 0, 0));
                        }
                        catch(Emgu.CV.Util.CvException ex)
                        {
                            Console.WriteLine(ex.ToString());
                        }
                }
                blur blurer = new blur();
                PointF[][] points = landmarks[j].ToArrayOfArray();
                //CvInvoke.DrawContours(img, new VectorOfPointF(points[0]), 1, new MCvScalar(255,0,0));
                img = new Image<Bgr, Byte>((Bitmap)blurer.BlurPath(img.ToImage<Bgr, byte>().Bitmap, points[0], faceRegions[j])).Mat;
            }
            //CvInvoke.Imwrite("rgb_ssd_facedetect.jpg", img);
            return img;

        }
        private static void CheckAndDownloadFile(String fileName, String url)
        {
            //String emguS3Base = "https://s3.amazonaws.com/emgu-public/";
            if (!File.Exists(fileName))
            {
                //Download the ssd file    
                String fileUrl = url + fileName;
                Trace.WriteLine("downloading file from:" + fileUrl + " to: " + fileName);
                System.Net.WebClient downloadClient = new System.Net.WebClient();
                try
                {
                    downloadClient.DownloadFile(fileUrl, fileName);
                }
                catch
                {
                    File.Delete(fileName);
                    throw;
                }
            }
        }

    }
}
