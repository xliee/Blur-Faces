using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Reflection;
using System.Runtime.InteropServices;

namespace XBF
{
    public class Analyzer
    {
        private bool DEBUG = false;
        private blur blurer;
        private String ssdFile;
        private String ssdProtoFile;
        private String facemarkFileName;

        public Analyzer(bool Debug = false, String ssdFile = null, String ssdProtoFile = null, String facemarkFileName = null)
        {
            
            this.DEBUG = Debug;
            this.ssdFile = ssdFile;
            this.ssdProtoFile = ssdProtoFile;
            this.facemarkFileName = facemarkFileName;
            blurer = new blur();
        }

        /// <summary>
        /// Get a list of rectangles of the faces.
        /// </summary>
        /// <seealso cref="getFaceRegions(Mat, string, string, int)"/>
        /// <param name="img">Source Image</param>
        /// <param name="ssdProtoFile">ssdProto File</param>
        /// <param name="ssdFile">ssd File</param>
        /// <param name="imgDim">Dim</param>
        /// <returns>Returns a list of rectangles </returns>
        public List<Rectangle> getFaceRegions(Bitmap img, String ssdProtoFile, String ssdFile, int imgDim = 300)
        {
            Mat Mimg = new Image<Bgr, Byte>(img).Mat;
            Emgu.CV.Dnn.Net net = DnnInvoke.ReadNetFromCaffe(ssdProtoFile, ssdFile);
            MCvScalar meanVal = new MCvScalar(104, 177, 123);
            Mat inputBlob = DnnInvoke.BlobFromImage(Mimg, 1.0, new Size(imgDim, imgDim), meanVal, false, false);
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
                    float xLeftBottom = values[3] * Mimg.Cols;
                    float yLeftBottom = values[4] * Mimg.Rows;
                    float xRightTop = values[5] * Mimg.Cols;
                    float yRightTop = values[6] * Mimg.Rows;
                    RectangleF objectRegion = new RectangleF(xLeftBottom, yLeftBottom, xRightTop - xLeftBottom, yRightTop - yLeftBottom);
                    Rectangle faceRegion = Rectangle.Round(objectRegion);
                    faceRegions.Add(faceRegion);

                } 
            }
            return faceRegions;
        }

        /// <summary>
        /// Get a list of the landmarks of the faces.
        /// </summary>
        /// <seealso cref="getFaceRegions(Mat, string, string, int)"/>
        /// <param name="img">Source image</param>
        /// <param name="faceRegions">List of tectangles of the faces. See <see cref="getFaceRegions(Mat, string, string, int)"/></param>
        /// <param name="facemarkFileName">Facemark File</param>
        /// <returns>Return a list of arrays of points of each face.</returns>
        public PointF[][][] getLandmarks(Bitmap img, List<Rectangle> faceRegions, String facemarkFileName)
        {
            Mat Mimg = new Image<Bgr, Byte>(img).Mat;
            List<VectorOfVectorOfPointF> landmarks = new List<VectorOfVectorOfPointF>(faceRegions.Count);
            for (int i = 0; i <= faceRegions.Count - 1; i++)
            {
                landmarks.Add(new VectorOfVectorOfPointF());
            }
            FacemarkLBFParams facemarkParam = new Emgu.CV.Face.FacemarkLBFParams();
            FacemarkLBF facemark = new Emgu.CV.Face.FacemarkLBF(facemarkParam);
            for (int j = 0; j <= faceRegions.Count - 1; j++)
            {
                Rectangle[] faces = { faceRegions[j] };
                VectorOfRect vr = new VectorOfRect(faces);
                try
                {
                    facemark.LoadModel(facemarkFileName);
                    facemark.Fit(Mimg, vr, landmarks[j]);
                }
                catch (Emgu.CV.Util.CvException ex)
                {
                    Console.WriteLine(ex.ToString());
                }

            }
            VectorOfVectorOfPointF[] list = landmarks.ToArray();
            List<PointF[][]> List2 = new List<PointF[][]>();
            for(int i = 0; i < list.Length; i++){
                List2.Add(list[i].ToArrayOfArray());
            }
            return List2.ToArray();
        }

        /// <summary>
        /// Blur faces inside an oval of the rectangle
        /// </summary>
        /// <param name="img">Source image</param>
        /// <param name="faceRegions">If Faces are given then it not recalculates them</param>
        /// <returns> Returns the image with the faces blurred inside the rectangle cropped to an oval</returns>
        public Bitmap BlurFaceOval(Bitmap img, List<Rectangle> faces = null, String ssdFile_ = null, String ssdProtoFile_ = null)
        {
            int imgDim = 300;
            MCvScalar meanVal = new MCvScalar(104, 177, 123);
            String ssdFile = (ssdFile_ == null ? this.ssdFile : ssdFile_);
            String ssdProtoFile = (ssdProtoFile_ == null ? this.ssdProtoFile : ssdProtoFile_);

            List<Rectangle> faceRegions = ((faces==null)?getFaceRegions(img, ssdProtoFile, ssdFile, imgDim):faces);
           
            int numFaces = faceRegions.Count;
            for (int j = 0; j <= faceRegions.Count - 1; j++)
            {
                if(DEBUG)
                    CvInvoke.Rectangle(new Image<Bgr, Byte>(img), faceRegions[j], new MCvScalar(0, 255, 0));
                Mat img2 = new Image<Bgr, Byte>(img).Mat;
                //Blur Oval
                img = new Image<Bgr, Byte>((Bitmap)blurer.BlurRectangle(img, faceRegions[j])).Bitmap;
                
                
            }
            return img;

        }


        /// <summary>
        /// Blur faces inside the landmarks
        /// </summary>
        /// <param name="img">Source image</param>
        /// <returns> Returns the image with the faces blurred inside the landmarks of each face</returns>
        public Bitmap BlurFaceWithLandmark(Bitmap img, int BlurSize, List<Rectangle> faces = null, PointF[][][] landmarks_ = null, Bitmap Mask = null, String ssdFile_ = null, String ssdProtoFile_ = null, String facemarkFileName_ = null)
        {
            
            int imgDim = 300;
            MCvScalar meanVal = new MCvScalar(104, 177, 123);
            String ssdFile = (ssdFile_ == null ? this.ssdFile : ssdFile_);
            String ssdProtoFile = (ssdProtoFile_ == null ? this.ssdProtoFile : ssdProtoFile_);
            String facemarkFileName = (facemarkFileName_ == null ? this.facemarkFileName : facemarkFileName_);

            List<Rectangle> faceRegions = ((faces == null) ? getFaceRegions(img, ssdProtoFile, ssdFile, imgDim) : faces);
            PointF[][][] landmarks__ = ((landmarks_ == null) ? getLandmarks(img, faceRegions, facemarkFileName) : landmarks_);

            List<VectorOfVectorOfPointF> landmarks = new List<VectorOfVectorOfPointF>(landmarks_.Length);
            foreach (PointF[][] lm in landmarks__)
            {
                landmarks.Add(new VectorOfVectorOfPointF(lm));
            }

            for (int j = 0; j <= faceRegions.Count - 1; j++)
            {
                if(DEBUG)
                    CvInvoke.Rectangle(new Image<Bgr, Byte>(img), faceRegions[j], new MCvScalar(0, 255, 0));
                if(DEBUG)
                    for (int i = 0; i < landmarks[i].Size; i++)
                    {
                        using (VectorOfPointF vpf = landmarks[j][i])
                            try
                            {
                                FaceInvoke.DrawFacemarks(new Image<Bgr, Byte>(img), vpf, new MCvScalar(255, 0, 0));
                            }
                            catch (Emgu.CV.Util.CvException ex)
                            {
                                Console.WriteLine(ex.ToString());
                            }
                    }
                PointF[] Facepoints = landmarks[j].ToArrayOfArray()[0];
                if(DEBUG)
                    try
                    {
                        CvInvoke.DrawContours(new Image<Bgr, Byte>(img), new VectorOfPointF(Facepoints), 2, new MCvScalar(255,0,0));
                    }
                    catch (Emgu.CV.Util.CvException ex)
                    {
                        Console.WriteLine(ex.ToString());
                    }
                // Blur Path
                img = new Image<Bgr, Byte>((Bitmap)blurer.BlurPath(img, BlurSize, Facepoints, faceRegions[j], Mask)).Bitmap;
            }
            return img;

        }


        public Bitmap getOpMask(Bitmap img, List<Rectangle> faces = null, PointF[][][] landmarks_ = null, String ssdFile_ = null, String ssdProtoFile_ = null, String facemarkFileName_ = null)
        {
            int imgDim = 300;
            MCvScalar meanVal = new MCvScalar(104, 177, 123);
            String ssdFile = (ssdFile_ == null ? this.ssdFile : ssdFile_);
            String ssdProtoFile = (ssdProtoFile_ == null ? this.ssdProtoFile : ssdProtoFile_);
            String facemarkFileName = (facemarkFileName_ == null ? this.facemarkFileName : facemarkFileName_);

            List<Rectangle> faceRegions = ((faces == null) ? getFaceRegions(img, ssdProtoFile, ssdFile, imgDim) : faces);
            PointF[][][] landmarks__ = ((landmarks_ == null) ? getLandmarks(img, faceRegions, facemarkFileName) : landmarks_);

            List<VectorOfVectorOfPointF> landmarks = new List<VectorOfVectorOfPointF>(landmarks_.Length);
            foreach (PointF[][] lm in landmarks__)
            {
                landmarks.Add(new VectorOfVectorOfPointF(lm));
            }

            Bitmap dstImage = new Bitmap(img.Width, img.Height, PixelFormat.Format32bppArgb);
            for (int j = 0; j <= faceRegions.Count - 1; j++)
            {
                if (DEBUG)
                    CvInvoke.Rectangle(new Image<Bgr, Byte>(img), faceRegions[j], new MCvScalar(0, 255, 0));
                if (DEBUG)
                    for (int i = 0; i < landmarks[i].Size; i++)
                    {
                        using (VectorOfPointF vpf = landmarks[j][i])
                            try
                            {
                                FaceInvoke.DrawFacemarks(new Image<Bgr, Byte>(img), vpf, new MCvScalar(255, 0, 0));
                            }
                            catch (Emgu.CV.Util.CvException ex)
                            {
                                Console.WriteLine(ex.ToString());
                            }
                    }
                PointF[] Facepoints = landmarks[j].ToArrayOfArray()[0];
                if (DEBUG)
                    CvInvoke.DrawContours(new Image<Bgr, Byte>(img), new VectorOfPointF(Facepoints), 2, new MCvScalar(255, 0, 0));
                dstImage = blurer.OpMask(dstImage, Facepoints, faceRegions[j]);
            }
            return dstImage;

        }

    }
}
