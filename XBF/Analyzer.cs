using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;

namespace XBF
{
    public interface IAnalyzer
    {
        /// <summary>
        /// Get a list of rectangles of the faces.
        /// </summary>
        /// <seealso cref="getFaceRegions(Mat, string, string, int)"/>
        /// <param name="img">Source Image</param>
        /// <param name="ssdProtoFile">ssdProto File</param>
        /// <param name="ssdFile">ssd File</param>
        /// <param name="imgDim">Dim</param>
        /// <returns>Returns a list of rectangles </returns>
        List<Rectangle> getFaceRegions(byte[] img, byte[] ssdProtoFile, byte[] ssdFile);

        /// <summary>
        /// Get a list of the landmarks of the faces.
        /// </summary>
        /// <seealso cref="getFaceRegions(Mat, string, string, int)"/>
        /// <param name="img">Source image</param>
        /// <param name="faceRegions">List of tectangles of the faces. See <see cref="getFaceRegions(Mat, string, string, int)"/></param>
        /// <param name="facemarkFileName">Facemark File</param>
        /// <returns>Return a list of arrays of points of each face.</returns>
        PointF[][][] getLandmarks(byte[] image, List<Rectangle> faceRegions, String facemarkFileName);

        /// <summary>
        /// Get Blur Opacity mask 
        /// </summary>
        /// <param name="img">Base Image</param>
        /// <param name="faces">Faces Matrix</param>
        /// <param name="landmarks_"></param>
        /// <param name="ssdFile_"></param>
        /// <param name="ssdProtoFile_"></param>
        /// <param name="facemarkFileName_"></param>
        /// <returns></returns>
        Bitmap getOpMask(byte[] image, List<Rectangle> faces, PointF[][][] landmarks_);
     
        /// <summary>
        /// Blur faces inside the landmarks
        /// </summary>
        /// <param name="img">Source image</param>
        /// <returns> Returns the image with the faces blurred inside the landmarks of each face</returns>
        Bitmap BlurFaceWithLandmark(byte[] image, int BlurSize, List<Rectangle> faces, PointF[][][] landmarks_, Bitmap Mask);
   
        /// <summary>
        /// Blur faces inside an oval of the rectangle
        /// </summary>
        /// <param name="img">Source image</param>
        /// <param name="faceRegions">If Faces are given then it not recalculates them</param>
        /// <returns> Returns the image with the faces blurred inside the rectangle cropped to an oval</returns>
        Bitmap BlurFaceOval(byte[] image, List<Rectangle> faces);
        
    }
    public class Analyzer : IAnalyzer
    {
        private bool DEBUG = false;
        private blur blurer;
        public byte[] ssdFile;
        public byte[] ssdProtoFile;
        public String facemarkFileName;

        public Analyzer(bool Debug = false)
        {

            var assembly = Assembly.GetExecutingAssembly();
            var resourceName = assembly.GetManifestResourceNames().Single(str => str.EndsWith("res10_300x300_ssd_iter_140000.caffemodel"));

            using (Stream stream = assembly.GetManifestResourceStream(resourceName))
            using (BinaryReader br = new BinaryReader(stream))
            {
                ssdFile = br.ReadBytes((int)stream.Length);
            }

            resourceName = assembly.GetManifestResourceNames().Single(str => str.EndsWith("deploy.prototxt"));
            using (Stream stream = assembly.GetManifestResourceStream(resourceName))
            using (BinaryReader br = new BinaryReader(stream))
            {
                ssdProtoFile = br.ReadBytes((int)stream.Length);
            }



            this.DEBUG = Debug;
            try
            {
                this.facemarkFileName = Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets\lbfmodel.yaml";
                System.IO.Directory.CreateDirectory(Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets");
                resourceName = assembly.GetManifestResourceNames().Single(str => str.EndsWith("lbfmodel.yaml"));
                using (Stream stream = assembly.GetManifestResourceStream(resourceName))
                using (var fileStream = new FileStream(this.facemarkFileName, FileMode.OpenOrCreate, FileAccess.Write))
                {
                    stream.CopyTo(fileStream);
                }
            }
            catch(Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            
            blurer = new blur();
        }

        public static byte[] ImageToByte(Image img)
        {
            return (byte[])(new ImageConverter()).ConvertTo(img, typeof(byte[]));
        }
        
        List<Rectangle> IAnalyzer.getFaceRegions(byte[] img, byte[] ssdProtoFile, byte[] ssdFile)
        {
            //Assembly.GetExecutingAssembly().GetManifestResourceInfo("").ResourceLocation.ToString();
            Bitmap image = (Bitmap)((new ImageConverter()).ConvertFrom(img));
            Mat Mimg = new Image<Bgr, Byte>(image).Mat;
            Emgu.CV.Dnn.Net net = DnnInvoke.ReadNetFromCaffe(ssdProtoFile, ssdFile);
            MCvScalar meanVal = new MCvScalar(104, 177, 123);
            Mat inputBlob = DnnInvoke.BlobFromImage(Mimg, 1.0, new Size(300, 300), meanVal, true, false);
            net.SetInput(inputBlob, "data");
            Mat detection = net.Forward("detection_out");

            float confidenceThreshold = 0.5f;

            List<Rectangle> faceRegions = new List<Rectangle>();

            int[] dim = detection.SizeOfDimension;
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
        
        PointF[][][] IAnalyzer.getLandmarks(byte[] image, List<Rectangle> faceRegions, String facemarkFileName)
        {
            Bitmap img = (Bitmap)((new ImageConverter()).ConvertFrom(image));

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

        Bitmap IAnalyzer.getOpMask(byte[] image, List<Rectangle> faces, PointF[][][] landmarks_)
        {
            Bitmap img = (Bitmap)((new ImageConverter()).ConvertFrom(image));
            MCvScalar meanVal = new MCvScalar(104, 177, 123);


            List<VectorOfVectorOfPointF> landmarks = new List<VectorOfVectorOfPointF>(landmarks_.Length);
            foreach (PointF[][] lm in landmarks_)
            {
                landmarks.Add(new VectorOfVectorOfPointF(lm));
            }

            Bitmap dstImage = new Bitmap(img.Width, img.Height, PixelFormat.Format32bppArgb);
            for (int j = 0; j <= faces.Count - 1; j++)
            {
                if (DEBUG)
                    CvInvoke.Rectangle(new Image<Bgr, Byte>(img), faces[j], new MCvScalar(0, 255, 0));
                if (DEBUG)
                    for (int i = 0; i < landmarks[j].Size; i++)
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
                //if (DEBUG)
                // CvInvoke.DrawContours(new Image<Bgr, Byte>(img), new VectorOfPointF(Facepoints), 2, new MCvScalar(255, 0, 0));
                dstImage = blurer.OpMask(dstImage, Facepoints, faces[j]);
            }
            return dstImage;

        }
        
        Bitmap IAnalyzer.BlurFaceWithLandmark(byte[] image, int BlurSize, List<Rectangle> faces, PointF[][][] landmarks_, Bitmap Mask)
        {
            Bitmap img = (Bitmap)((new ImageConverter()).ConvertFrom(image));
            MCvScalar meanVal = new MCvScalar(104, 177, 123);
            
            List<VectorOfVectorOfPointF> landmarks = new List<VectorOfVectorOfPointF>(landmarks_.Length);
            foreach (PointF[][] lm in landmarks_)
            {
                landmarks.Add(new VectorOfVectorOfPointF(lm));
            }

            for (int j = 0; j <= faces.Count - 1; j++)
            {
                if(DEBUG)
                    CvInvoke.Rectangle(new Image<Bgr, Byte>(img), faces[j], new MCvScalar(0, 255, 0));
                if(DEBUG)
                    for (int i = 0; i < landmarks[j].Size; i++)
                    {
                        using (VectorOfPointF vpf = landmarks[j][i])
                            try
                            {
                                FaceInvoke.DrawFacemarks(new Image<Bgr, Byte>(img), vpf, new MCvScalar(255, 128, 0));
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
                img = new Image<Bgr, Byte>((Bitmap)blurer.BlurPath(img, BlurSize, Facepoints, faces[j], Mask)).Bitmap;
            }
            return img;

        }

        Bitmap IAnalyzer.BlurFaceOval(byte[] image, List<Rectangle> faces)
        {
            Bitmap img = (Bitmap)((new ImageConverter()).ConvertFrom(image));
            
            MCvScalar meanVal = new MCvScalar(104, 177, 123);
            
           
            int numFaces = faces.Count;
            for (int j = 0; j <= faces.Count - 1; j++)
            {
                if(DEBUG)
                    CvInvoke.Rectangle(new Image<Bgr, Byte>(img), faces[j], new MCvScalar(0, 255, 0));
                Mat img2 = new Image<Bgr, Byte>(img).Mat;
                //Blur Oval
                img = new Image<Bgr, Byte>((Bitmap)blurer.BlurRectangle(img, faces[j])).Bitmap;
                
                
            }
            return img;

        }
        
    }
}
