using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing.Imaging;
using System.IO;
using System.Diagnostics;


using XBF;
using Emgu.CV.Face;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace Blur
{
    public partial class Form2 : Form
    {
        //private CascadeClassifier _cascadeClassifier;


        public Form2()
        {
            InitializeComponent();

            //pictureBox1.Image = _capture.QueryFrame().ToImage<Bgr, Byte>().Bitmap;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            //_cascadeClassifier = new CascadeClassifier(@"Assets/haarcascade_frontalface_default.xml");
            if (openFileDialog1.ShowDialog() == DialogResult.OK) {
                string path = openFileDialog1.FileName;

                Bitmap frame = (Bitmap)Image.FromFile(path);
                //Image<Bgr, Byte> imageFrame = new Image<Bgr, Byte>(frame);

                if (frame != null)
                {
                    //List<Rectangle> faces = new List<Rectangle>();
                    //var grayframe = imageFrame.Convert<Gray, Byte>();
                    ////var faces = _cascadeClassifier.DetectMultiScale(grayframe, 1.1, 6, Size.Empty); //the actual face detection happens here

                    //using (UMat ugray = new UMat())
                    //{
                    //    CvInvoke.CvtColor(imageFrame, ugray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);

                    //    //normalizes brightness and increases contrast of the image
                    //    CvInvoke.EqualizeHist(ugray, ugray);

                    //    //Detect the faces  from the gray scale image and store the locations as rectangle                   
                    //    Rectangle[] facesDetected = _cascadeClassifier.DetectMultiScale(
                    //       ugray, 1.1, 3, new Size(20, 20));

                    //    faces.AddRange(facesDetected);
                    //}

                    ///////////////////////////////////
                    //String ssdFile = "res10_300x300_ssd_iter_140000.caffemodel";
                    //String ssdProtoFile = "deploy.prototxt";
                    //Detector detector = new Detector();
                    //List<Rectangle> faceRegions = detector.getFaceRegions(imageFrame.Mat, ssdProtoFile, ssdFile);

                    //foreach (var face in faceRegions)
                    //{
                    //    blur blurer = new blur();
                    //    Image<Bgr, Byte> frame2 = new Image<Bgr, Byte>(frame);


                    //    //frame2.Draw(face, new Bgr(Color.BurlyWood), 2); //the detected face(s) is highlighted here using a box that is drawn around it/them




                    //    frame = blurer.FastBoxBlur(frame2.Bitmap, 20, face);
                    //    frame = (Bitmap)blurer.ClipToCircle(frame2.Bitmap, frame, new Point(face.X + face.Width / 2, face.Y + face.Height / 2), face.Width / 2, face);

                    //}
                    Analyzer analyzer = new Analyzer(true);
                    //frame = analyzer.BlurFaceOval(imageFrame.Mat).ToImage<Bgr, Byte>().Bitmap;
                    pictureBox1.Image = frame;
                }
                
                
            }
        }



       
        private void button2_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                string path = openFileDialog1.FileName;

                Bitmap frame = (Bitmap)Image.FromFile(path);

                Analyzer analyzer = new Analyzer(false);

                List<Rectangle> Faces = ((IAnalyzer)analyzer).getFaceRegions(Analyzer.ImageToByte(frame), analyzer.ssdProtoFile, analyzer.ssdFile);
                PointF[][][] Landmarks_ = ((IAnalyzer)analyzer).getLandmarks(Analyzer.ImageToByte(frame), Faces, analyzer.facemarkFileName);

                //List<VectorOfVectorOfPointF> landmarks = new List<VectorOfVectorOfPointF>(Landmarks_.Length);
                //foreach (PointF[][] lm in Landmarks_)
                //{
                //    landmarks.Add(new VectorOfVectorOfPointF(lm));
                //}
                Image<Bgr, Byte> img = new Image<Bgr, Byte>(frame);
                //for (int i = 0; i < landmarks[0].Size; i++)
                //{
                //    using (VectorOfPointF vpf = landmarks[0][i])
                //        FaceInvoke.DrawFacemarks(img, vpf, new MCvScalar(255, 128, 0));
                //}



                pictureBox1.Image = img.Bitmap;

            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                string path = openFileDialog1.FileName;
                Bitmap imageFrame = Image.FromFile(path) as Bitmap;



                Analyzer analyzer = new Analyzer(false);
                
                List<Rectangle> Faces = ((IAnalyzer)analyzer).getFaceRegions(Analyzer.ImageToByte(imageFrame), analyzer.ssdProtoFile, analyzer.ssdFile);
                PointF[][][] Landmarks = ((IAnalyzer)analyzer).getLandmarks(Analyzer.ImageToByte(imageFrame), Faces, analyzer.facemarkFileName);

                Bitmap Mask = ((IAnalyzer)analyzer).getOpMask(Analyzer.ImageToByte(imageFrame), Faces, Landmarks);
                Bitmap Final = ((IAnalyzer)analyzer).BlurFaceWithLandmark(Analyzer.ImageToByte(imageFrame), 12, Faces, Landmarks, Mask);

                //List<VectorOfVectorOfPointF> landmarks = new List<VectorOfVectorOfPointF>(Landmarks.Length);
                //foreach (PointF[][] lm in Landmarks)
                //{
                //    landmarks.Add(new VectorOfVectorOfPointF(lm));
                //}
                Image<Bgr, Byte> img = new Image<Bgr, Byte>(Final);
                //for(int j = 0; j < Faces.Count; j++)
                //{
                //    for (int i = 0; i < landmarks[j].Size; i++)
                //    {
                //        using (VectorOfPointF vpf = landmarks[j][i])
                //            FaceInvoke.DrawFacemarks(img, vpf, new MCvScalar(255, 128, 0));
                //    }
                //}



                pictureBox1.Image = img.Bitmap;

            }
        }

        private void button4_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                string path = openFileDialog1.FileName;
                Bitmap imageFrame = Image.FromFile(path) as Bitmap;

                Analyzer analyzer = new Analyzer(false);

                List<Rectangle> Faces = ((IAnalyzer)analyzer).getFaceRegions(Analyzer.ImageToByte(imageFrame), analyzer.ssdProtoFile, analyzer.ssdFile);
                PointF[][][] Landmarks = ((IAnalyzer)analyzer).getLandmarks(Analyzer.ImageToByte(imageFrame), Faces, analyzer.facemarkFileName);

                pictureBox1.Image = ((IAnalyzer)analyzer).getOpMask(Analyzer.ImageToByte(imageFrame), Faces, Landmarks);
            }
        }
    }
}
