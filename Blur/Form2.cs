using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.UI;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using System.Drawing.Imaging;
using Blur;
using Emgu.CV.Util;
using System.Runtime.InteropServices;
using Emgu.CV.Dnn;
using System.IO;
using System.Diagnostics;


using XBF;


namespace Blur
{
    public partial class Form2 : Form
    {
        private VideoCapture _capture;
        //private CascadeClassifier _cascadeClassifier;


        public Form2()
        {
            InitializeComponent();
            _capture = new VideoCapture();

            //pictureBox1.Image = _capture.QueryFrame().ToImage<Bgr, Byte>().Bitmap;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            //_cascadeClassifier = new CascadeClassifier(@"Assets/haarcascade_frontalface_default.xml");
            if (openFileDialog1.ShowDialog() == DialogResult.OK) {
                string path = openFileDialog1.FileName;

                Bitmap frame = (Bitmap)Image.FromFile(path);
                Image<Bgr, Byte> imageFrame = new Image<Bgr, Byte>(frame);

                if (imageFrame != null)
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
                    Analyzer analyzer = new Analyzer(true, ssdFile : Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets\res10_300x300_ssd_iter_140000.caffemodel", ssdProtoFile : Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets\deploy.prototxt", facemarkFileName : Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets\lbfmodel.yaml");
                    frame = analyzer.BlurFaceOval(imageFrame.Mat).ToImage<Bgr, Byte>().Bitmap;
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
                Image<Bgr, Byte> imageFrame = new Image<Bgr, Byte>(frame);

                Analyzer analyzer = new Analyzer(false,  Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets\res10_300x300_ssd_iter_140000.caffemodel",  Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets\deploy.prototxt",  Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets\lbfmodel.yaml");

                
                pictureBox1.Image = analyzer.BlurFaceWithLandmark(imageFrame.Mat, 20).ToImage<Bgr, Byte>().Bitmap;

            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                string path = openFileDialog1.FileName;
                Image<Bgr, Byte> imageFrame = new Image<Bgr, Byte>(Image.FromFile(path) as Bitmap);


                Analyzer analyzer = new Analyzer(false, Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets\res10_300x300_ssd_iter_140000.caffemodel", Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets\deploy.prototxt", Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets\lbfmodel.yaml");
                
                List<Rectangle> Faces = analyzer.getFaceRegions(imageFrame.Mat, Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets\deploy.prototxt", Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets\res10_300x300_ssd_iter_140000.caffemodel");
                List<VectorOfVectorOfPointF> Landmarks = analyzer.getLandmarks(imageFrame.Mat, Faces, Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location) + @"\Assets\lbfmodel.yaml");

                Bitmap Mask = analyzer.getOpMask(imageFrame.Mat, Faces, Landmarks);
                Mat Final = analyzer.BlurFaceWithLandmark(imageFrame.Mat, 12, Faces, Landmarks, Mask);

           


                pictureBox1.Image = Final.ToImage<Bgr, Byte>().Bitmap;

            }
        }
    }
}
