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
namespace Blur
{
    public partial class Form1 : Form
    {
        private VideoCapture _capture;
        private CascadeClassifier _cascadeClassifier;


        

        public Form1()
        {
            InitializeComponent();

            _capture = new VideoCapture();

            imgCamUser.Image = _capture.QueryFrame().ToImage<Bgr, Byte>().Bitmap;

           
        }

        private void button1_Click(object sender, EventArgs e)
        {
            _cascadeClassifier = new CascadeClassifier(@"W:\Documents\Web\c++\face_detect_n_track-master\haarcascade_frontalface_default.xml");
            using (var imageFrame = _capture.QueryFrame().ToImage<Bgr, Byte>())
            {
                Bitmap frame =imageFrame.ToBitmap();
                if (imageFrame != null)
                {

                    var grayframe = imageFrame.Convert<Gray, Byte>();
                    var faces = _cascadeClassifier.DetectMultiScale(grayframe, 1.1, 3, Size.Empty); //the actual face detection happens here
                    foreach (var face in faces)
                    {
                        blur blurer = new blur();
                        Image<Bgr, Byte> frame2 = new Image<Bgr, Byte>(frame);


                        //frame2.Draw(face, new Bgr(Color.BurlyWood), 2); //the detected face(s) is highlighted here using a box that is drawn around it/them

                        frame = blurer.FastBoxBlur(frame2.Bitmap, 20, new Rectangle() { X = face.X, Y = face.Y, Width = face.Width, Height = face.Height });
                        frame = (Bitmap)blurer.ClipToCircle(frame2.Bitmap, frame, new Point(face.X + face.Width / 2, face.Y + face.Height / 2), face.Width / 2, new Rectangle() { X = face.X, Y = face.Y, Width = face.Width, Height = face.Height });
                        
                    }
                }
                imgCamUser.Image = frame;

            }
        }
    }


}
