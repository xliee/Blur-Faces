using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Linq;


namespace XBF
{
    class blur
    {
        public Image ClipToCircle(Image srcImage, Image blurImage, PointF center, float radius, Rectangle box)
        {
            Image dstImage = new Bitmap(srcImage.Width, srcImage.Height, srcImage.PixelFormat);
            using (Graphics g = Graphics.FromImage(dstImage))
            {
                RectangleF r = new RectangleF(center.X - radius, center.Y - radius,radius * 2, radius * 2);
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.DrawImage(srcImage, 0, 0);
                GraphicsPath path = new GraphicsPath();
                path.AddEllipse(box);
                g.SetClip(path);
                g.DrawImage(blurImage, 0, 0);
                return dstImage;
            }
        }
        public Image BlurRectangle(Image srcImage, Rectangle face)
        {
            Bitmap firstb = FastBoxBlur(srcImage, 20, face);
            Image dstImage = new Bitmap(srcImage.Width, srcImage.Height, srcImage.PixelFormat);
            using (Graphics g = Graphics.FromImage(dstImage))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.DrawImage(srcImage, 0, 0);
                GraphicsPath Gpath = new GraphicsPath();
                Gpath.AddEllipse(face);
                g.SetClip(Gpath);
                g.DrawImage(firstb, 0, 0);
                return dstImage;
            }
        }
        public Image BlurPath(Image srcImage, int blursize, PointF[] path, Rectangle face, Bitmap Mask)
        {
            Bitmap firstb = new Bitmap(srcImage);
            if (Mask == null)
            {
                firstb = FastBoxBlur(srcImage, blursize, face);
            }
            else
            {
                GaussianBlur blurer = new GaussianBlur(srcImage as Bitmap);
                firstb =  blurer.Process(blursize);
            }
            Image dstImage = new Bitmap(srcImage.Width, srcImage.Height, PixelFormat.Format32bppArgb);

            using (Graphics g = Graphics.FromImage(dstImage))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.DrawImage(srcImage, 0, 0);
                if (Mask == null)
                {
                    GraphicsPath gPath = new GraphicsPath();
                    if (path.Length > 1)
                    {
                        QuickHull QH = new QuickHull();
                        List<PointF> fPoints = path.ToList();
                        fPoints.Add(new PointF(face.X + face.Width, face.Y + face.Height / 2));
                        fPoints.Add(new PointF(face.X + face.Width / 2, face.Y));
                        fPoints.Add(new PointF(face.X, face.Y + face.Height / 2));
                        PointF[] Hull = QH.Run(fPoints).ToArray();
                        List<PointF> pathS = new List<PointF>();
                        for (int i = 0; i < Hull.Length; i++)
                        {
                            pathS.Add(scaleP(Hull[i], centerP(path), 1.5f));
                        }
                        gPath.AddPolygon(pathS.ToArray());
                        g.SetClip(gPath);
                        
                    }
                    else
                    {
                        gPath.AddEllipse(face);
                        g.SetClip(gPath);
                    }
                    g.DrawImage(firstb, 0, 0);
                }
                else
                    g.DrawImage(applyOpMask(firstb, Mask), 0, 0);
            }
            return dstImage;
        }


        public Bitmap OpMask(Bitmap srcImage, PointF[] path, Rectangle face)
        {
            using (Graphics g = Graphics.FromImage(srcImage))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                if (path.Length > 1)
                {
                    QuickHull QH = new QuickHull();
                    List<PointF> fPoints = path.ToList();
                    fPoints[fPoints.Count - 1] = (new PointF(face.X + face.Width, face.Y + face.Height / 2));

                    double c = Math.Atan2(addP(new PointF(face.X + face.Width, face.Y + face.Height / 2), new PointF(face.X + face.Width / 2, face.Y)).Y, addP(new PointF(face.X + face.Width, face.Y + face.Height / 2), new PointF(face.X + face.Width / 2, face.Y)).X);
                    double x_mid = (face.X + face.Width - centerP(path).X) * Math.Cos(c) + face.X;
                    double y_mid = (face.X + face.Width - centerP(path).X) * Math.Sin(c) + face.Y;
                    fPoints.Add(new PointF((float)x_mid, (float)y_mid));

                    fPoints.Add(new PointF(face.X + face.Width / 2, face.Y));

                    c = Math.Atan2(addP(new PointF(face.X + face.Width / 2, face.Y), new PointF(face.X, face.Y + face.Height / 2)).Y, addP(new PointF(face.X + face.Width / 2, face.Y), new PointF(face.X, face.Y + face.Height / 2)).X);
                    x_mid = (face.X + face.Width - centerP(path).X) * Math.Cos(c) + face.X + face.Width / 2;
                    y_mid = (face.X + face.Width - centerP(path).X) * Math.Sin(c) + face.Y + face.Height / 2;
                    fPoints.Add(new PointF((float)x_mid, (float)y_mid));

                    fPoints.Add(new PointF(face.X, face.Y + face.Height / 2));
                    PointF[] Hull = QH.Run(fPoints).ToArray();
                    PointF[] pathS = new PointF[Hull.Length];
                    for (int i = 0; i < Hull.Length; i++)
                    {
                        pathS[i] = scaleP(Hull[i], centerP(Hull), 1.5f);
                    }
                    GraphicsPath gPath = new GraphicsPath();
                    gPath.AddPolygon(pathS);
                    g.SetClip(gPath);
                    PathGradientBrush bush = new PathGradientBrush(gPath);
                    bush.CenterPoint = centerP(pathS);
                    bush.CenterColor = Color.FromArgb(255, Color.FromArgb(0, 0, 255));
                    Blend blnd = new Blend();
                    blnd.Positions = new float[] { 0f, .15f, .25f, .5f, .75f, 1f };
                    blnd.Factors = new float[] { .05f, .5f, .95f, 1f, 1f, 1f };
                    bush.Blend = blnd;
                    bush.SurroundColors = new Color[] { Color.FromArgb(255, Color.FromArgb(255, 255, 0)) };
                    g.FillPolygon(bush, pathS);
                }
                else
                {
                    GraphicsPath gPath = new GraphicsPath();
                    gPath.AddEllipse(face);
                    g.SetClip(gPath);
                    PathGradientBrush bush = new PathGradientBrush(gPath);
                    bush.CenterPoint = new PointF(face.X+face.Width/2, face.Y+face.Height/2);
                    bush.CenterColor = Color.FromArgb(255, Color.FromArgb(0, 0, 255));
                    Blend blnd = new Blend();
                    blnd.Positions = new float[] { 0f, .15f, .25f, .5f, .75f, 1f };
                    blnd.Factors = new float[] { .05f, .5f, .95f, 1f, 1f, 1f };
                    bush.Blend = blnd;
                    bush.SurroundColors = new Color[] { Color.FromArgb(255, Color.FromArgb(255, 255, 0)) };
                    g.FillEllipse(bush, face);
                }
                return srcImage;
            }
        }
        public Bitmap applyOpMask(Bitmap input, Bitmap mask)
        {

            Bitmap output = new Bitmap(input.Width, input.Height, PixelFormat.Format32bppArgb);
            var rect = new Rectangle(0, 0, input.Width, input.Height);
            var bitsMask = mask.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            var bitsInput = input.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            var bitsOutput = output.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            unsafe
            {
                for (int y = 0; y < input.Height; y++)
                {
                    byte* ptrMask = (byte*)bitsMask.Scan0 + y * bitsMask.Stride;
                    byte* ptrInput = (byte*)bitsInput.Scan0 + y * bitsInput.Stride;
                    byte* ptrOutput = (byte*)bitsOutput.Scan0 + y * bitsOutput.Stride;
                    for (int x = 0; x < input.Width; x++)
                    {
                        ptrOutput[4 * x] = ptrInput[4 * x];           // blue
                        ptrOutput[4 * x + 1] = ptrInput[4 * x + 1];   // green
                        ptrOutput[4 * x + 2] = ptrInput[4 * x + 2];   // red
                        ptrOutput[4 * x + 3] = ptrMask[4 * x];        // alpha
                    }
                }
            }
            mask.UnlockBits(bitsMask);
            input.UnlockBits(bitsInput);
            output.UnlockBits(bitsOutput);

            return output;
        }


        private PointF centerP(PointF[] Points)
        {
            float[] X = new float[Points.Length];
            float[] Y = new float[Points.Length];
            for (int i = 0; i < Points.Length; i++)
            {
                X[i] = Points[i].X;
                Y[i] = Points[i].Y;
            }
            float sumX = 0;
            float sumY = 0;
            for (int i = 0; i < X.Length; i++)
                sumX += X[i];
            for (int i = 0; i < Y.Length; i++)
                sumY += Y[i];

            float resultX = sumX / X.Length;
            float resultY = sumY / Y.Length;

            return new PointF(resultX, resultY);
        }
        private PointF scaleP(PointF Point, PointF Center, float Scale)
        {
            PointF v2 = substractP(Point, Center); // get a vector to v relative to the centerpoint
            PointF v2_scaled = multiplyP(v2, Scale); // scale the cp-relative-vector
            return addP(v2_scaled, Center);
        }
        private PointF substractP(PointF p1, PointF p2)
        {
            return new PointF(p1.X-p2.X, p1.Y-p2.Y);
        }
        private PointF addP(PointF p1, PointF p2)
        {
            return new PointF(p1.X + p2.X, p1.Y + p2.Y);
        }
        private PointF multiplyP(PointF p1, float Scale)
        {
            return new PointF(Scale * p1.X, Scale * p1.Y);
        }
        private float distanceP(PointF p1, PointF p2)
        {
            return (float)Math.Sqrt(Math.Pow(p1.X+ p2.X, 2) + Math.Pow(p1.Y + p2.Y, 2));
        }

     
        private Bitmap Convolve(Bitmap input, float[,] filter, Rectangle box)
        {
            //Find center of filter
            int xMiddle = (int)Math.Floor(filter.GetLength(0) / 2.0);
            int yMiddle = (int)Math.Floor(filter.GetLength(1) / 2.0);

            //Create new image
            Bitmap output = new Bitmap((Image)input);
            
            FastBitmap reader = new FastBitmap(input);
            FastBitmap writer = new FastBitmap(output);
            reader.LockImage();
            writer.LockImage();

            for (int x = box.X; x < box.X + box.Width; x++)
            {
                for (int y = box.Y; y < box.Y + box.Height; y++)
                {
                    float r = 0;
                    float g = 0;
                    float b = 0;
                    //Apply filter
                    for (int xFilter = 0; xFilter < filter.GetLength(0); xFilter++)
                    {
                        for (int yFilter = 0; yFilter < filter.GetLength(1); yFilter++)
                        {

                            //distance = Math.Pow(distance, distance/10)
                            int x0 = x - xMiddle + xFilter;
                            int y0 = y - yMiddle + yFilter;

                            //Only if in bounds
                            if (x0 >= 0 && x0 < input.Width &&
                                y0 >= 0 && y0 < input.Height)
                            {
                                Color clr = reader.GetPixel(x0, y0);

                                r += clr.R * (filter[xFilter, yFilter]);
                                g += clr.G * (filter[xFilter, yFilter]);
                                b += clr.B * (filter[xFilter, yFilter]);
                            }
                        }
                    }

                    //Normalize (basic)
                    if (r > 255)
                        r = 255;
                    if (g > 255)
                        g = 255;
                    if (b > 255)
                        b = 255;

                    if (r < 0)
                        r = 0;
                    if (g < 0)
                        g = 0;
                    if (b < 0)
                        b = 0;

                    //Set the pixel
                    writer.SetPixel(x, y, Color.FromArgb((int)r, (int)g, (int)b));
                }
            }

            reader.UnlockImage();
            writer.UnlockImage();

            return output;
        }
   
        /// <summary>
        /// Returns a box filter 1D kernel that is in the format {1,..,n}
        /// </summary>
        private float[,] GetHorizontalFilter(int size)
        {
            float[,] smallFilter = new float[size, 1];
            float constant = size;
            for (int i = 0; i < size; i++)
            {
                smallFilter[i, 0] = 1.0f / constant;
            }



            return smallFilter;
        }

        /// <summary>
        /// Returns a box filter 1D kernel that is in the format {1},...,{n}
        /// </summary>
        private float[,] GetVerticalFilter(int size)
        {
            
            float[,] smallFilter = new float[1, size];
            float constant = size;
            for (int i = 0; i < size; i++)
            {

                smallFilter[0, i] = 1.0f / constant;
            }



            return smallFilter;
        }

        /// <summary>
        /// Returns a box filter 2D kernel in the format {1,...,n},...,{1,...,n}
        /// </summary>
        private float[,] GetBoxFilter(int size)
        {
            float[,] filter = new float[size, size];
            float constant = size * size;

            for (int i = 0; i < filter.GetLength(0); i++)
            {
                for (int j = 0; j < filter.GetLength(1); j++)
                {
                    filter[i, j] = 1.0f / constant;
                }
            }

            return filter;
        }

        public Bitmap BoxBlur(Image img, int size, Rectangle box)
        {
            //Apply a box filter by convolving the image with a 2D kernel
            return Convolve(new Bitmap(img), GetBoxFilter(size), box);
        }

        public Bitmap FastBoxBlur(Image img, int size, Rectangle box)
        {
            //Apply a box filter by convolving the image with two separate 1D kernels (faster)
            return Convolve(Convolve(new Bitmap(img), GetHorizontalFilter(size), box), GetVerticalFilter(size), box);
        }

    }
}
