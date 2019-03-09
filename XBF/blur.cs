using Emgu.CV;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
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
                RectangleF r = new RectangleF(center.X - radius, center.Y - radius,
                                                         radius * 2, radius * 2);

                // enables smoothing of the edge of the circle (less pixelated)
                g.SmoothingMode = SmoothingMode.AntiAlias;

                // fills background color
                //using (Brush br = new SolidBrush(backGround))
                //{
                //    g.FillRectangle(br, 0, 0, dstImage.Width, dstImage.Height);
                //}

                g.DrawImage(srcImage, 0, 0);
                // adds the new ellipse & draws the image again 
                GraphicsPath path = new GraphicsPath();

                //path.AddEllipse(r);
                path.AddEllipse(box);
                g.SetClip(path);
                g.DrawImage(blurImage, 0, 0);

                return dstImage;
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="srcImage"></param>
        /// <param name="path"></param>
        /// <param name="face"></param>
        /// <returns></returns>
        public Image BlurPath(Image srcImage, PointF[] path, Rectangle face)
        {
            Bitmap firstb = FastBoxBlur(srcImage, 20, face);
            Image bluredImage = new Bitmap(srcImage.Width, srcImage.Height, srcImage.PixelFormat);

            using (Graphics g = Graphics.FromImage(bluredImage))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.DrawImage(srcImage, 0, 0);
                

                GraphicsPath ppath = new GraphicsPath();
                //POINTS
                float[] X = new float[path.Length];
                float[] Y = new float[path.Length];
                for (int i = 0; i<path.Length;i++)
                {
                    X[i] = path[i].X;
                    Y[i] = path[i].Y;
                }

                //MEAN
                float sumX = 0;
                float sumY = 0;
                for (int i = 0; i < X.Length; i++)
                    sumX += X[i];
                for (int i = 0; i < Y.Length; i++)
                    sumY += Y[i];

                float resultX = sumX / X.Length;
                float resultY = sumY / Y.Length;
                List<PointF> pathf = new List<PointF>();
                for(int i = 0; i < path.Length; i++)
                {
                   if(CvInvoke.PointPolygonTest(new VectorOfPointF(path), path[i], false)!=1)
                    {
                        pathf.Add(path[i]);
                    }
                }
                PointF center = new PointF(resultX, resultY);

                


                
                g.DrawLines(new Pen(new SolidBrush(Color.Red), 3), pathf.ToArray());


                g.DrawEllipse(new Pen(new SolidBrush(Color.Red), 5), center.X-10, center.Y-10, 10, 10);



                g.SetClip(ppath);
                g.DrawImage(firstb, 0, 0);

                return bluredImage;
            }
        }

        public Image BlurRectangle(Image srcImage, Rectangle face)
        {
            Bitmap firstb =  FastBoxBlur(srcImage, 20, face);
            Image bluredImage = new Bitmap(srcImage.Width, srcImage.Height, srcImage.PixelFormat);

            using (Graphics g = Graphics.FromImage(bluredImage))
            {
                //RectangleF r = new RectangleF(center.X - radius, center.Y - radius,radius * 2, radius * 2);

                // enables smoothing of the edge of the circle (less pixelated)
                g.SmoothingMode = SmoothingMode.AntiAlias;

                // fills background color
                //using (Brush br = new SolidBrush(backGround))
                //{
                //    g.FillRectangle(br, 0, 0, dstImage.Width, dstImage.Height);
                //}

                g.DrawImage(srcImage, 0, 0);
                // adds the new ellipse & draws the image again 
                GraphicsPath Gpath = new GraphicsPath();
                //ppath.AddPolygon(path);
                //path.AddEllipse(r);
                Gpath.AddEllipse(face);
                //g.DrawLines(new Pen(new SolidBrush(Color.Green), 1), path);

                g.SetClip(Gpath);
                g.DrawImage(firstb, 0, 0);

                return bluredImage;
            }
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
