using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace XBF
{
    class QuickHull
    {
        private List<PointF> hull = new List<PointF>();


        private int Side(PointF p1, PointF p2, PointF p)
        {
            float val = (p.Y - p1.Y) * (p2.X - p1.X) -
                      (p2.Y - p1.Y) * (p.X - p1.X);

            if (val > 0)
                return 1;
            if (val < 0)
                return -1;
            return 0;
        }

        private float Distance(PointF p1, PointF p2, PointF p)
        {
            return Math.Abs((p.Y - p1.Y) * (p2.X - p1.X) - (p2.Y - p1.Y) * (p.X - p1.X));
        }


        public List<PointF> Run(List<PointF> points)
        {
            hull.Clear();
            if (points.Count <= 3)
            {
                foreach (var p in points)
                {
                    hull.Add(p);
                }
                return null;
            }

            PointF pmin = points
                .Select(p => new { point = p, x = p.X })
                .Aggregate((p1, p2) => p1.x < p2.x ? p1 : p2).point;

            PointF pmax = points
                .Select(p => new { point = p, x = p.X })
                .Aggregate((p1, p2) => p1.x > p2.x ? p1 : p2).point;

            hull.Add(pmin);
            hull.Add(pmax);

            List<PointF> left = new List<PointF>();
            List<PointF> right = new List<PointF>();

            for (int i = 0; i < points.Count; i++)
            {
                PointF p = points[i];
                if (Side(pmin, pmax, p) == 1)
                    left.Add(p);
                else
                if (Side(pmin, pmax, p) == -1)
                    right.Add(p);
            }
            CreateHull(pmin, pmax, left);
            CreateHull(pmax, pmin, right);
            return hull;
        }

        private void CreateHull(PointF a, PointF b, List<PointF> points)
        {
            int pos = hull.IndexOf(b);

            if (points.Count == 0)
                return;

            if (points.Count == 1)
            {
                PointF pp = points[0];
                hull.Insert(pos, pp);
                return;
            }

            float dist = float.MinValue;
            int point = 0;

            for (int i = 0; i < points.Count; i++)
            {
                PointF pp = points[i];
                float distance = Distance(a, b, pp);
                if (distance > dist)
                {
                    dist = distance;
                    point = i;
                }
            }

            PointF p = points[point];
            hull.Insert(pos, p);
            List<PointF> ap = new List<PointF>();
            List<PointF> pb = new List<PointF>();

            // слева от AP
            for (int i = 0; i < points.Count; i++)
            {
                PointF pp = points[i];
                if (Side(a, p, pp) == 1)
                {
                    ap.Add(pp);
                }
            }
            // слева от PB
            for (int i = 0; i < points.Count; i++)
            {
                PointF pp = points[i];
                if (Side(p, b, pp) == 1)
                {
                    pb.Add(pp);
                }
            }
            CreateHull(a, p, ap);
            CreateHull(p, b, pb);
        }
    }
}
