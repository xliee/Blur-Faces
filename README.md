# xFaceBlur
Detects Faces in photos and blur them
## Dependences
### Emgu.CV
https://www.nuget.org/packages/EMGU.CV/
```
Install-Package Emgu.CV -Version 4.0.1.3373
```
### ZedGraph
https://www.nuget.org/packages/ZedGraph/
```
Install-Package ZedGraph -Version 5.1.7
```

## Example:

```csharp
//Load Image
Bitmap imageFrame = Image.FromFile(path) as Bitmap;

//Create the analyzer instance
//Parameter DEBUG = false
Analyzer analyzer = new Analyzer(false);

//Get the Faces of the picture
List<Rectangle> Faces = ((IAnalyzer)analyzer).getFaceRegions(Analyzer.ImageToByte(imageFrame), analyzer.ssdProtoFile, analyzer.ssdFile);
//Get the face Landmarks
PointF[][][] Landmarks = ((IAnalyzer)analyzer).getLandmarks(Analyzer.ImageToByte(imageFrame), Faces, analyzer.facemarkFileName);

//Create the Face Mask
Bitmap Mask = ((IAnalyzer)analyzer).getOpMask(Analyzer.ImageToByte(imageFrame), Faces, Landmarks);
//blur the face with the Mask
Bitmap Final = ((IAnalyzer)analyzer).BlurFaceWithLandmark(Analyzer.ImageToByte(imageFrame), 12, Faces, Landmarks, Mask);
```
