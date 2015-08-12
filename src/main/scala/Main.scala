// Author::    TAC (tac@tac42.net)

import java.lang.Math
import org.opencv.core.{Core, CvType, Mat, MatOfPoint, Point, Scalar}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

object NonoSolv {

  // 与えられた画像中から 縦線ぽいのを抽出する
  def edge_v(src: Mat, threshold:Int = 20, procImg:Mat = new Mat): List[Tuple2[Point,Point]] = {

    // Sobelフィルタでエッジ検出
    val scale = 1.0
    val delta = 0
    val x_order = 1
    val y_order = 0
    val averture = 1
    Imgproc.Sobel(src, procImg, CvType.CV_8U, x_order, y_order, averture, scale, delta)

    // ハフ変換で直線検出
    val houghThreshold = 80
    val minLineSize = procImg.rows / 10
    val lineGap = 10
    val rad = Math.PI / 180
    var result = new Mat
    Imgproc.HoughLinesP(procImg, result, 1, rad, houghThreshold, minLineSize, lineGap)

    // x方向にソートして始点終点のタプルにする
    val lines: List[Tuple2[Point,Point]] = (0 to result.rows - 1)
      .map { i => result.get(i, 0) }
      .map { vec => (new Point(vec(0), vec(1)), new Point(vec(2), vec(3)))}
      .sortBy { case (start,end) =>  (end.x + start.x) / 2 }.toList

    // 重複して検出してそうなやつは取り除く
    var prev_x = -1.0
    val reducedLiens: List[Tuple2[Point,Point]] = lines.filter {
      case (start,end) =>
        val x_avr = (end.x + start.x) / 2

        val diff = x_avr - prev_x
        if(diff > threshold){
          prev_x = x_avr
        }
        diff > threshold
    }

    return reducedLiens
  }


  // 与えられた画像中から 横線ぽいのを抽出する
  def edge_h(src: Mat, threshold:Int = 20, procImg:Mat = new Mat): List[Tuple2[Point,Point]] = {

    // Sobelフィルタでエッジ検出
    val scale = 1.0
    val delta = 0
    val x_order = 0
    val y_order = 1
    val averture = 1
    Imgproc.Sobel(src, procImg, CvType.CV_8U, x_order, y_order, averture, scale, delta)

    // ハフ変換で直線検出
    val houghThreshold = 100
    val minLineSize = procImg.rows / 10
    val lineGap = 10
    val rad = Math.PI / 180
    var result = new Mat
    Imgproc.HoughLinesP(procImg, result, 1, rad, houghThreshold, minLineSize, lineGap)

    // y方向にソートして始点終点のタプルにする
    val lines: List[Tuple2[Point,Point]] = (0 to result.rows - 1)
      .map { i => result.get(i, 0) }
      .map { vec => (new Point(vec(0), vec(1)), new Point(vec(2), vec(3)))}
      .sortBy { case (start,end) =>  (end.y + start.y) / 2 }.toList

    // 重複して検出してそうなやつは取り除く
    var prev_y = -1.0
    val reducedLiens: List[Tuple2[Point,Point]] = lines.filter {
      case (start,end) =>
        val y_avr = (end.y + start.y) / 2

        val diff = y_avr - prev_y
        if(diff > threshold){
          prev_y = y_avr
        }
        diff > threshold
    }

    return reducedLiens
  }

  private def detectCrossPoint(l1: Tuple2[Point,Point], l2: Tuple2[Point,Point]): Point = {
    val (p1, p3) = l1
    val (p2, p4) = l2

    val s1 = (p4.x - p2.x) * (p1.y - p2.y) - (p4.y - p2.y) * (p1.x - p2.x)
    val s2 = (p4.x - p2.x) * (p2.y - p3.y) - (p4.y - p2.y) * (p2.x - p3.x)

    new Point(
      p1.x + (p3.x - p1.x) * s1 / (s1+s2),
      p1.y + (p3.y - p1.y) * s1 / (s1+s2) )

  }

  // 与えられた縦v本、横h本のから、(v-1)x(h-x)の格子を計算して返す
  def detectRectangles( vLines:List[Tuple2[Point,Point]], hLines:List[Tuple2[Point,Point]] ): Seq[Seq[MatOfPoint]] = {

    0 to hLines.size -2 map {   h =>
      val hLine = hLines(h)
      val hLineNext = hLines(h+1)

      0 to vLines.size -2 map { v =>
        val vLine = vLines(v)
        val vLineNext = vLines(v+1)

        new MatOfPoint(
          detectCrossPoint(hLine, vLine),
          detectCrossPoint(hLine, vLineNext),
          detectCrossPoint(hLineNext, vLineNext),
          detectCrossPoint(hLineNext, vLine)
        )

      }
    }
  }
}

object Main {

  // 画像ファイルへのパスを取得
  val imgDirPath = getClass.getResource(".").getPath + "../../../src/main/resource/"

  val srcFilePath = imgDirPath + "src.jpg"



  def sample = {
    println("Load OpenCV" + Core.VERSION)
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)

    // 画像ファイルの読み込み
    val src = Imgcodecs.imread(srcFilePath, CvType.CV_8U)

    Imgproc.threshold(src, src, 0.0, 255.0, Imgproc.THRESH_TRIANGLE)

    // 膨張収縮でノイズ除去
    Imgproc.dilate(src, src, new Mat)
    Imgproc.erode(src, src, new Mat)
    Imgproc.Canny(src, src, 80,100)

    // x,y方向のエッジを抽出
    val linesV = NonoSolv.edge_v(src, 20)
    val linesH = NonoSolv.edge_h(src, 20)

    // 検出した軸を使って補正
    val xl: Seq[Double] = linesV.map{ case (start,end) => (end.x + start.x) / 2 }
    val yl: Seq[Double] = linesH.map{ case (start,end) => (end.y + start.y) / 2 }

    val maxX = xl.max
    val minX = xl.min
    val maxY = yl.max
    val minY = yl.min

    val fixLinesV = linesV.map { case(start,end) =>
      val a = (end.x - start.x) / (end.y - start.y)
      val startX = start.x - ((start.y - minY) * a)
      val endX   = end.x + ((maxY - end.y) * a)
      (new Point(startX, minY), new Point(endX, maxY))
    }
    val fixLinesH = linesH.map { case(start,end) =>
      val a = (end.y - start.y) / (end.x - start.x)
      val startY = start.y - ((start.x - minX) * a)
      val endY   = end.y + ((maxX - end.x) * a)
      (new Point(minX, startY), new Point(maxX, endY))
    }

    // 描画して出力
    val dst = Imgcodecs.imread(srcFilePath)

    fixLinesV.foreach{ case (start,end) => Imgproc.line(dst, start, end, new Scalar(255,0,0), 5)}
    fixLinesH.foreach{ case (start,end) => Imgproc.line(dst, start, end, new Scalar(0,0,255), 5)}

    val detectedImgPath = imgDirPath + "linedetected.png"
    Imgcodecs.imwrite(detectedImgPath, dst)

    val points = NonoSolv.detectRectangles(fixLinesV, fixLinesH)
    var t = 2
    points.flatten.foreach{ p =>
      (t % 3) match {
        case 0 => Imgproc.fillConvexPoly(dst, p, new Scalar(255-(t*5), (t*5), 0))
        case 1 => Imgproc.fillConvexPoly(dst, p, new Scalar((t*5), 255-t, 0))
        case 2 => Imgproc.fillConvexPoly(dst, p, new Scalar(255-t, 255-t, (t*5)))
      }
      t += 1
    }

    // 画像ファイルの書き込み
    val dstFilePath = imgDirPath + "output.png"
    Imgcodecs.imwrite(dstFilePath, dst)


    // 切り出してみる
    points.flatten.foreach { p => println(p.toList()) }

  }

  def main(args: Array[String]) {
    sample
  }
}
