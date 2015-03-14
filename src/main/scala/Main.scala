// Author::    TAC (tac@tac42.net)

import java.lang.Math
import org.opencv.core.{Core, CvType, Mat, Point, Scalar}
import org.opencv.highgui.Highgui
import org.opencv.imgproc.Imgproc

object NonoSolv {

  // 与えられた画像中から 縦線ぽいのを抽出する
  def edge_x(src: Mat, threshold:Int = 20, procImg:Mat = new Mat): List[Tuple2[Point,Point]] = {

    // Sobelフィルタでエッジ検出
    val scale = 1.0
    val delta = 0
    val x_order = 1
    val y_order = 0
    val averture = 1
    Imgproc.Sobel(src, procImg, CvType.CV_8U, x_order, y_order, averture, scale, delta)

    // ハフ変換で直線検出
    val houghThreshold = 300
    val minLineSize = procImg.rows / 3
    val lineGap = 10
    var result = new Mat
    Imgproc.HoughLinesP(procImg, result, 1, Math.PI / 180, houghThreshold, minLineSize, lineGap)

    // x方向にソートして始点終点のタプルにする
    val lines: List[Tuple2[Point,Point]] = (0 to result.cols - 1)
      .map { i => result.get(0, i) }
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
  def edge_y(src: Mat, threshold:Int = 20, procImg:Mat = new Mat): List[Tuple2[Point,Point]] = {

    // Sobelフィルタでエッジ検出
    val scale = 1.0
    val delta = 0
    val x_order = 0
    val y_order = 1
    val averture = 1
    Imgproc.Sobel(src, procImg, CvType.CV_8U, x_order, y_order, averture, scale, delta)

    // ハフ変換で直線検出
    val houghThreshold = 300
    val minLineSize = procImg.rows / 3
    val lineGap = 10
    var result = new Mat
    Imgproc.HoughLinesP(procImg, result, 1, Math.PI / 180, houghThreshold, minLineSize, lineGap)

    // y方向にソートして始点終点のタプルにする
    val lines: List[Tuple2[Point,Point]] = (0 to result.cols - 1)
      .map { i => result.get(0, i) }
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


}

object Main {

  // 画像ファイルへのパスを取得
  val imgDirPath = getClass.getResource(".").getPath + "../../../src/main/resource/"

  val filePath = imgDirPath + "src.jpg"

  val dstFilePath = imgDirPath + "edge.png"

  def sample = {
    println("Load OpenCV" + Core.VERSION)
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)

    // 画像ファイルの読み込み
    val src = Highgui.imread(filePath, CvType.CV_8U)


    // 膨張収縮でノイズ除去
    Imgproc.dilate(src, src, new Mat)
    Imgproc.dilate(src, src, new Mat)
    Imgproc.erode(src, src, new Mat)
    Imgproc.erode(src, src, new Mat)

    // x,y方向のエッジを抽出
    val linesX = NonoSolv.edge_x(src, 20)
    val linesY = NonoSolv.edge_y(src, 20)

    // 検出した軸を使って補正
    val xl: Seq[Double] = linesX.map{ case (start,end) => (end.x + start.x) / 2 }
    val yl: Seq[Double] = linesY.map{ case (start,end) => (end.y + start.y) / 2 }

    val maxX = xl.max
    val minX = xl.min
    val maxY = yl.max
    val minY = yl.min

    val fixLinesX = linesX.map { case(start,end) =>
      val a = (end.x - start.x) / (end.y - start.y)
      val startX = start.x - ((start.y - minY) * a)
      val endX   = end.x + ((maxY - end.y) * a)
      (new Point(startX, minY), new Point(endX, maxY))
    }
    val fixLinesY = linesY.map { case(start,end) =>
      val a = (end.y - start.y) / (end.x - start.x)
      val startY = start.y - ((start.x - minX) * a)
      val endY   = end.y + ((maxX - end.x) * a)
      (new Point(minX, startY), new Point(maxX, endY))
    }

        // 枠を検出するよ
    // 描画して出力
    val dst = Highgui.imread(filePath)
    fixLinesX.foreach{ case (start,end) => Core.line(dst, start, end, new Scalar(255,0,0), 5)}
    fixLinesY.foreach{ case (start,end) => Core.line(dst, start, end, new Scalar(0,0,255), 5)}


    // 画像ファイルの書き込み
    Highgui.imwrite(dstFilePath, dst)
  }

  def main(args: Array[String]) = {
    sample
  }
}
