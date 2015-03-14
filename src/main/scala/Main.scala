// Author::    TAC (tac@tac42.net)

import org.opencv.core.Core

object Main {

  def sample = {
    println("Load OpenCV" + Core.VERSION)
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
  }

  def main(args: Array[String]) = {
    sample
  }
}
