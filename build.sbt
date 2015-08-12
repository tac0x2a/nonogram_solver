name := "nonogram_solver"

version := "0.0.1-SNAPSHOT"

scalaVersion := "2.11.4"

resolvers ++= Seq(
  "tess4j" at "net.sourceforge.tess4j"
)

libraryDependencies ++= Seq(
  "net.sourceforge.tess4j" % "tess4j" % "2.0.1" % "provided"
)

fork := true

javaOptions ++= Seq(
   "-Djava.library.path=.:./lib"
)

connectInput in run := true

run in Compile <<= Defaults.runTask(fullClasspath in Compile, mainClass in (Compile, run), runner in (Compile, run))
runMain in Compile <<= Defaults.runMainTask(fullClasspath in Compile, runner in (Compile, run))
