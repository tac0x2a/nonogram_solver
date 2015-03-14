name := "nonogram_solver"

version := "0.0.1-SNAPSHOT"

scalaVersion := "2.11.4"

libraryDependencies += "org.apache.commons" % "commons-lang3" % "3.3.2"

fork := true

javaOptions ++= Seq(
   "-Djava.library.path=.:./lib"
)

connectInput in run := true
