//This code implements an (finite impulse response)FIR filter and peak finder using scala and spatial.
//Testing was performed using the bash command "spatial myFIRPEAK.scala --test" where "spatial" is located
//(as of 3/1/2017) in "psanaphi110:/reg/common/package/sioan/hyperdsl/spatial/published/Spatial/bin".  Using the 
//--chisel instead of --test argument overides DRAM allocations determined by the chisel configuration.

import spatial.compiler._
import spatial.library._
import spatial.shared._

object myFIRPEAK extends SpatialAppCompiler with myFIRPEAKApp
trait myFIRPEAKApp extends SpatialApp {

type T = SInt			//SInt stands for "signed integer".  This facilitates later code modifications by removing the need to replace every SInt 
type Array[T] = ForgeArray[T]	//ForgeArray is a unique "spatial" type array

def printArr(a: Rep[Array[SInt]], str: String = "") {	//function for printing arrays
    println(str)

    (0 until a.length) foreach { i => print(a(i) + " ") }
   println("")
}

  def main() {

    // Declare SW-HW interface vals.  The commented out lines were from ArgInOut.scala
  	//val x = ArgIn[SInt]		//a single instance of SInt into the FPGA fabric
  	//val y = ArgOut[SInt]		//a single instance of SInt out from the FPGA fabric
  	//val N = args(0).to[SInt]	//the first argument from the command line passed to N
	val c = ArgOut[SInt]		//this holds the peak position from dResult 
	val L = 10			//size of the FIR convolution filter
	val K = 100			//size of the input signal
	val dFil = DRAM[SInt](L)	//instance of DRAM that contains the filter
	val dData = DRAM[SInt](K)	//instance of DRAM that contains the input signal
	val dResult = DRAM[SInt](K)	//instance of DRAM that contains the output signal
	
	//val memPar = 1
	//val tp = memPar (1 -> 64)


	val myFil = Array.fill[SInt](L)(1)	//fills myFil(short for myFilter) with ones
	val mySignal = Array.fill[SInt](K)(0)	//fills myFil(short for mySignal) with zeros
	mySignal(50) = -10			//and adds a negative pulse
	mySignal(55) = 10			//and positive pulse five time steps later
	
	// Connect SW vals to HW vals
	//setArg(x, N)
	//setArg(K,90)
	//setArg(L,10)
	setMem(dFil,myFil)			//load the filter kernel into the designated DRAM
	setMem(dData,mySignal)    		//load the input signal into the designated DRAM 
	//setMem(dResult,mySignal)	

    // Create HW accelerator.  This is what gets run in the FPGA as opposed to ARM? (does spatial require an (system on chip)SoC?)
    Accel {
 	//Pipe { y := x + 5}		//vestige from ArgInOut.scala. 
	val Fil = SRAM[SInt](L)		//instance of SRAM for the filter kernel.  (Because math operations can't be performed from DRAM).
	val Data = SRAM[SInt](K)	//instance of SRAM for the input signal
	//val b = 1

	//val blkA = Reg[SInt]
		
	val Result = SRAM[SInt](K)	//instance of SRAM for the (filtered) ouptut signal

	Fil := dFil(0::L)		//loading the DRAM into SRAM
	Data := dData(0::K) 		//more loading the DRAM into SRAM
	//Result := dResult(0::K,1)

	Pipe (K by 1) {i=>		//pipe is some type of "for" loop?  Typically need a "Par someinteger" to indicate amount of parallelization 
  			// 0 is initial value
  			val reg = Reduce(L by 1) (0) { j=>	// the scala equivalent of reduceleft.
     			Fil(j)*Data(i+j)			// multiplying filter by data
  			} {_+_}					//{_+_} indicates accumulate everything in the preceding curly brackets 
  		Result(i) = reg

		}
		
		dResult(0::K) := Result				//loading SRAM result into DRAM 
	

		val peak = Reduce(K by 1)(pack((0.as[SInt],100.as[SInt]))){ct => //	this section is for finding the peak
		pack(Result(ct),ct)						// iterates over ct and "packs" (spatial syntax?) into a two element array
		//Result(ct)							// the first element is the largest value in the array, and the second element 											//is its index 
		}{(a,b)=>mux(a._1>b._1, a, b)}	//instead of accumulating (i.e. {_+_}) the results are being multiplexed according to the largest value 
	
		c := peak.value._2		//putting the index of the largest value (Result's x axis value).

	}
    


    // Extract results from accelerator
    //val result = getArg(y)
    val outputResult = getMem(dResult)	//getting the DRAM out of the fabric
    val outputResult2 = getArg(c)	//getting the peak location out of the fabric

    // Create validation checks and debug code
    //val gold = N + 4
    //println("expected: " + gold)
    println("first result: " + outputResult2)
    println("second result: ")
    printArr(outputResult)
  }
}
