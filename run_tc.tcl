# to run: vitis-run --mode hls --tcl run_tcl.tcl
# Create a project
open_component -reset component_name_tf1 -flow_target vivado

# Add design files
#########add_files example.cpp
#########add_files accel.cc
# Add test bench & files
#########add_files -tb example_test.cpp
#########add_files -tb result.golden.dat
add_files Accel.cpp -cflags "-ILayers"
add_files -tb TestBenchAccel.cpp -cflags "-ITests -Wno-unknown-pragmas"

# Set the top-level function
#########set_top top
set_top accel

# ########################################################
# Create a solution
# Define technology and clock rate
set_part  {xczu1cg-sbva484-1-e}
create_clock -period 5

# Set variable to select which steps to execute
set hls_exec 1


#########csim_design # pre-synthesis sim 
# Set any optimization directives

# End of directives

if {$hls_exec == 1} {
	# Run Synthesis and Exit
	csynth_design	
} elseif {$hls_exec == 2} {
	# Run Synthesis, RTL Simulation and Exit
	csynth_design
	
	cosim_design
} elseif {$hls_exec == 3} { 
	# Run Synthesis, RTL Simulation, RTL implementation and Exit
	csynth_design
	
	cosim_design
	export_design
} else {
	# Default is to exit after setup
	csynth_design
}

exit
