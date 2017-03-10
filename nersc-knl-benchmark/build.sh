CC angular-integration.cpp -qopenmp -std=c++11 -qopt-report=3 -o haswell.out

module swap craype-haswell craype-mic-knl
CC angular-integration.cpp -qopenmp -std=c++11 -qopt-report=3 -o knl.out
