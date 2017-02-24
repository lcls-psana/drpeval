CC angular-integration.cpp -std=c++11 -o haswell.out

module swap craype-haswell craype-mic-knl
CC angular-integration.cpp -std=c++11 -o knl.out
