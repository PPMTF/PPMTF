CXXFLAGS	= -march=native -O3
CXXFLAGS	+= -std=c++11
CXXFLAGS	+= -isystem ./include
CXXFLAGS	+= -MMD
CC	= g++

TARGET	=  PPITF PPMTF PPMTF_time SynData_PPITF SynData_PPMTF SynData_PPMTF_time PDTest_Trace PDTest_Trace_time SynData_SGD SynData_SGD_time

all: $(TARGET)

t: EvalPPITFError
	./$<

t0: SynData_SGD.0
	./$< PF TK 10 5

PDTest_Trace: PDTest_Trace.o common.o

PDTest_Trace_time: PDTest_Trace_time.o common.o

PPMTF: PPMTF.o PPXTF.o common.o

PPMTF_time: PPMTF_time.o PPXTF.o common.o

PPITF: PPITF.o PPXTF.o common.o

SynData_PPITF: SynData_PPITF.o common.o

SynData_PPMTF: SynData_PPMTF.o common.o

SynData_PPMTF_time: SynData_PPMTF_time.o common.o

SynData_SGD: SynData_SGD.o common.o

SynData_SGD_time: SynData_SGD_time.o common.o

clean:
	rm -f *~ $(TARGET) *.d *.o

-include *.d
