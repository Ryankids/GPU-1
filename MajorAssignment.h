#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NO_STEPS 2000000


// Declare functions
void Init();
void LoadFile();
void Allocate_Memory();
void Send_To_Device();
void Call_GPUHeatContactFunction();
void Call_GPUTimeStepFunction();
void CPUHeatContactFunction();
void CalRenewResult();
void Get_From_Device();
void Save_Data();
void Free_Memory();
