#include "MajorAssignment.h"

#define NX 500                          // No. of cells in x direction
#define NY 500                          // No. of cells in y direction
#define NZ 500                          // No. of cells in z direction
#define N (NX*NY*NZ)            // N = total number of cells in domain
#define USE_INTRINSICS 1        // 1 = Use AVX Intrinsics, 2 = Use standard code
#define L 1                             // L = length of domain (m)
#define H 1                             // H = Height of domain (m)
#define W 1                             // W = Width of domain (m)
#define DX (L/NX)                       // DX, DY, DZ = grid spacing in x,y,z.
#define DY (H/NY)
#define DZ (W/NZ)
#define ALPHA 0.1                       // Thermal diffusivity
#define DT 2.5e-7                       // Time step (seconds)
#define PHI_X (DT*ALPHA/(DX*DX))  // CFL in x, y and z respectively.
#define PHI_Y (DT*ALPHA/(DY*DY))
#define PHI_Z (DT*ALPHA/(DZ*DZ))
#define Timestep


#define R (1.0)           // Dimensionless specific gas constant
#define GAMA (7.0/5.0)    // Ratio of specific heats
#define CV (R/(GAMA-1.0)) // Cv
#define CP (CV + R)       // Cp

float *dens;              //density
float *temprature;        //temprature
float *xv;                //velocity in x
float *yv;                //velocity in y
float *zv;                //velocity in z
float *press;             //pressure

float *d_dens;              //density 
float *d_temprature;        //temprature
float *d_xv;                //velocity in x
float *d_yv;                //velocity in y
float *d_zv;                //velocity in z
float *d_press;             //pressure


float *U;
float *U_new;
float *E;
float *F;
float *W;
float *FF;
float *FB;
float *FR;
float *FL;
float *FU;
float *FD;

int *h_body;
int *d_body;

int total_cells = 0;            // A counter for computed cells


FILE *OutputResult;

__global__ void GPUTimeStepFunction(float *a, float *b, int *body);
__global__ void GPUHeatContactFunction(float *a, float *b, int *body);

// Declare important variables
void Allocate_Memory() {
        size_t size = N*sizeof(float);
        cudaError_t Error;
        dens = (float*)malloc(size);
        temprature = (float*)malloc(size);
        xv = (float*)malloc(size);
        yv = (float*)malloc(size);
        zv = (float*)malloc(size);
        press = (float*)malloc(size);
        E = (float*)malloc(size);
        F = (float*)malloc(size);
        W = (float*)malloc(size);

        
        Error = cudaMalloc((void**)&d_dens, size);
        printf("CUDA error (malloc d_dens) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&d_temprature, size);
        printf("CUDA error (malloc d_temprature) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&d_xv, size);
        printf("CUDA error (malloc d_xv) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&d_yv, size);
        printf("CUDA error (malloc d_yv) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&d_zv, size);
        printf("CUDA error (malloc d_zv) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&d_press, size);
        printf("CUDA error (malloc d_press) = %s\n",cudaGetErrorString(Error));

        size_t size2 = N*sizeof(float)*5;
        U = (float*)malloc(size2);
        U_new = (float*)malloc(size2);
        FF = (float*)malloc(size2);
        FB = (float*)malloc(size2);
        FU = (float*)malloc(size2);
        FD = (float*)malloc(size2);
        FL = (float*)malloc(size2);
        FR = (float*)malloc(size2);

	/*Error = cudaMalloc((void**)&U, size2);
        printf("CUDA error (malloc U) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&U_new, size2);
        printf("CUDA error (malloc U_new) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&E, size2);
        printf("CUDA error (malloc E) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&F, size2);
        printf("CUDA error (malloc F) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&W, size2);
        printf("CUDA error (malloc W) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&FF, size2);
        printf("CUDA error (malloc FF) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&FB, size2);
        printf("CUDA error (malloc FB) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&FR, size2);
        printf("CUDA error (malloc FR) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&FL, size2);
        printf("CUDA error (malloc FL) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&FU, size2);
        printf("CUDA error (malloc FU) = %s\n",cudaGetErrorString(Error));
        Error = cudaMalloc((void**)&FD, size2);
        printf("CUDA error (malloc FD) = %s\n",cudaGetErrorString(Error));*/


        size_t size3 = N*sizeof(int);
        h_body = (int*)malloc(size3);
        Error = cudaMalloc((void**)&d_body, size3);
        printf("CUDA error (malloc d_body) = %s\n",cudaGetErrorString(Error));
}
void LoadFile(){
	
}
void Init(){
	int i, j, k;
	for(i = 0;i < NX; i++)
	{
		for(j = 0; j < NY; j++)
		{
			for(k = 0;k < NZ; k++)
			{
				if(h_body[i + j*NX + k*NX*NY])
				{
                                        dens[i + j*NX + k*NX*NY] = 0.0;
                                        xv[i + j*NX + k*NX*NY] = 0.0;
                                        yv[i + j*NX + k*NX*NY] = 0.0;
                                        zv[i + j*NX + k*NX*NY] = 0.0;
                                        press[i + j*NX + k*NX*NY] = 0.0;
                                        temprature[i + j*NX + k*NX*NY] = 0.0;
				}
				else
				{
					dens[i + j*NX + k*NX*NY] = 0.01;
					xv[i + j*NX + k*NX*NY] = 5362;
					yv[i + j*NX + k*NX*NY] = 0.0;
                                        zv[i + j*NX + k*NX*NY] = 4450;
					press[i + j*NX + k*NX*NY] = 0.0001;
                                        temprature[i + j*NX + k*NX*NY] = -45;
				}
			}
		}
	}

	for(i = 0;i < NX; i++)
	{
		for(j = 0; j < NY; j++)
		{
			for(k = 0;k < NZ; k++)
			{
				U[i+j*NX + k*NX*NY +0*N] = dens[i+j*NX + k*NX*NY];
        			U[i+j*NX + k*NX*NY +1*N] = dens[i+j*NX + k*NX*NY] * (xv[i+j*NX + k*NX*NY]);
        			U[i+j*NX + k*NX*NY +2*N] = dens[i+j*NX + k*NX*NY] * (yv[i+j*NX + k*NX*NY]);
        			U[i+j*NX + k*NX*NY +3*N] = dens[i+j*NX + k*NX*NY] * (zv[i+j*NX + k*NX*NY]);
				if(h_body[i + j*NX + k*NX*NY] == 0)
				{
        				U[i+j*NX + k*NX*NY +4*N] = dens[i+j*NX + k*NX*NY] * (CV*(press[i+j*NX + k*NX*NY]/dens[i+j*NX + k*NX*NY]/R)
                				+ 0.5*((xv[i+j*NX + k*NX*NY] * xv[i+j*NX + k*NX*NY]) + (yv[i+j*NX + k*NX*NY] * yv[i+j*NX + k*NX*NY]) 
						+ (zv[i+j*NX + k*NX*NY] * zv[i+j*NX + k*NX*NY]));
				}
				else
				{
					U[i+j*NX + k*NX*NY +4*N] = 0;
				}
			}
		}
	}
				
}
void CPUHeatContactFunction(){
	int i, j, k, z;
        for (k = 0; k < NZ; k++) {
		for (j = 0; j < NY; j++) {
			for (i = 0; i < NX; i++) {
				if(!h_body[i + j*NX + k*NX*NY])
				{
					E[i+j*NX + k*NX*NY +0*N] = dens[i+j*NX + k*NX*NY]*xv[i+j*NX + k*NX*NY];
					E[i+j*NX + k*NX*NY +1*N] = dens[i+j*NX]*xv[i+j*NX + k*NX*NY]*xv[i+j*NX + k*NX*NY] + press[i+j*NX + k*NX*NY];
					E[i+j*NX + k*NX*NY +2*N] = dens[i+j*NX + k*NX*NY]*xv[i+j*NX + k*NX*NY]*yv[i+j*NX + k*NX*NY];
                        		E[i+j*NX + k*NX*NY +3*N] = dens[i+j*NX + k*NX*NY]*xv[i+j*NX + k*NX*NY]*zv[i+j*NX + k*NX*NY];
					E[i+j*NX + k*NX*NY +4*N] = xv[i+j*NX + k*NX*NY] * (U[i+j*NX + k*NX*NY +4*N] + press[i+j*NX + k*NX*NY]);
			
					F[i+j*NX + k*NX*NY +0*N] = dens[i+j*NX + k*NX*NY]*yv[i+j*NX + k*NX*NY];
					F[i+j*NX + k*NX*NY +1*N] = dens[i+j*NX + k*NX*NY]*xv[i+j*NX + k*NX*NY]*yv[i+j*NX + k*NX*NY];
					F[i+j*NX + k*NX*NY +2*N] = dens[i+j*NX + k*NX*NY]*yv[i + j*NX + k*NX*NY]*yv[i+j*NX + k*NX*NY] + press[i+j*NX + k*NX*NY];
                        		F[i+j*NX + k*NX*NY +3*N] = dens[i+j*NX + k*NX*NY]*yv[i+j*NX + k*NX*NY]*zv[i+j*NX + k*NX*NY];
					F[i+j*NX + k*NX*NY +4*N] = yv[i+j*NX + k*NX*NY] * (U[i+j*NX + k*NX*NY +4*N] + press[i+j*NX + k*NX*NY] );

                        		W[i+j*NX + k*NX*NY +0*N] = dens[i+j*NX + k*NX*NY]*zv[i+j*NX + k*NX*NY];
                        		W[i+j*NX + k*NX*NY +1*N] = dens[i+j*NX + k*NX*NY]*xv[i+j*NX + k*NX*NY]*zv[i+j*NX + k*NX*NY];
                        		W[i+j*NX + k*NX*NY +2*N] = dens[i+j*NX + k*NX*NY]*yv[i+j*NX + k*NX*NY]*zv[i+j*NX + k*NX*NY];
                        		W[i+j*NX + k*NX*NY +3*N] = dens[i+j*NX + k*NX*NY]*zv[i + j*NX + k*NX*NY]*zv[i+j*NX + k*NX*NY] + press[i+j*NX + k*NX*NY];
                        		W[i+j*NX + k*NX*NY +4*N] = zv[i+j*NX + k*NX*NY] * (U[i+j*NX + k*NX*NY +4*N] + press[i+j*NX + k*NX*NY] );
				}
			}
		}
	}
	// Rusanov flux:Left, Right, Back, Front, Up, Down
	#pragma omp for collapse(2)// private(speed)
	
        for (k = 1; k < (NZ - 1); k++) {
		for (j = 1; j < (NY - 1); j++) {
			for (i = 1; i < (NX - 1); i++) {		
				speed = sqrt(GAMA*press[i+j*NX + k*NX*NY]/dens[i+j*NX + k*NX*NY]);		// speed of sound in air
				for(z = 0; z < 5; z++){
					if(!h_body[i + j*NX + k*NX*NY])
					{
					FL[i+j*NX + k*NX*NY +z*N] = 0.5*(E[i + j*NX + k*NX*NY + z*N] + E[(i-1) + j*NX + k*NX*NY + z*N]) 
							- speed*(U[i + j*NX + k*NX*NY + z*N] - U[(i-1) + j*NX + k*NX*NY + z*N]);
					FR[i+j*NX + k*NX*NY +z*N] = 0.5*(E[i + j*NX + k*NX*NY + z*N] + E[(i+1) + j*NX + k*NX*NY + z*N]) 
							- speed*(U[(i + 1) + j*NX + k*NX*NY + z*N] - U[i + j*NX + k*NX*NY + z*N]);

					FB[i+j*NX + k*NX*NY +z*N] = 0.5*(F[i + (j-1)*NX + k*NX*NY + z*N] + F[i + j*NX + k*NX*NY + z*N])
							- speed*(U[i + j*NX + k*NX*NY + z*N] - U[i + (j-1)*NX + k*NX*NY + z*N]);
					FF[i+j*NX + k*NX*NY +z*N] = 0.5*(F[i + j*NX + k*NX*NY + z*N] + F[i + (j+1)*NX + k*NX*NY + z*N])
							- speed*(U[i + (j+1)*NX + k*NX*NY + z*N] - U[i + j*NX + k*NX*NY + z*N]);
        			        FD[i+j*NX + k*NX*NY +z*N] = 0.5*(W[i + j*NX + k*NX*NY + z*N] + W[i + j*NX + (k-1)*NX*NY + z*N])
							- speed*(U[i + j*NX + k*NX*NY + z*N] - U[i + j*NX + (k-1)*NX*NY + z*N]);
        			        FU[i+j*NX + k*NX*NY +z*N] = 0.5*(W[i + j*NX + k*NX*NY + z*N] + W[i + j*NX + (k+1)*NX*NY + z*N])
							- speed*(U[i + j*NX + (k+1)*NX*NY + z*N] - U[i + j*NX + k*NX*NY + z*N]);
				}
			}
		}
	}
	#pragma omp barrier
}

void CalRenewResult(){
	int i, j, k, z;
	// Update U by FVM
	#pragma omp for collapse(2)
        for (k = 1; k < (NZ - 1); k++) {
		for (j = 1; j < (NY - 1); j++) {
			for (i = 1; i < (NX - 1); i++) {
                        	for (z = 0; z < 5; z++) {
					if(!h_body[i + j*NX + k*NX*NY])
					{
					U_new[i+j*NX + k*NX*NY +z*N] = U[i+j*NX + k*NX*NY +z*N] - (dt/dx)*(FR[i+j*NX + k*NX*NY +z*N] - FL[i+j*NX + k*NX*NY +z*N])
					- (dt/dy)*(FF[i+j*NX + k*NX*NY +z*N] - FB[i+j*NX + k*NX*NY +z*N]) 
					- (dt/dz)*(FU[i+j*NX + k*NX*NY +z*N] - FD[i+j*NX + k*NX*NY +z*N]);
					}
				}
			}
		}
	}
	#pragma omp barrier

	//Renew left and right boundary condition
	#pragma omp for
	for (i = 1; i < (NX - 1); i++) {
        	for (k = 1; k < (NZ - 1); k++) {
                	for (z = 0; z < 5; z++) {
		U_new[i + k*NX*NY + z*N] = U_new[i + NX + k*NX*NY + z*N];
		U_new[i + (NY-1)*NX + k*NX*NY + z*N] = U_new[i + (NY-2)*NX + k*NX*NY + z*N];
			}
		}
	}
	#pragma omp barrier

	//Renew back and front boundary condition
	#pragma omp for
	for (j = 0; j < NY; j++) {
                for (k = 1; k < (NZ - 1); k++) {
                        for (z = 0; z < 5; z++) {
		U_new[j*NX + k*NX*NY + z*N] = U_new[1 + j*NX + k*NX*NY + z*N];
		U_new[(NX-1)+j*NX + k*NX*NY + z*N] = U_new[(NX-2)+j*NX + k*NX*NY + z*N];
			}
		}
	}
	#pragma omp barrier
	//Renew top and down boundary condition
        #pragma omp for
        for (i = 0; i < NX; i++) {
	        for (j = 0; j < NY; j++) {
	                for (z = 0; z < 5; z++) {
                U_new[i + j*NX + (NZ-1)*NX*NY + z*N] = U_new[i + j*NX + (NZ-2)*NX*NY + z*N];
                U_new[i + j*NX + z*N] = U_new[i +j*NX + NX*NY + z*N];
			}
		}
        }
        #pragma omp barrier

	// Update density, velocity, pressure, and U
	#pragma omp for collapse(2)
        for (k = 0; k < NZ; k++) {
		for (j = 0; j < NY; j++) {
			for (i = 0; i < NX; i++) {	
					if(!h_body[i + j*NX + k*NX*NY])
					{
						dens[i + j*NX + k*NX*NY] = U_new[i + j*NX + k*NX*NY + 0*N];
						xv[i + j*NX + k*NX*NY] = U_new[i + j*NX + k*NX*NY + 1*N] / U_new[i + j*NX + k*NX*NY + 0*N];
						yv[i + j*NX + k*NX*NY] = U_new[i + j*NX + k*NX*NY + 2*N] / U_new[i + j*NX + k*NX*NY + 0*N];
                        			zv[i + j*NX + k*NX*NY] = U_new[i + j*NX + k*NX*NY + 3*N] / U_new[i + j*NX + k*NX*NY + 0*N];
						press[i+j*NX + k*NX*NY] = (GAMA-1) * (U_new[i+j*NX + k*NX*NY + 3*N] - 0.5 * dens[i+j*NX + k*NX*NY] * (xv[i+j*NX + k*NX*NY]*xv[i+j*NX + k*NX*NY] + yv[i+j*NX + k*NX*NY]*yv[i+j*NX + k*NX*NY] +  zv[i+j*NX + k*NX*NY]*zv[i+j*NX + k*NX*NY]));
                        			temprature[i + j*NX + k*NX*NY] = (press[i + j*NX + k*NX*NY + 3*N]*28.8 / dens[i + j*NX + k*NX*NY + 0*N]) - 273;
						U[i+j*NX + k*NX*NY + 0*N] = U_new[i+j*NX + k*NX*NY + 0*N];
						U[i+j*NX + k*NX*NY + 1*N] = U_new[i+j*NX + k*NX*NY + 1*N];
						U[i+j*NX + k*NX*NY + 2*N] = U_new[i+j*NX + k*NX*NY + 2*N];
						U[i+j*NX + k*NX*NY + 3*N] = U_new[i+j*NX + k*NX*NY + 3*N];
						U[i+j*NX + k*NX*NY + 4*N] = U_new[i+j*NX + k*NX*NY + 4*N];
					}
			}
		}
	}
	#pragma omp barrier

}
void Call_GPUHeatContactFunction(){
}
__global__ void GPUHeatContactFunction(float *a, float *b, int *body){
		
}
void Call_GPUTimeStepFunction(){
}
__global__ void GPUTimeStepFunction(float *a, float *b, int *body){
}
void Send_To_Device(){
}
void Get_From_Device(){
}
void Free_Memory() {
        if (dens) free(dens);
        if (temprature) free(temprature);
	if (xv) free(xv);
        if (yv) free(yv);
        if (zv) free(zv);
        if (press) free(press);
	if (h_body) free(h_body);
        
        if (U) free(U);
        if (U_new) free(U_new);
        if (FF) free(FF);
        if (FB) free(FB);
        if (FU) free(FU);
        if (FD) free(FD);
        if (FL) free(FL);
        if (FR) free(FR);
        if (E) free(E);
        if (F) free(F);
        if (W) free(W);

	if (d_dens) cudaFree(d_dens);
        if (d_temprature) cudafree(d_temprature);
        if (d_xv) cudafree(d_xv);
        if (d_yv) cudafree(d_yv);
        if (d_zv) cudafree(d_zv);
        if (d_press) cudafree(d_press);
        if (d_body) cudaFree(d_body);
}
void Save_Data() {
	int i, j, k;	
	for (k = 0; k < NZ; k++) {	
		for (j = 0; j < NY; j++) {
			for (i = 0; i < NX; i++) {
			fprintf(OutputResult, "%d %d %d %f\n", i+1, j+1, k+1, dens[i+j*NX]);
			}
		}
	}
}
