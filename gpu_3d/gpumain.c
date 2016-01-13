#include "gpumain.h"

#define NX 100                          // No. of cells in x direction
#define NY 100                          // No. of cells in y direction
#define NZ 100                          // No. of cells in z direction
#define N NX*NY*NZ            // N = total number of cells in domain
#define L 100                             // L = length of domain (m)
#define H 100                             // H = Height of domain (m)
#define W 100                             // W = Width of domain (m)
#define DX L/NX                       // DX, DY, DZ = grid spacing in x,y,z.
#define DY H/NY
#define DZ W/NZ
#define DT 0.00001                       // Time step (seconds)

#define R 287.0           // Gas Constant -> unit: J/(kg*K)
#define GAMA 7.0/5.0    // Ratio of specific heats
#define CV R/(GAMA-1.0) // Cv
#define CP CV + R       // Cp

float *dens;          // host density
float *xv;            // host velocity in x
float *yv;            // host velocity in y
float *zv;            // host velocity in z
float *press;         // host pressure
float *temperature;   // host temperature

float *d_dens;        // device density
float *d_xv;          // device velocity in x
float *d_yv;          // device velocity in y
float *d_zv;          // device velocity in z
float *d_press;       // device pressure

float *U;

float *d_U;
float *d_E;
float *d_F;
float *d_G;
float *d_U_new;

float *d_FL;
float *d_FR;
float *d_FB;
float *d_FF;
float *d_FD;
float *d_FU;

float *h_body;
float *d_body;

__global__ void CalculateFlux(float* d_body, float* dens, float* xv, float* yv, float* zv, float* press,float* E, float* F, float* G,float* U);
__global__ void CalculateFlux2(float* d_body, float* dens, float* xv, float* yv, float* zv,float* press,float* E, float* F, float* G,
		float* FL, float* FR, float* FB, float* FF, float* FD, float* FU,float* U);
__global__ void CalculateFlux3(float* d_body, float* dens, float* xv, float* yv, float* zv,float* press,float* U, float* U_new);

void Load_Dat_To_Array(char* input_file_name) {
	FILE *pFile;
	pFile = fopen(input_file_name, "r");
	if (!pFile) { printf("Open failure"); }

	int x, y, z;
	for (z = 0; z < NZ; z++) {
		for (y = 0; y < NY; y++) {
			for (x = 0; x < NX; x++) {
				h_body[z * NX * NY + y * NX + x] = -1.0;
			}
		}
	}
	float tmp1, tmp2, tmp3;
	int idx = 0;

	// According to the 50x50x50 order
	for (x = 75; x > 25; x--) {
		for (z = 25; z < 75; z++) {
			for (y = 25; y < 75; y++) {
				idx = z * NX * NY + y * NX + x;
				fscanf(pFile, "%f%f%f%f", &tmp1, &tmp2, &tmp3, &h_body[idx]);
				/* test... 0.040018	 -0.204846	 -0.286759	 -1 */
				//if(body[idx] == 0.0) { 
				//	printf("%d, %f\n", idx, body[idx]);
				//	system("pause"); 
				//}
			}
		}
	}
	fclose(pFile);
}

void Init() {
	int i, j, k;
	for (i = 0; i < NX; i++) {
		for (j = 0; j < NY; j++) {
			for (k = 0; k < NZ; k++) {
				if (h_body[i + j*NX + k*NX*NY] == 0.0) { // body
					dens[i + j*NX + k*NX*NY] = 1.0;
					xv[i + j*NX + k*NX*NY] = 1.0;
					yv[i + j*NX + k*NX*NY] = 1.0;
					zv[i + j*NX + k*NX*NY] = 1.0;;
					press[i + j*NX + k*NX*NY] = 1.0;
					temperature[i + j*NX + k*NX*NY] = 1.0;
				}
				else { // air reference: http://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
					dens[i + j*NX + k*NX*NY] = 0.00082 * 0.1;				// unit: kg / m^3
					xv[i + j*NX + k*NX*NY] = -7000.0;						// unit: m / s
					yv[i + j*NX + k*NX*NY] = 0.0;						// unit: m / s
					zv[i + j*NX + k*NX*NY] = 0.0;						// unit: m / s
					press[i + j*NX + k*NX*NY] = 0.00052 * 10000;		// unit: (kg*m/s^2) / m^2
					temperature[i + j*NX + k*NX*NY] = -53 + 273.15;		// unit: K
				}

				U[i + j*NX + k*NX*NY + 0 * N] = dens[i + j*NX + k*NX*NY];
				U[i + j*NX + k*NX*NY + 1 * N] = dens[i + j*NX + k*NX*NY] * (xv[i + j*NX + k*NX*NY]);
				U[i + j*NX + k*NX*NY + 2 * N] = dens[i + j*NX + k*NX*NY] * (yv[i + j*NX + k*NX*NY]);
				U[i + j*NX + k*NX*NY + 3 * N] = dens[i + j*NX + k*NX*NY] * (zv[i + j*NX + k*NX*NY]);
				U[i + j*NX + k*NX*NY + 4 * N] = dens[i + j*NX + k*NX*NY] * (CV*(press[i + j*NX + k*NX*NY] / dens[i + j*NX + k*NX*NY] / R)
					+ 0.5*((xv[i + j*NX + k*NX*NY] * xv[i + j*NX + k*NX*NY]) + (yv[i + j*NX + k*NX*NY] * yv[i + j*NX + k*NX*NY])
					+ (zv[i + j*NX + k*NX*NY] * zv[i + j*NX + k*NX*NY])));
			}
		}
	}
}

void Allocate_Memory() {
	size_t size = N*sizeof(float);
	cudaError_t Error;
	dens = (float*)malloc(size);
	xv = (float*)malloc(size);
	yv = (float*)malloc(size);
	zv = (float*)malloc(size);
	press = (float*)malloc(size);
	temperature = (float*)malloc(size);
	h_body = (float*)malloc(size);

        size_t size2 = 5*N*sizeof(float);
        U = (float*)malloc(size2);

	Error = cudaMalloc((void**)&d_body, size);
	printf("CUDA error (malloc d_body) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_dens, size);
	printf("CUDA error (malloc d_dens) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_xv, size);
	printf("CUDA error (malloc d_xv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_yv, size);
	printf("CUDA error (malloc d_yv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_zv, size);
	printf("CUDA error (malloc d_zv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_press, size);
	printf("CUDA error (malloc d_press) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_U, size2);
	printf("CUDA error (malloc d_U) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_U_new, size2);
	printf("CUDA error (malloc d_U_new) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_E, size2);
	printf("CUDA error (malloc d_E) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_F, size2);
	printf("CUDA error (malloc d_F) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_G, size2);
	printf("CUDA error (malloc d_G) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FR, size2);
	printf("CUDA error (malloc d_FR) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FL, size2);
	printf("CUDA error (malloc d_FL) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FU, size2);
	printf("CUDA error (malloc d_FU) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FD, size2);
	printf("CUDA error (malloc d_FD) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FF, size2);
	printf("CUDA error (malloc d_FF) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_FB, size2);
	printf("CUDA error (malloc d_FB) = %s\n", cudaGetErrorString(Error));
}

void Free_Memory() {
	if (dens) free(dens);
	if (xv) free(xv);
	if (yv) free(yv);
	if (zv) free(zv);
	if (press) free(press);
	if (temperature) free(temperature);
	if (U) free(U);
	if (h_body) free(h_body);
	if (d_dens) cudaFree(d_dens);
	if (d_xv) cudaFree(d_xv);
	if (d_yv) cudaFree(d_yv);
	if (d_zv) cudaFree(d_zv);
	if (d_U) cudaFree(d_U);
	if (d_U_new) cudaFree(d_U_new);
	if (d_E) cudaFree(d_E);
	if (d_F) cudaFree(d_F);
	if (d_G) cudaFree(d_G);
	if (d_FR) cudaFree(d_FR);
	if (d_FL) cudaFree(d_FL);
	if (d_FU) cudaFree(d_FU);
	if (d_FD) cudaFree(d_FD);
	if (d_FB) cudaFree(d_FB);
	if (d_FF) cudaFree(d_FF);
	if (d_body) cudaFree(d_body);

}

void Send_To_Device() {
	size_t size = N*sizeof(float);
	cudaError_t Error;
	Error = cudaMemcpy(d_body, h_body, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy h_body -> d_body) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(d_dens, dens, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy dens -> d_dens) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(d_xv, xv, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy xv -> d_xv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(d_yv, yv, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy yv -> d_yv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(d_zv, zv, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy zv -> d_zv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(d_press, press, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy press -> d_press) = %s\n", cudaGetErrorString(Error));

        size_t size2 = 5*N*sizeof(float);
	Error = cudaMemcpy(d_U, U, size2, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy U -> d_U) = %s\n", cudaGetErrorString(Error));

}

void Get_From_Device() {
	size_t size = N*sizeof(float);
	cudaError_t Error;
	Error = cudaMemcpy(dens, d_dens, size, cudaMemcpyDeviceToHost);
	printf("CUDA error (memcpy d_dens -> dens) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(xv, d_xv, size, cudaMemcpyDeviceToHost);
	printf("CUDA error (memcpy d_xv -> xv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(yv, d_yv, size, cudaMemcpyDeviceToHost);
	printf("CUDA error (memcpy d_yv -> yv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(zv, d_zv, size, cudaMemcpyDeviceToHost);
	printf("CUDA error (memcpy d_zv -> zv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMemcpy(press, d_press, size, cudaMemcpyDeviceToHost);
	printf("CUDA error (memcpy d_press -> press) = %s\n", cudaGetErrorString(Error));
}

__global__ void CalculateFlux(float* d_body, float* dens, float* xv, float* yv, float* zv, float* press,float* E, float* F, float* G,float* U) {
	
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (i < N) {
		if (d_body[i] == -1.0) { // air
				E[i + 0 * N] = dens[i] * xv[i];
				E[i + 1 * N] = dens[i] * xv[i] * xv[i] + press[i];
				E[i + 2 * N] = dens[i] * xv[i] * yv[i];
				E[i + 3 * N] = dens[i] * xv[i] * zv[i];
				E[i + 4 * N] = xv[i] * (U[i + 4 * N] + press[i]);

				F[i + 0 * N] = dens[i] * yv[i];
				F[i + 1 * N] = dens[i] * xv[i] * yv[i];
				F[i + 2 * N] = dens[i] * yv[i] * yv[i] + press[i];
				F[i + 3 * N] = dens[i] * yv[i] * zv[i];
				F[i + 4 * N] = yv[i] * (U[i + 4 * N] + press[i]);

				G[i + 0 * N] = dens[i] * zv[i];
				G[i + 1 * N] = dens[i] * xv[i] * zv[i];
				G[i + 2 * N] = dens[i] * yv[i] * zv[i];
				G[i + 3 * N] = dens[i] * zv[i] * zv[i] + press[i];
				G[i + 4 * N] = zv[i] * (U[i + 4 * N] + press[i]);
		}
	}
}
__global__ void CalculateFlux2(float* d_body, float* dens, float* xv, float* yv, float* zv,float* press,float* E, float* F, float* G,
                float* FL, float* FR, float* FB, float* FF, float* FD, float* FU,float* U){
        int i = blockDim.x*blockIdx.x + threadIdx.x;

	float speed;
	speed = (float)((DX) / DT / 3.0)*0.5; // speed of sound in air
	if (i < N) {
		if (d_body[i] == -1.0) { // air
			// Rusanov flux: Left, Right, Back, Front, Down, Up
			if ( (i % NX) != 0 && (i % NX) != (NX - 1)) {
				if( i % (NX*NY) >= NX && i % (NX*NY) < NX*(NY - 1)){
					if( i > NX*NY && i < NX*NY*(NZ - 1)){ 
				// ((i % 100) != 0) && ((i % 100) != 99) && (i % 10000 >= 100) && (i % 10000 <= 9899) && (i >= 10000) && (i <= 989999) 
				// delete left plane, delete right plane, delete back plane, delete front plane, delete down plane, delete up plane 

				FL[i + 0 * N] = 0.5*(E[i + 0 * N] + E[i - 1 + 0 * N]) - speed*(U[i + 0 * N] - U[i - 1 + 0 * N]);
				FR[i + 0 * N] = 0.5*(E[i + 0 * N] + E[i + 1 + 0 * N]) - speed*(U[i + 1 + 0 * N] - U[i + 0 * N]);
				FL[i + 1 * N] = 0.5*(E[i + 1 * N] + E[i - 1 + 1 * N]) - speed*(U[i + 1 * N] - U[i - 1 + 1 * N]);
				FR[i + 1 * N] = 0.5*(E[i + 1 * N] + E[i + 1 + 1 * N]) - speed*(U[i + 1 + 1 * N] - U[i + 1 * N]);
				FL[i + 2 * N] = 0.5*(E[i + 2 * N] + E[i - 1 + 2 * N]) - speed*(U[i + 2 * N] - U[i - 1 + 2 * N]);
				FR[i + 2 * N] = 0.5*(E[i + 2 * N] + E[i + 1 + 2 * N]) - speed*(U[i + 1 + 2 * N] - U[i + 2 * N]);
				FL[i + 3 * N] = 0.5*(E[i + 3 * N] + E[i - 1 + 3 * N]) - speed*(U[i + 3 * N] - U[i - 1 + 3 * N]);
				FR[i + 3 * N] = 0.5*(E[i + 3 * N] + E[i + 1 + 3 * N]) - speed*(U[i + 1 + 3 * N] - U[i + 3 * N]);
				FL[i + 4 * N] = 0.5*(E[i + 4 * N] + E[i - 1 + 4 * N]) - speed*(U[i + 4 * N] - U[i - 1 + 4 * N]);
				FR[i + 4 * N] = 0.5*(E[i + 4 * N] + E[i + 1 + 4 * N]) - speed*(U[i + 1 + 4 * N] - U[i + 4 * N]);

				FB[i + 0 * N] = 0.5*(F[i + 0 * N] + F[i - NX + 0 * N]) - speed*(U[i + 0 * N] - U[i - NX + 0 * N]);
				FF[i + 0 * N] = 0.5*(F[i + 0 * N] + F[i + NX + 0 * N]) - speed*(U[i + NX + 0 * N] - U[i + 0 * N]);
				FB[i + 1 * N] = 0.5*(F[i + 1 * N] + F[i - NX + 1 * N]) - speed*(U[i + 1 * N] - U[i - NX + 1 * N]);
				FF[i + 1 * N] = 0.5*(F[i + 1 * N] + F[i + NX + 1 * N]) - speed*(U[i + NX + 1 * N] - U[i + 1 * N]);
				FB[i + 2 * N] = 0.5*(F[i + 2 * N] + F[i - NX + 2 * N]) - speed*(U[i + 2 * N] - U[i - NX + 2 * N]);
				FF[i + 2 * N] = 0.5*(F[i + 2 * N] + F[i + NX + 2 * N]) - speed*(U[i + NX + 2 * N] - U[i + 2 * N]);
				FB[i + 3 * N] = 0.5*(F[i + 3 * N] + F[i - NX + 3 * N]) - speed*(U[i + 3 * N] - U[i - NX + 3 * N]);
				FF[i + 3 * N] = 0.5*(F[i + 3 * N] + F[i + NX + 3 * N]) - speed*(U[i + NX + 3 * N] - U[i + 3 * N]);
				FB[i + 4 * N] = 0.5*(F[i + 4 * N] + F[i - NX + 4 * N]) - speed*(U[i + 4 * N] - U[i - NX + 4 * N]);
				FF[i + 4 * N] = 0.5*(F[i + 4 * N] + F[i + NX + 4 * N]) - speed*(U[i + NX + 4 * N] - U[i + 4 * N]);

				FD[i + 0 * N] = 0.5*(G[i + 0 * N] + G[i - NX*NY + 0 * N]) - speed*(U[i + 0 * N] - U[i - NX*NY + 0 * N]);
				FU[i + 0 * N] = 0.5*(G[i + 0 * N] + G[i + NX*NY + 0 * N]) - speed*(U[i + NX*NY + 0 * N] - U[i + 0 * N]);
				FD[i + 1 * N] = 0.5*(G[i + 1 * N] + G[i - NX*NY + 1 * N]) - speed*(U[i + 1 * N] - U[i - NX*NY + 1 * N]);
				FU[i + 1 * N] = 0.5*(G[i + 1 * N] + G[i + NX*NY + 1 * N]) - speed*(U[i + NX*NY + 1 * N] - U[i + 1 * N]);
				FD[i + 2 * N] = 0.5*(G[i + 2 * N] + G[i - NX*NY + 2 * N]) - speed*(U[i + 2 * N] - U[i - NX*NY + 2 * N]);
				FU[i + 2 * N] = 0.5*(G[i + 2 * N] + G[i + NX*NY + 2 * N]) - speed*(U[i + NX*NY + 2 * N] - U[i + 2 * N]);
				FD[i + 3 * N] = 0.5*(G[i + 3 * N] + G[i - NX*NY + 3 * N]) - speed*(U[i + 3 * N] - U[i - NX*NY + 3 * N]);
				FU[i + 3 * N] = 0.5*(G[i + 3 * N] + G[i + NX*NY + 3 * N]) - speed*(U[i + NX*NY + 3 * N] - U[i + 3 * N]);
				FD[i + 4 * N] = 0.5*(G[i + 4 * N] + G[i - NX*NY + 4 * N]) - speed*(U[i + 4 * N] - U[i - NX*NY + 4 * N]);
				FU[i + 4 * N] = 0.5*(G[i + 4 * N] + G[i + NX*NY + 4 * N]) - speed*(U[i + NX*NY + 4 * N] - U[i + 4 * N]);
			
				// revise body condition when it is near air
				if (d_body[(i - 1)] == 0.0) { // left is body
					// change sign
					E[(i - 1) + 0 * N] = -E[i + 0 * N];
					E[(i - 1) + 4 * N] = -E[i + 4 * N];
					U[(i - 1) + 0 * N] = U[i + 0 * N];
					U[(i - 1) + 4 * N] = U[i + 4 * N];

					E[(i - 1) + 1 * N] = E[i + 1 * N];
					E[(i - 1) + 2 * N] = E[i + 2 * N];
					E[(i - 1) + 3 * N] = E[i + 3 * N];
					U[(i - 1) + 1 * N] = -U[i + 1 * N];
					U[(i - 1) + 2 * N] = -U[i + 2 * N];
					U[(i - 1) + 3 * N] = -U[i + 3 * N];

					FL[i + 0 * N] = 0.5*(E[i + 0 * N] + E[i - 1 + 0 * N]) - speed*(U[i + 0 * N] - U[i - 1 + 0 * N]);
					FL[i + 1 * N] = 0.5*(E[i + 1 * N] + E[i - 1 + 1 * N]) - speed*(U[i + 1 * N] - U[i - 1 + 1 * N]);
					FL[i + 2 * N] = 0.5*(E[i + 2 * N] + E[i - 1 + 2 * N]) - speed*(U[i + 2 * N] - U[i - 1 + 2 * N]);
					FL[i + 3 * N] = 0.5*(E[i + 3 * N] + E[i - 1 + 3 * N]) - speed*(U[i + 3 * N] - U[i - 1 + 3 * N]);
					FL[i + 4 * N] = 0.5*(E[i + 4 * N] + E[i - 1 + 4 * N]) - speed*(U[i + 4 * N] - U[i - 1 + 4 * N]);
				}

				if (d_body[(i + 1)] == 0.0) { // right is body
					// change sign
					E[(i + 1) + 0 * N] = -E[i + 0 * N];
					E[(i + 1) + 4 * N] = -E[i + 4 * N];
					U[(i + 1) + 0 * N] = U[i + 0 * N];
					U[(i + 1) + 4 * N] = U[i + 4 * N];

					E[(i + 1) + 1 * N] = E[i + 1 * N];
					E[(i + 1) + 2 * N] = E[i + 2 * N];
					E[(i + 1) + 3 * N] = E[i + 3 * N];
					U[(i + 1) + 1 * N] = -U[i + 1 * N];
					U[(i + 1) + 2 * N] = -U[i + 2 * N];
					U[(i + 1) + 3 * N] = -U[i + 3 * N];

					FR[i + 0 * N] = 0.5*(E[i + 0 * N] + E[i + 1 + 0 * N]) - speed*(U[i + 1 + 0 * N] - U[i + 0 * N]);
					FR[i + 1 * N] = 0.5*(E[i + 1 * N] + E[i + 1 + 1 * N]) - speed*(U[i + 1 + 1 * N] - U[i + 1 * N]);
					FR[i + 2 * N] = 0.5*(E[i + 2 * N] + E[i + 1 + 2 * N]) - speed*(U[i + 1 + 2 * N] - U[i + 2 * N]);
					FR[i + 3 * N] = 0.5*(E[i + 3 * N] + E[i + 1 + 3 * N]) - speed*(U[i + 1 + 3 * N] - U[i + 3 * N]);
					FR[i + 4 * N] = 0.5*(E[i + 4 * N] + E[i + 1 + 4 * N]) - speed*(U[i + 1 + 4 * N] - U[i + 4 * N]);
				}

				if (d_body[i - NX] == 0.0) { // back is body
					// change sign
					F[(i - NX) + 0 * N] = -F[i + 0 * N];
					F[(i - NX) + 4 * N] = -F[i + 4 * N];
					U[(i - NX) + 0 * N] = U[i + 0 * N];
					U[(i - NX) + 4 * N] = U[i + 4 * N];

					F[(i - NX) + 1 * N] = F[i + 1 * N];
					F[(i - NX) + 2 * N] = F[i + 2 * N];
					F[(i - NX) + 3 * N] = F[i + 3 * N];
					U[(i - NX) + 1 * N] = -U[i + 1 * N];
					U[(i - NX) + 2 * N] = -U[i + 2 * N];
					U[(i - NX) + 3 * N] = -U[i + 3 * N];

					FB[i + 0 * N] = 0.5*(F[i + 0 * N] + F[i - NX + 0 * N]) - speed*(U[i + 0 * N] - U[i - NX + 0 * N]);
					FB[i + 1 * N] = 0.5*(F[i + 1 * N] + F[i - NX + 1 * N]) - speed*(U[i + 1 * N] - U[i - NX + 1 * N]);
					FB[i + 2 * N] = 0.5*(F[i + 2 * N] + F[i - NX + 2 * N]) - speed*(U[i + 2 * N] - U[i - NX + 2 * N]);
					FB[i + 3 * N] = 0.5*(F[i + 3 * N] + F[i - NX + 3 * N]) - speed*(U[i + 3 * N] - U[i - NX + 3 * N]);
					FB[i + 4 * N] = 0.5*(F[i + 4 * N] + F[i - NX + 4 * N]) - speed*(U[i + 4 * N] - U[i - NX + 4 * N]);
				}

				if (d_body[i + NX] == 0.0) { // front is body
					// change sign
					F[(i + NX) + 0 * N] = -F[i + 0 * N];
					F[(i + NX) + 4 * N] = -F[i + 4 * N];
					U[(i + NX) + 0 * N] = U[i + 0 * N];
					U[(i + NX) + 4 * N] = U[i + 4 * N];

					F[(i + NX) + 1 * N] = F[i + 1 * N];
					F[(i + NX) + 2 * N] = F[i + 2 * N];
					F[(i + NX) + 3 * N] = F[i + 3 * N];
					U[(i + NX) + 1 * N] = -U[i + 1 * N];
					U[(i + NX) + 2 * N] = -U[i + 2 * N];
					U[(i + NX) + 3 * N] = -U[i + 3 * N];

					FF[i + 0 * N] = 0.5*(F[i + 0 * N] + F[i + NX + 0 * N]) - speed*(U[i + NX + 0 * N] - U[i + 0 * N]);
					FF[i + 1 * N] = 0.5*(F[i + 1 * N] + F[i + NX + 1 * N]) - speed*(U[i + NX + 1 * N] - U[i + 1 * N]);
					FF[i + 2 * N] = 0.5*(F[i + 2 * N] + F[i + NX + 2 * N]) - speed*(U[i + NX + 2 * N] - U[i + 2 * N]);
					FF[i + 3 * N] = 0.5*(F[i + 3 * N] + F[i + NX + 3 * N]) - speed*(U[i + NX + 3 * N] - U[i + 3 * N]);
					FF[i + 4 * N] = 0.5*(F[i + 4 * N] + F[i + NX + 4 * N]) - speed*(U[i + NX + 4 * N] - U[i + 4 * N]);
				}

				if (d_body[i + (-1)*NX*NY] == 0.0) { // down is body
					G[i + (-1)*NX*NY + 0 * N] = -G[(i)+0 * N];
					U[i + (-1)*NX*NY + 0 * N] = U[(i)+0 * N];
					G[i + (-1)*NX*NY + 4 * N] = -G[(i)+4 * N];
					U[i + (-1)*NX*NY + 4 * N] = U[(i)+4 * N];

					G[i + (-1)*NX*NY + 1 * N] = G[(i)+1 * N];
					U[i + (-1)*NX*NY + 1 * N] = -U[(i)+1 * N];
					G[i + (-1)*NX*NY + 2 * N] = G[(i)+2 * N];
					U[i + (-1)*NX*NY + 2 * N] = -U[(i)+2 * N];
					G[i + (-1)*NX*NY + 3 * N] = G[(i)+3 * N];
					U[i + (-1)*NX*NY + 3 * N] = -U[(i)+3 * N];

					FD[i + 0 * N] = 0.5*(G[i + 0 * N] + G[i + (-1)*NX*NY + 0 * N]) - speed*(U[i + 0 * N] - U[i + (-1)*NX*NY + 0 * N]);
					FD[i + 1 * N] = 0.5*(G[i + 1 * N] + G[i + (-1)*NX*NY + 1 * N]) - speed*(U[i + 1 * N] - U[i + (-1)*NX*NY + 1 * N]);
					FD[i + 2 * N] = 0.5*(G[i + 2 * N] + G[i + (-1)*NX*NY + 2 * N]) - speed*(U[i + 2 * N] - U[i + (-1)*NX*NY + 2 * N]);
					FD[i + 3 * N] = 0.5*(G[i + 3 * N] + G[i + (-1)*NX*NY + 3 * N]) - speed*(U[i + 3 * N] - U[i + (-1)*NX*NY + 3 * N]);
					FD[i + 4 * N] = 0.5*(G[i + 4 * N] + G[i + (-1)*NX*NY + 4 * N]) - speed*(U[i + 4 * N] - U[i + (-1)*NX*NY + 4 * N]);
				}

				if (d_body[i + (1)*NX*NY] == 0.0) { // up is body
					G[i + (1)*NX*NY + 0 * N] = -G[(i)+0 * N];
					U[i + (1)*NX*NY + 0 * N] = U[(i)+0 * N];
					G[i + (1)*NX*NY + 4 * N] = -G[(i)+4 * N];
					U[i + (1)*NX*NY + 4 * N] = U[(i)+4 * N];

					G[i + (1)*NX*NY + 1 * N] = G[(i)+1 * N];
					U[i + (1)*NX*NY + 1 * N] = -U[(i)+1 * N];
					G[i + (1)*NX*NY + 2 * N] = G[(i)+2 * N];
					U[i + (1)*NX*NY + 2 * N] = -U[(i)+2 * N];
					G[i + (1)*NX*NY + 3 * N] = G[(i)+3 * N];
					U[i + (1)*NX*NY + 3 * N] = -U[(i)+3 * N];

					FU[i + 0 * N] = 0.5*(G[i + 0 * N] + G[i + (1)*NX*NY + 0 * N]) - speed*(U[i + (1)*NX*NY + 0 * N] - U[i + 0 * N]);
					FU[i + 1 * N] = 0.5*(G[i + 1 * N] + G[i + (1)*NX*NY + 1 * N]) - speed*(U[i + (1)*NX*NY + 1 * N] - U[i + 1 * N]);
					FU[i + 2 * N] = 0.5*(G[i + 2 * N] + G[i + (1)*NX*NY + 2 * N]) - speed*(U[i + (1)*NX*NY + 2 * N] - U[i + 2 * N]);
					FU[i + 3 * N] = 0.5*(G[i + 3 * N] + G[i + (1)*NX*NY + 3 * N]) - speed*(U[i + (1)*NX*NY + 3 * N] - U[i + 3 * N]);
					FU[i + 4 * N] = 0.5*(G[i + 4 * N] + G[i + (1)*NX*NY + 4 * N]) - speed*(U[i + (1)*NX*NY + 4 * N] - U[i + 4 * N]);
							}
						}
					}
				}
			}
		}
	}
	

	// Update U by U_new using FVM (Rusanov Flux)
	if (i < N) {
		if (d_body[i] == -1.0) { // air
			if ( i % NX != 0 && i % NX != (NX - 1)) {
				if( i % (NX*NY) >= NX && i % (NX*NY) < NX*(NY - 1)){
					if( i > (NX*NY - 1) && i < NX*NY*(NZ - 1)){
				// ((i % 100) != 0) && ((i % 100) != 99) && (i % 10000 >= 100) && (i % 10000 <= 9899) && (i >= 10000) && (i <= 989999) 
				// delete left plane, delete right plane, delete back plane, delete front plane, delete down plane, delete up plane 
				U_new[i + 0 * N] = U[i + 0 * N] - (DT / DX)*(FR[i + 0 * N] - FL[i + 0 * N])
					- (DT / DY)*(FF[i + 0 * N] - FB[i + 0 * N]) - (DT / DZ)*(FU[i + 0 * N] - FD[i + 0 * N]);
				U_new[i + 1 * N] = U[i + 1 * N] - (DT / DX)*(FR[i + 1 * N] - FL[i + 1 * N])
					- (DT / DY)*(FF[i + 1 * N] - FB[i + 1 * N]) - (DT / DZ)*(FU[i + 1 * N] - FD[i + 1 * N]);
				U_new[i + 2 * N] = U[i + 2 * N] - (DT / DX)*(FR[i + 2 * N] - FL[i + 2 * N])
					- (DT / DY)*(FF[i + 2 * N] - FB[i + 2 * N]) - (DT / DZ)*(FU[i + 2 * N] - FD[i + 2 * N]);
				U_new[i + 3 * N] = U[i + 3 * N] - (DT / DX)*(FR[i + 3 * N] - FL[i + 3 * N])
					- (DT / DY)*(FF[i + 3 * N] - FB[i + 3 * N]) - (DT / DZ)*(FU[i + 3 * N] - FD[i + 3 * N]);
				U_new[i + 4 * N] = U[i + 4 * N] - (DT / DX)*(FR[i + 4 * N] - FL[i + 4 * N])
					- (DT / DY)*(FF[i + 4 * N] - FB[i + 4 * N]) - (DT / DZ)*(FU[i + 4 * N] - FD[i + 4 * N]);
					}		
				}
			}
		}
	}
}
__global__ void CalculateFlux3(float* d_body, float* dens, float* xv, float* yv, float* zv,float* press,float* U, float* U_new){

        int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < N) {
		//Renew down boundary condition
		if ( i < (NX*NY - 1) && i%NX > 0){
			if( i%NX < (NX - 1) && i % (NX*NY) > NX){
				if( i % (NX*NY) < NX*(NY - 1)){
			// (i <= 9999-1) && (i%100 >= 1) && (i%100 <= 98) && (i%10000 >= 101) && (i%10000 <= 9899)
			// U_new[i] of down boundary = U_new[i+NX*NY]
			U_new[i + 0 * N] = U_new[i + NX * NY + 0 * N];
			U_new[i + 1 * N] = U_new[i + NX * NY + 1 * N];
			U_new[i + 2 * N] = U_new[i + NX * NY + 2 * N];
			U_new[i + 3 * N] = U_new[i + NX * NY + 3 * N];
			U_new[i + 4 * N] = U_new[i + NX * NY + 4 * N];
				}
			}
		}
		//Renew up boundary condition
		else if ( i > (NX*NY*(NZ - 1) - 1) && i%NX > 0 ) {
			if ( i%NX < (NX - 1) && i % (NX*NY) > NX){
				if(i % (NX*NY) < NX*(NY - 1)){
			// (i >= 999900) && (i%100 >= 1) && (i%100 <= 98) && (i%10000 >= 101) && (i%10000 <= 9899)
			// U_new[i] of up boundary = U_new[i-NX*NY]
			U_new[i + 0 * N] = U_new[i - NX * NY + 0 * N];
			U_new[i + 1 * N] = U_new[i - NX * NY + 1 * N];
			U_new[i + 2 * N] = U_new[i - NX * NY + 2 * N];
			U_new[i + 3 * N] = U_new[i - NX * NY + 3 * N];
			U_new[i + 4 * N] = U_new[i - NX * NY + 4 * N];
				}
			}
		}
	}

	if (i < N) {
		//Renew left boundary condition
		if ( i%NX == 0 && i % (NX*NY) > (NX - 1)) {
			if( i % (NX*NY) < NX*(NY - 1)){
			// (i%100 == 0) && (i%10000 >= 100) && (i%10000 <= 9899)
			// U_new[i] of left boundary = U_new[i+1]
			U_new[i + 0 * N] = U_new[i + 1 + 0 * N];
			U_new[i + 1 * N] = U_new[i + 1 + 1 * N];
			U_new[i + 2 * N] = U_new[i + 1 + 2 * N];
			U_new[i + 3 * N] = U_new[i + 1 + 3 * N];
			U_new[i + 4 * N] = U_new[i + 1 + 4 * N];
			}
		}
		//Renew right boundary condition
		else if ( i%NX == (NX - 1) &&  i % (NX*NY) > (NX - 1)) {
			if( i % (NX*NY) < NX*(NY - 1)){
			// (i%100 == 99) && (i%10000 >= 100) && (i%10000 <= 9899)
			// U_new[i] of right boundary = U_new[i-1]
			U_new[i + 0 * N] = U_new[i - 1 + 0 * N];
			U_new[i + 1 * N] = U_new[i - 1 + 1 * N];
			U_new[i + 2 * N] = U_new[i - 1 + 2 * N];
			U_new[i + 3 * N] = U_new[i - 1 + 3 * N];
			U_new[i + 4 * N] = U_new[i - 1 + 4 * N];
			}
		}
		// Renew back boundary condition
		else if ( i % (NX*NY) < NX && i%NX > 0) {
			if(i%NX < (NX - 1)){
			// (i%10000 <= 99) && (i%100 >= 1) && (i%100 <= 98)
			// U_new[i] of back boundary = U_new[i+NX]
			U_new[i + 0 * N] = U_new[i + NX + 0 * N];
			U_new[i + 1 * N] = U_new[i + NX + 1 * N];
			U_new[i + 2 * N] = U_new[i + NX + 2 * N];
			U_new[i + 3 * N] = U_new[i + NX + 3 * N];
			U_new[i + 4 * N] = U_new[i + NX + 4 * N];
			}
		}
		// Renew front boundary condition
		else if ( i % (NX*NY) > NX*(NY - 1) - 1 && i%NX > 0) {
			if(i%NX < (NX - 1)){
			// (i%10000 >= 9900) && (i%100 >= 1) && (i%100 <= 98)
			// U_new[i] of front boundary = U_new[i-NX]
			U_new[i + 0 * N] = U_new[i - NX + 0 * N];
			U_new[i + 1 * N] = U_new[i - NX + 1 * N];
			U_new[i + 2 * N] = U_new[i - NX + 2 * N];
			U_new[i + 3 * N] = U_new[i - NX + 3 * N];
			U_new[i + 4 * N] = U_new[i - NX + 4 * N];
			}
		}
	}

	if (i < N && i > 1) {
		// edge
		// left back 
		if ((i % (NX*NY) == 0)) {
			// (i%10000 == 0)
			// U_new[i] = U_new[i] of right 
			U_new[i + 0 * N] = U_new[i + 1 + 0 * N];
			U_new[i + 1 * N] = U_new[i + 1 + 1 * N];
			U_new[i + 2 * N] = U_new[i + 1 + 2 * N];
			U_new[i + 3 * N] = U_new[i + 1 + 3 * N];
			U_new[i + 4 * N] = U_new[i + 1 + 4 * N];
		}
		// right back 
		else if ((i % (NX*NY) == (NX - 1))) {
			// (i%10000 == 99)
			// U_new[i] of front boundary = U_new[i-NX]
			U_new[i + 0 * N] = U_new[i - 1 + 0 * N];
			U_new[i + 1 * N] = U_new[i - 1 + 1 * N];
			U_new[i + 2 * N] = U_new[i - 1 + 2 * N];
			U_new[i + 3 * N] = U_new[i - 1 + 3 * N];
			U_new[i + 4 * N] = U_new[i - 1 + 4 * N];
		}
		// left front 
		else if ((i % (NX*NY) == (NX*(NY - 1)))) {
			// (i%10000 == 9900)
			// U_new[i] of front boundary = U_new[i-NX]
			U_new[i + 0 * N] = U_new[i + 1 + 0 * N];
			U_new[i + 1 * N] = U_new[i + 1 + 1 * N];
			U_new[i + 2 * N] = U_new[i + 1 + 2 * N];
			U_new[i + 3 * N] = U_new[i + 1 + 3 * N];
			U_new[i + 4 * N] = U_new[i + 1 + 4 * N];
		}
		// right front 
		else if ((i % (NX*NY) == (NX*NY - 1))) {
			// (i%10000 == 9999)
			// U_new[i] of front boundary = U_new[i-NX]
			U_new[i + 0 * N] = U_new[i - 1 + 0 * N];
			U_new[i + 1 * N] = U_new[i - 1 + 1 * N];
			U_new[i + 2 * N] = U_new[i - 1 + 2 * N];
			U_new[i + 3 * N] = U_new[i - 1 + 3 * N];
			U_new[i + 4 * N] = U_new[i - 1 + 4 * N];
		}
	}

	// Update density, velocity, pressure, and U
	if (i < N) {
		if (d_body[i] == -1.0) { // air
			dens[i] = U_new[i + 0 * N];
			xv[i] = U_new[i + 1 * N] / dens[i];
			yv[i] = U_new[i + 2 * N] / dens[i];
			zv[i] = U_new[i + 3 * N] / dens[i];
			press[i] = (GAMA - 1) * (U_new[i + 4 * N] - 0.5 * dens[i] * (xv[i] * xv[i] + yv[i] * yv[i] + zv[i] * zv[i]));
			U[i + 0 * N] = U_new[i + 0 * N];
			U[i + 1 * N] = U_new[i + 1 * N];
			U[i + 2 * N] = U_new[i + 2 * N];
			U[i + 3 * N] = U_new[i + 3 * N];
			U[i + 4 * N] = U_new[i + 4 * N];
		}
	}
}

void Call_CalculateFlux() {
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	
	CalculateFlux << <blocksPerGrid, threadsPerBlock >> >(d_body, d_dens, d_xv, d_yv, d_zv, d_press, d_E, d_F, d_G, d_U);
        CalculateFlux2 << <blocksPerGrid, threadsPerBlock >> >(d_body, d_dens, d_xv, d_yv, d_zv, d_press, d_E, d_F, d_G,d_FL,d_FR,d_FB,d_FF,d_FD,d_FU, d_U);
        CalculateFlux3 << <blocksPerGrid, threadsPerBlock >> >(d_body, d_dens, d_xv, d_yv, d_zv, d_press, d_U,d_U_new);

}

void Save_Data_To_File(char *output_file_name) {
	FILE *pOutPutFile;
	pOutPutFile = fopen(output_file_name, "w");
	if (!pOutPutFile) { printf("Open failure"); }

	fprintf(pOutPutFile, "TITLE=\"Flow Field of X-37\"\n");
	fprintf(pOutPutFile, "VARIABLES=\"X\", \"Y\", \"Z\", \"U\", \"V\", \"W\", \"Pressure\", \"Temperature\", \"Body\"\n");
	fprintf(pOutPutFile, "ZONE I = 100, J = 100, K = 100, F = POINT\n");

	int i, j, k;
	for (k = 0; k < NZ; k++) {
		for (j = 0; j < NY; j++) {
			for (i = 0; i < NX; i++) {
				temperature[i + j*NX + k*NX*NY] = press[i + j*NX + k*NX*NY] / (dens[i + j*NX + k*NX*NY] * R);
				/* ...test body...*/
				//fprintf(pOutPutFile, "%d %d %d %f\n", i, j, k, h_body[i + j*NX + k*NX*NY]);
				/* ...test body...*/
				fprintf(pOutPutFile, "%d %d %d %f %f %f %f %f %1.0f\n", i, j, k, xv[i + j*NX + k*NX*NY], yv[i + j*NX + k*NX*NY], zv[i + j*NX + k*NX*NY], press[i + j*NX + k*NX*NY], temperature[i + j*NX + k*NX*NY], -h_body[i + j*NX + k*NX*NY]);
			}
		}
	}
}
