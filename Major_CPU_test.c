#include <stdio.h>
#include <math.h>

#define NX 100                          // No. of cells in x direction
#define NY 100                          // No. of cells in y direction
#define NZ 100                          // No. of cells in z direction
#define N (NX*NY*NZ)            // N = total number of cells in domain
#define USE_INTRINSICS 1        // 1 = Use AVX Intrinsics, 2 = Use standard code
#define L 100                             // L = length of domain (m)
#define H 100                             // H = Height of domain (m)
#define W 100                             // W = Width of domain (m)
#define DX (L/NX)                       // DX, DY, DZ = grid spacing in x,y,z.
#define DY (H/NY)
#define DZ (W/NZ)
#define DT 0.001                       // Time step (seconds)
#define Travelheight
#define Travelspeed
#define NO_STEPS 10

#define R (1.0)           // Dimensionless specific gas constant
#define GAMA (7.0/5.0)    // Ratio of specific heats
#define CV (R/(GAMA-1.0)) // Cv
#define CP (CV + R)       // Cp

//#define DEBUG_VALUE

void Allocate_Memory();
void Load_Dat_To_Array(char *input_file_name, float *body);
void Init();
void CPUHeatContactFunction();
void CalRenewResult();
#ifdef GPU
void Call_GPUHeatContactFunction();
void Call_GPUTimeStepFunction();
void Send_To_Device();
void Get_From_Device();
#endif
void Free_Memory();
void Save_Data_To_File(char *output_file_name);
float *dens;              //density
float *temperature;        //temperature
float *xv;                //velocity in x
float *yv;                //velocity in y
float *zv;                //velocity in z
float *press;             //pressure

float *d_dens;              //density 
float *d_temperature;       //temperature
float *d_xv;                //velocity in x
float *d_yv;                //velocity in y
float *d_zv;                //velocity in z
float *d_press;             //pressure

float *U;
float *U_new;
float *E;
float *F;
float *G;
float *FF;
float *FB;
float *FR;
float *FL;
float *FU;
float *FD;

float *h_body;
float *d_body;

int total_cells = 0;            // A counter for computed cells

int main() {
	int t;
	// Need to allocate memory first
	Allocate_Memory();

	char *input_file_name = "Export_50x50x50.dat";
	Load_Dat_To_Array(input_file_name, h_body);
	Init();
	for (t = 0; t < NO_STEPS; t++) {
		CPUHeatContactFunction();
		CalRenewResult();
	}

	char *output_file_name = "3DResults.dat";
	Save_Data_To_File(output_file_name);

	//// GPU code
	// Send to device
	//Send_To_Device();
	// Call our function
	//Call_GPU_Function();
	// Get from device
	//Get_From_Device();
	// Print out the first 5 values of b

	// Free the memory
	Free_Memory();
	return 0;
}


void Allocate_Memory() {
	size_t size = N*sizeof(float);
	dens = (float*)malloc(size);
	temperature = (float*)malloc(size);
	xv = (float*)malloc(size);
	yv = (float*)malloc(size);
	zv = (float*)malloc(size);
	press = (float*)malloc(size);

#ifdef GPU
	cudaError_t Error;
	Error = cudaMalloc((void**)&d_dens, size);
	printf("CUDA error (malloc d_dens) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_temperature, size);
	printf("CUDA error (malloc d_temperature) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_xv, size);
	printf("CUDA error (malloc d_xv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_yv, size);
	printf("CUDA error (malloc d_yv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_zv, size);
	printf("CUDA error (malloc d_zv) = %s\n", cudaGetErrorString(Error));
	Error = cudaMalloc((void**)&d_press, size);
	printf("CUDA error (malloc d_press) = %s\n", cudaGetErrorString(Error));
#endif
	size_t size2 = N*sizeof(float)* 5;
	E = (float*)malloc(size2);
	F = (float*)malloc(size2);
	G = (float*)malloc(size2);
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
	Error = cudaMalloc((void**)&G, size2);
	printf("CUDA error (malloc G) = %s\n",cudaGetErrorString(Error));
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

	size_t size3 = N*sizeof(float);
	h_body = (float*)malloc(size3);
#ifdef GPU
	Error = cudaMalloc((void**)&d_body, size3);
	printf("CUDA error (malloc d_body) = %s\n", cudaGetErrorString(Error));
#endif
}

void Load_Dat_To_Array(char* input_file_name, float* body) {
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
	char line[60];

	// According to the 50x50x50 order
	for (x = 75; x > 25; x--) {
		for (z = 25; z < 75; z++) {
			for (y = 25; y < 75; y++) {
				idx = z * NX * NY + y * NX + x;
				fscanf(pFile, "%f%f%f%f", &tmp1, &tmp2, &tmp3, &body[idx]);
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
	int i, j, k, z;
	for (i = 0; i < NX; i++) {
		for (j = 0; j < NY; j++) {
			for (k = 0; k < NZ; k++) {
				if (!h_body[i + j*NX + k*NX*NY]) {
					// body
					dens[i + j*NX + k*NX*NY] = 1.0;
					xv[i + j*NX + k*NX*NY] = 1.0;
					yv[i + j*NX + k*NX*NY] = 1.0;
					zv[i + j*NX + k*NX*NY] = 1.0;;
					press[i + j*NX + k*NX*NY] = 1.0;
					temperature[i + j*NX + k*NX*NY] = 273;
				}
				else {
					// air reference: http://www.engineeringtoolbox.com/standard-atmosphere-d_604.html
					dens[i + j*NX + k*NX*NY] = 0.001 * 0.1;				// unit: kg / m^3
					xv[i + j*NX + k*NX*NY] = 5362;						// unit: m / s
					yv[i + j*NX + k*NX*NY] = 0.01;						// unit: m / s
					zv[i + j*NX + k*NX*NY] = 4450;						// unit: m / s
					press[i + j*NX + k*NX*NY] = 0.00052 * 100000;		// unit: (kg*m/s^2) / m^2
					temperature[i + j*NX + k*NX*NY] = -53 + 273.15;		// unit: K
				}
			}
		}
	}

	for (i = 0; i < NX; i++) {
		for (j = 0; j < NY; j++) {
			for (k = 0; k < NZ; k++) {
				if (h_body[i + j*NX + k*NX*NY] != 0.0) {
					// air
					U[i + j*NX + k*NX*NY + 0 * N] = dens[i + j*NX + k*NX*NY];
					U[i + j*NX + k*NX*NY + 1 * N] = dens[i + j*NX + k*NX*NY] * (xv[i + j*NX + k*NX*NY]);
					U[i + j*NX + k*NX*NY + 2 * N] = dens[i + j*NX + k*NX*NY] * (yv[i + j*NX + k*NX*NY]);
					U[i + j*NX + k*NX*NY + 3 * N] = dens[i + j*NX + k*NX*NY] * (zv[i + j*NX + k*NX*NY]);
					U[i + j*NX + k*NX*NY + 4 * N] = dens[i + j*NX + k*NX*NY] * (CV*(press[i + j*NX + k*NX*NY] / dens[i + j*NX + k*NX*NY] / R)
						+ 0.5*((xv[i + j*NX + k*NX*NY] * xv[i + j*NX + k*NX*NY]) + (yv[i + j*NX + k*NX*NY] * yv[i + j*NX + k*NX*NY])
						+ (zv[i + j*NX + k*NX*NY] * zv[i + j*NX + k*NX*NY])));
				}
				else {
					// body
					for (z = 0; z < 5; z++) {
						U[i + j*NX + k*NX*NY + z * N] = 1.0;
					}
				}
			}
		}
	}
}

void CPUHeatContactFunction() {
	int i, j, k, z;
	for (k = 0; k < NZ; k++) {
		for (j = 0; j < NY; j++) {
			for (i = 0; i < NX; i++) {
				if (h_body[i + j*NX + k*NX*NY] != 0.0) {
					// air
					E[i + j*NX + k*NX*NY + 0 * N] = dens[i + j*NX + k*NX*NY] * xv[i + j*NX + k*NX*NY];
					E[i + j*NX + k*NX*NY + 1 * N] = dens[i + j*NX + k*NX*NY] * xv[i + j*NX + k*NX*NY] * xv[i + j*NX + k*NX*NY] + press[i + j*NX + k*NX*NY];
					E[i + j*NX + k*NX*NY + 2 * N] = dens[i + j*NX + k*NX*NY] * xv[i + j*NX + k*NX*NY] * yv[i + j*NX + k*NX*NY];
					E[i + j*NX + k*NX*NY + 3 * N] = dens[i + j*NX + k*NX*NY] * xv[i + j*NX + k*NX*NY] * zv[i + j*NX + k*NX*NY];
					E[i + j*NX + k*NX*NY + 4 * N] = xv[i + j*NX + k*NX*NY] * (U[i + j*NX + k*NX*NY + 4 * N] + press[i + j*NX + k*NX*NY]);

					F[i + j*NX + k*NX*NY + 0 * N] = dens[i + j*NX + k*NX*NY] * yv[i + j*NX + k*NX*NY];
					F[i + j*NX + k*NX*NY + 1 * N] = dens[i + j*NX + k*NX*NY] * xv[i + j*NX + k*NX*NY] * yv[i + j*NX + k*NX*NY];
					F[i + j*NX + k*NX*NY + 2 * N] = dens[i + j*NX + k*NX*NY] * yv[i + j*NX + k*NX*NY] * yv[i + j*NX + k*NX*NY] + press[i + j*NX + k*NX*NY];
					F[i + j*NX + k*NX*NY + 3 * N] = dens[i + j*NX + k*NX*NY] * yv[i + j*NX + k*NX*NY] * zv[i + j*NX + k*NX*NY];
					F[i + j*NX + k*NX*NY + 4 * N] = yv[i + j*NX + k*NX*NY] * (U[i + j*NX + k*NX*NY + 4 * N] + press[i + j*NX + k*NX*NY]);

					G[i + j*NX + k*NX*NY + 0 * N] = dens[i + j*NX + k*NX*NY] * zv[i + j*NX + k*NX*NY];
					G[i + j*NX + k*NX*NY + 1 * N] = dens[i + j*NX + k*NX*NY] * xv[i + j*NX + k*NX*NY] * zv[i + j*NX + k*NX*NY];
					G[i + j*NX + k*NX*NY + 2 * N] = dens[i + j*NX + k*NX*NY] * yv[i + j*NX + k*NX*NY] * zv[i + j*NX + k*NX*NY];
					G[i + j*NX + k*NX*NY + 3 * N] = dens[i + j*NX + k*NX*NY] * zv[i + j*NX + k*NX*NY] * zv[i + j*NX + k*NX*NY] + press[i + j*NX + k*NX*NY];
					G[i + j*NX + k*NX*NY + 4 * N] = zv[i + j*NX + k*NX*NY] * (U[i + j*NX + k*NX*NY + 4 * N] + press[i + j*NX + k*NX*NY]);

#ifdef DEBUG_VALUE
					/* ...test... */
					if (E[i + j*NX + k*NX*NY + 1 * N] > 1000.0) {
						int z;
						for (z = 0; z < 5; z++) {
							printf("xv[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, xv[i + j*NX + k*NX*NY]);
							printf("E[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, E[i + j*NX + k*NX*NY + z*N]);
							printf("F[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, F[i + j*NX + k*NX*NY + z*N]);
							printf("G[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, G[i + j*NX + k*NX*NY + z*N]);
						}
						system("pause");
					}
#endif // DEBUG_VALUE

				}
				else {
					// body
					for (z = 0; z < 5; z++) {
						E[i + j*NX + k*NX*NY + z * N] = 1.0;
						F[i + j*NX + k*NX*NY + z * N] = 1.0;
						G[i + j*NX + k*NX*NY + z * N] = 1.0;
					}
				}
			}
		}
	}

	float speed = 0.0;
	// Rusanov flux:Left, Right, Up, Down
#pragma omp for //collapse(2) //private(speed)	
	for (k = 1; k < (NZ - 1); k++) {
		for (j = 1; j < (NY - 1); j++) {
			for (i = 1; i < (NX - 1); i++) {
				if (h_body[i + j*NX + k*NX*NY] != 0.0) {
					// air
					speed = sqrt(GAMA*press[i + j*NX + k*NX*NY] / dens[i + j*NX + k*NX*NY]);		// speed of sound in air
					for (z = 0; z < 5; z++) {
						FL[i + j*NX + k*NX*NY + z*N] = 0.5*(E[i + j*NX + k*NX*NY + z*N] + E[(i - 1) + j*NX + k*NX*NY + z*N])
							- speed*(U[i + j*NX + k*NX*NY + z*N] - U[(i - 1) + j*NX + k*NX*NY + z*N]);
						FR[i + j*NX + k*NX*NY + z*N] = 0.5*(E[i + j*NX + k*NX*NY + z*N] + E[(i + 1) + j*NX + k*NX*NY + z*N])
							- speed*(U[(i + 1) + j*NX + k*NX*NY + z*N] - U[i + j*NX + k*NX*NY + z*N]);

						FB[i + j*NX + k*NX*NY + z*N] = 0.5*(F[i + (j - 1)*NX + k*NX*NY + z*N] + F[i + j*NX + k*NX*NY + z*N])
							- speed*(U[i + j*NX + k*NX*NY + z*N] - U[i + (j - 1)*NX + k*NX*NY + z*N]);
						FF[i + j*NX + k*NX*NY + z*N] = 0.5*(F[i + j*NX + k*NX*NY + z*N] + F[i + (j + 1)*NX + k*NX*NY + z*N])
							- speed*(U[i + (j + 1)*NX + k*NX*NY + z*N] - U[i + j*NX + k*NX*NY + z*N]);

						FD[i + j*NX + k*NX*NY + z*N] = 0.5*(G[i + j*NX + k*NX*NY + z*N] + G[i + j*NX + (k - 1)*NX*NY + z*N])
							- speed*(U[i + j*NX + k*NX*NY + z*N] - U[i + j*NX + (k - 1)*NX*NY + z*N]);
						FU[i + j*NX + k*NX*NY + z*N] = 0.5*(G[i + j*NX + k*NX*NY + z*N] + G[i + j*NX + (k + 1)*NX*NY + z*N])
							- speed*(U[i + j*NX + (k + 1)*NX*NY + z*N] - U[i + j*NX + k*NX*NY + z*N]);

#ifdef DEBUG_VALUE
						/* ...test... */
						if (FR[i + j*NX + k*NX*NY + z*N] > 1000 || FR < 0) {
							printf("speed = %f\n", i, j, k, z, speed);
							printf("E[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, E[i + j*NX + k*NX*NY + z*N]);
							printf("F[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, F[i + j*NX + k*NX*NY + z*N]);
							printf("G[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, G[i + j*NX + k*NX*NY + z*N]);
							printf("FR[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, FR[i + j*NX + k*NX*NY + z*N]);
							printf("FL[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, FL[i + j*NX + k*NX*NY + z*N]);
							printf("FF[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, FF[i + j*NX + k*NX*NY + z*N]);
							printf("FB[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, FB[i + j*NX + k*NX*NY + z*N]);
							printf("FU[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, FU[i + j*NX + k*NX*NY + z*N]);
							printf("FD[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, FD[i + j*NX + k*NX*NY + z*N]);
							system("pause");
						}
#endif // DEBUG_VALUE

					}
					
					if (!h_body[(i - 1) + j*NX + k*NX*NY]) {
							FL[i + j*NX + k*NX*NY + 0*N] = 0;
							FL[i + j*NX + k*NX*NY + 1*N] = E[i + j*NX + k*NX*NY + 1*N] + (-2)*speed*U[i + j*NX + k*NX*NY + 1*N]
							FL[i + j*NX + k*NX*NY + 2*N] = (-2)*speed*U[i + j*NX + k*NX*NY + 2*N];
							FL[i + j*NX + k*NX*NY + 3*N] = (-2)*speed*U[i + j*NX + k*NX*NY + 3*N];
							FL[i + j*NX + k*NX*NY + 4*N] = 0;
					}
					else if (!h_body[(i + 1) + j*NX + k*NX*NY) {
							FR[i + j*NX + k*NX*NY + 0*N] = 0;
							FR[i + j*NX + k*NX*NY + 1*N] = E[i + j*NX + k*NX*NY + 1*N] + (-2)*speed*U[i + j*NX + k*NX*NY + 1*N]
							FR[i + j*NX + k*NX*NY + 2*N] = (-2)*speed*U[i + j*NX + k*NX*NY + 2*N];
							FR[i + j*NX + k*NX*NY + 3*N] = (-2)*speed*U[i + j*NX + k*NX*NY + 3*N];
							FR[i + j*NX + k*NX*NY + 4*N] = 0;
					}
					else if (!h_body[i + (j - 1)*NX + k*NX*NY]) {
							FB[i + j*NX + k*NX*NY + 0*N] = 0;
							FB[i + j*NX + k*NX*NY + 1*N] = (-2)*speed*U[i + j*NX + k*NX*NY + 1*N]
							FB[i + j*NX + k*NX*NY + 2*N] = E[i + j*NX + k*NX*NY + 2*N] + (-2)*speed*U[i + j*NX + k*NX*NY + 2*N];
							FB[i + j*NX + k*NX*NY + 3*N] = (-2)*speed*U[i + j*NX + k*NX*NY + 3*N];
							FB[i + j*NX + k*NX*NY + 4*N] = 0;
					}
					else if (!h_body[i + (j + 1)*NX + k*NX*NY]) {
							FF[i + j*NX + k*NX*NY + 0*N] = 0;
							FF[i + j*NX + k*NX*NY + 1*N] = (-2)*speed*U[i + j*NX + k*NX*NY + 1*N]
							FF[i + j*NX + k*NX*NY + 2*N] = E[i + j*NX + k*NX*NY + 2*N] + (-2)*speed*U[i + j*NX + k*NX*NY + 2*N];
							FF[i + j*NX + k*NX*NY + 3*N] = (-2)*speed*U[i + j*NX + k*NX*NY + 3*N];
							FF[i + j*NX + k*NX*NY + 4*N] = 0;
					}
					else if (!h_body[i + j*NX + (k - 1)*NX*NY]) {
							FD[i + j*NX + k*NX*NY + 0*N] = 0;
							FD[i + j*NX + k*NX*NY + 1*N] = (-2)*speed*U[i + j*NX + k*NX*NY + 1*N]
							FD[i + j*NX + k*NX*NY + 2*N] = (-2)*speed*U[i + j*NX + k*NX*NY + 2*N];
							FD[i + j*NX + k*NX*NY + 3*N] = E[i + j*NX + k*NX*NY + 3*N] + (-2)*speed*U[i + j*NX + k*NX*NY + 3*N];
							FD[i + j*NX + k*NX*NY + 4*N] = 0;
					}
					else if (!h_body[i + j*NX + (k + 1)*NX*NY]) {
							FU[i + j*NX + k*NX*NY + 0*N] = 0;
							FU[i + j*NX + k*NX*NY + 1*N] = (-2)*speed*U[i + j*NX + k*NX*NY + 1*N]
							FU[i + j*NX + k*NX*NY + 2*N] = (-2)*speed*U[i + j*NX + k*NX*NY + 2*N];
							FU[i + j*NX + k*NX*NY + 3*N] = E[i + j*NX + k*NX*NY + 3*N] + (-2)*speed*U[i + j*NX + k*NX*NY + 3*N];
							FU[i + j*NX + k*NX*NY + 4*N] = 0;
					}
				}
			}
		}
	}
#pragma omp barrier
}

void CalRenewResult() {
	int i, j, k, z;
	// Update U by FVM
#pragma omp for //collapse(2)
	for (k = 1; k < (NZ - 1); k++) {
		for (j = 1; j < (NY - 1); j++) {
			for (i = 1; i < (NX - 1); i++) {
				if (h_body[i + j*NX + k*NX*NY] != 0.0) {
					// air
					for (z = 0; z < 5; z++) {
						U_new[i + j*NX + k*NX*NY + z*N] = U[i + j*NX + k*NX*NY + z*N] - (DT / DX)*(FR[i + j*NX + k*NX*NY + z*N] - FL[i + j*NX + k*NX*NY + z*N])
							- (DT / DY)*(FF[i + j*NX + k*NX*NY + z*N] - FB[i + j*NX + k*NX*NY + z*N])
							- (DT / DZ)*(FU[i + j*NX + k*NX*NY + z*N] - FD[i + j*NX + k*NX*NY + z*N]);

#ifdef DEBUG_VALUE
						/* ...test... */
						if (U[i + j*NX + k*NX*NY + 4 * N] - 0.01 > 0.0) {
							printf("(DT / DX) = %f\n", (DT / DX));
							printf("U[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, U[i + j*NX + k*NX*NY + z*N]);
							printf("FR[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, FR[i + j*NX + k*NX*NY + z*N]);
							printf("FL[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, FL[i + j*NX + k*NX*NY + z*N]);
							printf("FF[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, FF[i + j*NX + k*NX*NY + z*N]);
							printf("FB[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, FB[i + j*NX + k*NX*NY + z*N]);
							printf("FU[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, FU[i + j*NX + k*NX*NY + z*N]);
							printf("FD[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, FD[i + j*NX + k*NX*NY + z*N]);
							system("pause");
						}
#endif // DEBUG_VALUE
					}
				}
				else {
					// body
					for (z = 0; z < 5; z++) {
						U_new[i + j*NX + k*NX*NY + z*N] = 1.0;
					}
				}

#ifdef DEBUG_VALUE
				/* ...test... */
				if (U_new[i + j*NX + k*NX*NY + 0 * N] - 0.01 > 0.0) {
					printf("U_new[%d + %d*NX + %d*NX*NY + %d*N] = %f\n", i, j, k, z, U_new[i + j*NX + k*NX*NY + z*N]);
					system("pause");
				}
#endif // DEBUG_VALUE


			}
		}
	}
#pragma omp barrier

	//Renew left and right boundary condition
#pragma omp for
	for (i = 1; i < (NX - 1); i++) {
		for (k = 1; k < (NZ - 1); k++) {
			for (z = 0; z < 5; z++) {
				// left = left+100: 10001 = 10101
				U_new[i + k*NX*NY + z*N] = U_new[i + NX + k*NX*NY + z*N];
				// right = right-100: 19901 = 19801
				U_new[i + (NY - 1)*NX + k*NX*NY + z*N] = U_new[i + (NY - 2)*NX + k*NX*NY + z*N];
			}
		}
	}
#pragma omp barrier

	//Renew back and front boundary condition
#pragma omp for
	for (j = 0; j < NY; j++) {
		for (k = 1; k < (NZ - 1); k++) {
			for (z = 0; z < 5; z++) {
				// front = front+1: 10000 = 10001
				U_new[j*NX + k*NX*NY + z*N] = U_new[1 + j*NX + k*NX*NY + z*N];
				// back = back-1: 10099 = 10098
				U_new[(NX - 1) + j*NX + k*NX*NY + z*N] = U_new[(NX - 2) + j*NX + k*NX*NY + z*N];
			}
		}
	}
#pragma omp barrier

	//Renew top and down boundary condition
#pragma omp for
	for (i = 0; i < NX; i++) {
		for (j = 0; j < NY; j++) {
			for (z = 0; z < 5; z++) {
				// top = top-10000: 990000 = 980000
				U_new[i + j*NX + (NZ - 1)*NX*NY + z*N] = U_new[i + j*NX + (NZ - 2)*NX*NY + z*N];
				// bottom = bottom+10000: 0 = 10000
				U_new[i + j*NX + z*N] = U_new[i + j*NX + NX*NY + z*N];
			}
		}
	}
#pragma omp barrier

	// Update density, velocity, pressure, and U
#pragma omp for //collapse(2)
	for (k = 0; k < NZ; k++) {
		for (j = 0; j < NY; j++) {
			for (i = 0; i < NX; i++) {
				dens[i + j*NX + k*NX*NY] = U_new[i + j*NX + k*NX*NY + 0 * N];

#ifdef DEBUG_VALUE
				/* ...test... */
				//if (U_new[i + j*NX + k*NX*NY + 0 * N] - 0.01 > 0.0) {
				//	printf("%d, %d, %d, %f\n", k, j, i, U_new[i + j*NX + k*NX*NY + 0 * N]);
				//	system("pause");
				//}  
#endif // DEBUG_VALUE

				xv[i + j*NX + k*NX*NY] = U_new[i + j*NX + k*NX*NY + 1 * N] / U_new[i + j*NX + k*NX*NY + 0 * N];
				yv[i + j*NX + k*NX*NY] = U_new[i + j*NX + k*NX*NY + 2 * N] / U_new[i + j*NX + k*NX*NY + 0 * N];
				zv[i + j*NX + k*NX*NY] = U_new[i + j*NX + k*NX*NY + 3 * N] / U_new[i + j*NX + k*NX*NY + 0 * N];
				press[i + j*NX + k*NX*NY] = (GAMA - 1) * (U_new[i + j*NX + k*NX*NY + 4 * N] - 0.5 * dens[i + j*NX + k*NX*NY] * (xv[i + j*NX + k*NX*NY] * xv[i + j*NX + k*NX*NY] + yv[i + j*NX + k*NX*NY] * yv[i + j*NX + k*NX*NY] + zv[i + j*NX + k*NX*NY] * zv[i + j*NX + k*NX*NY]));

				if (h_body[i + j*NX + k*NX*NY] != 0.0) {
					// air
					U[i + j*NX + k*NX*NY + 0 * N] = U_new[i + j*NX + k*NX*NY + 0 * N];
					U[i + j*NX + k*NX*NY + 1 * N] = U_new[i + j*NX + k*NX*NY + 1 * N];
					U[i + j*NX + k*NX*NY + 2 * N] = U_new[i + j*NX + k*NX*NY + 2 * N];
					U[i + j*NX + k*NX*NY + 3 * N] = U_new[i + j*NX + k*NX*NY + 3 * N];
					U[i + j*NX + k*NX*NY + 4 * N] = U_new[i + j*NX + k*NX*NY + 4 * N];
				}
				else {
					// body
					for (z = 0; z < 5; z++) {
						U[i + j*NX + k*NX*NY + z * N] = 1.0;
					}
				}
			}
		}
	}
#pragma omp barrier
}

#ifdef GPU
void Call_GPUHeatContactFunction() {
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	size_t size = N*sizeof(float);
	cudaError_t Error;
	GPUHeatContactFunction << <blocksPerGrid, threadsPerBlock >> >(d_a, d_b, d_body);
	Error = cudaMemcpy(d_a, d_b, size, cudaMemcpyDeviceToDevice);
	printf("CUDA error (memcpy d_b -> d_a) = %s\n", cudaGetErrorString(Error));
}

__global__ void GPUHeatContactFunction(float *a, float *b, int *body) {

}

void Call_GPUTimeStepFunction() {
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	size_t size = N*sizeof(float);
	cudaError_t Error;
	GPUTimeStepFunction << <blocksPerGrid, threadsPerBlock >> >(d_a, d_b, d_body);
	Error = cudaMemcpy(d_a, d_b, size, cudaMemcpyDeviceToDevice);
	printf("CUDA error (memcpy d_b -> d_a) = %s\n", cudaGetErrorString(Error));
}

__global__ void GPUTimeStepFunction(float *a, float *b, int *body){

}

void Send_To_Device() {
	size_t size = N*sizeof(float);
	cudaError_t Error;
	Error = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy h_a -> d_a) = %s\n", cudaGetErrorString(Error));
	size = N*sizeof(float);
	Error = cudaMemcpy(d_body, h_body, size, cudaMemcpyHostToDevice);
	printf("CUDA error (memcpy h_body -> d_body) = %s\n", cudaGetErrorString(Error));
}

void Get_From_Device() {
	size_t size = N*sizeof(float);
	cudaError_t Error;
	Error = cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
	printf("CUDA error (memcpy d_a -> h_a) = %s\n", cudaGetErrorString(Error));
}
#endif

void Free_Memory() {
	if (dens) free(dens);
	if (temperature) free(temperature);
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
	if (G) free(G);
#ifdef GPU
	if (d_dens) cudaFree(d_dens);
	if (d_temperature) cudafree(d_temperature);
	if (d_xv) cudafree(d_xv);
	if (d_yv) cudafree(d_yv);
	if (d_zv) cudafree(d_zv);
	if (d_press) cudafree(d_press);
	if (d_body) cudaFree(d_body);
#endif
}

void Save_Data_To_File(char *output_file_name) {
	FILE *pOutPutFile;
	pOutPutFile = fopen(output_file_name, "w");
	if (!pOutPutFile) { printf("Open failure"); }

	fprintf(pOutPutFile, "TITLE=\"Flow Field of X-37\"\n");
	/* ...test body...*/
	//fprintf(pOutPutFile, "VARIABLES=\"X\", \"Y\", \"Z\", \"Body\"\n");
	/* ...test body...*/
	fprintf(pOutPutFile, "VARIABLES=\"X\", \"Y\", \"Z\", \"Vx\", \"Vy\", \"Vz\", \"Pressure\", \"Temperature\"\n");
	fprintf(pOutPutFile, "ZONE I = 100, J = 100, K = 100, F = POINT\n");

	int i, j, k;
	for (k = 0; k < NZ; k++) {
		for (j = 0; j < NY; j++) {
			for (i = 0; i < NX; i++) {
				/* ...test body...*/
				//fprintf(pOutPutFile, "%d %d %d %f\n", i, j, k, h_body[i + j*NX + k*NX*NY]);
				/* ...test body...*/
				fprintf(pOutPutFile, "%d %d %d %f %f %f %f %f\n", i, j, k, xv[i + j*NX + k*NX*NY], yv[i + j*NX + k*NX*NY], zv[i + j*NX + k*NX*NY], press[i + j*NX + k*NX*NY], temperature[i + j*NX + k*NX*NY]);
			}
		}
	}
}

