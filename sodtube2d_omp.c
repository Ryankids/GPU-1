#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/time.h>

#define num_threads 4 //number of threads wanted to use
#define NX 100   // Number of cells in X direction
#define NY 100   // Number of cells in Y direction
#define N  NX*NY // Number of total cells
#define L 1.0   // Dimensionless length of tube
#define W 1.0   // Dimensionless width of tube
#define dx (L/NX)   // Lenth of cell
#define dy (W/NY)   // Width of cell
#define dt 0.01*0.02  // Size of time step
#define no_steps 4000  // No. of time steps

#define R (1.0)         // Dimensionless specific gas constant
#define GAMA (7.0/5.0)     // Ratio of specific heats
#define CV (R/(GAMA-1.0)) // Cv
#define CP (CV + R)       // Cp

float *dens;     //density
float *xv;       //velocity in x
float *yv;		 //velocity in y
float *press;    //pressure

float U[N][4];
float U_new[N][4];
float E[N][4];
float F[N][4];
float FR[N][4];
float FL[N][4];
float FU[N][4];
float FD[N][4];


float speed;

void Allocate_Memory();
void Init();
void Free();
void CalculateFlux();
void CalculateResult();
void Save_Results();

int main()
{
	clock_t start, end;

	struct timeval t1, t2;
	double timecost;

	// start timer
	gettimeofday(&t1, NULL);

	int i;
	Allocate_Memory();
	Init();
	omp_set_num_threads(num_threads);
	#pragma omp parallel private(speed,i)
	{
		for (i = 0; i < no_steps; i++)
		{
			CalculateFlux();
			CalculateResult();
		}
	}
	Save_Results();
	Free();

	// stop timer
	gettimeofday(&t2, NULL);

	// compute and print time cost in ms
	timecost = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
	timecost += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
	printf("Code time cost: %f\n", timecost);
	return 0;
}

void Allocate_Memory()
{
	size_t size = N*sizeof(float);
	dens = (float*)malloc(size);
	xv = (float*)malloc(size);
	yv = (float*)malloc(size);
	press = (float*)malloc(size);
}

void Init()
{
	int i, j;

	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			float d =0;
			float cx = dx*(i+0.5);
			float cy = dy*(j+0.5);
			d = (cx - 0.5)*(cx - 0.5) + (cy-0.5)* (cy-0.5);
			if(d <= 0.5*L)
			{
				//Initialize the right side gas condition
				dens[i + j*NX] = 10.0;
				xv[i + j*NX] = 0.0;
				yv[i + j*NX] = 0.0;
				press[i + j*NX] = 1.0;
			}
			else
			{
				dens[i + j*NX] = 1.0;
				xv[i + j*NX] = 0.0;
				yv[i + j*NX] = 0.0;
				press[i + j*NX] = 1.0;
				
			}
			/*if (i > 0.1*NX)
			{
				//Initialize the right side gas condition
				dens[i + j*NX] = 3.81;
				xv[i + j*NX] = 0.0;
				yv[i + j*NX] = 0.0;
				press[i + j*NX] = 10.0;
			}
			else{
				d = (cx - 0.3)*(cx - 0.3) + cy* cy;
			 	if(d <= 0.2*L)
				{
				//Initialize the left side gas condition
					dens[i + j*NX] = 0.1;
					xv[i + j*NX] = 0.0;
					yv[i + j*NX] = 0.0;
					press[i + j*NX] = 10.0;
					break;
				}
				//Initialize the left side gas condition
				dens[i + j*NX] = 1.0;
				xv[i + j*NX] = 0.0;
				yv[i + j*NX] = 0.0;
				press[i + j*NX] = 1.0;
				
			}*/
			U[i + j*NX][0] = dens[i + j*NX];
			U[i + j*NX][1] = dens[i + j*NX] * (xv[i + j*NX]);
			U[i + j*NX][2] = dens[i + j*NX] * (yv[i + j*NX]);
			U[i + j*NX][3] = dens[i + j*NX] * (CV*(press[i + j*NX]/dens[i + j*NX]/R)
				+ 0.5*((xv[i + j*NX] * xv[i + j*NX]) + (yv[i + j*NX] * yv[i + j*NX])));
		}
	}
}

void CalculateFlux()
{
	int i, j;

	
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			E[i+j*NX][0] = dens[i + j*NX] * xv[i + j*NX];
			E[i+j*NX][1] = dens[i + j*NX] * xv[i + j*NX]*xv[i + j*NX] + press[i + j*NX];
			E[i+j*NX][2] = dens[i + j*NX] * xv[i + j*NX] * yv[i + j*NX];
			E[i+j*NX][3] = xv[i + j*NX] * ( U[i + j*NX][3] +  press[i + j*NX] );
			F[i+j*NX][0] = dens[i + j*NX] * yv[i + j*NX];
			F[i+j*NX][1] = dens[i + j*NX] * xv[i + j*NX] * yv[i + j*NX];
			F[i+j*NX][2] = dens[i + j*NX] * yv[i + j*NX] * yv[i + j*NX] + press[i + j*NX];
			F[i+j*NX][3] = yv[i + j*NX] * ( U[i + j*NX][3] +  press[i + j*NX] );
		}
	}
	#pragma omp for collapse(2)// private(speed)
	for (i = 1; i < (NX - 1); i++)
	{
		for (j = 1; j < (NY - 1); j++)
		{
			speed = sqrt(GAMA*press[i + j*NX]/dens[i + j*NX]);
			
			FL[i + j*NX][0] = 0.5*(E[i + j*NX][0] + E[i + j*NX - 1][0])- speed*(U[i + j*NX][0] - U[i + j*NX - 1][0]);
			FR[i + j*NX][0] = 0.5*(E[i + j*NX][0] + E[i + j*NX + 1][0])- speed*(U[i + j*NX + 1][0] - U[i + j*NX][0]);
			FL[i + j*NX][1] = 0.5*(E[i + j*NX][1] + E[i + j*NX - 1][1])- speed*(U[i + j*NX][1] - U[i + j*NX - 1][1]);
			FR[i + j*NX][1] = 0.5*(E[i + j*NX][1] + E[i + j*NX + 1][1])- speed*(U[i + j*NX + 1][1] - U[i + j*NX][1]);
			FL[i + j*NX][2] = 0.5*(E[i + j*NX][2] + E[i + j*NX - 1][2])- speed*(U[i + j*NX][2] - U[i + j*NX - 1][2]);
			FR[i + j*NX][2] = 0.5*(E[i + j*NX][2] + E[i + j*NX + 1][2])- speed*(U[i + j*NX + 1][2] - U[i + j*NX][2]);
			FL[i + j*NX][3] = 0.5*(E[i + j*NX][3] + E[i + j*NX - 1][3])- speed*(U[i + j*NX][3] - U[i + j*NX - 1][3]);
			FR[i + j*NX][3] = 0.5*(E[i + j*NX][3] + E[i + j*NX + 1][3])- speed*(U[i + j*NX + 1][3] - U[i + j*NX][3]);


			FD[i + j*NX][0] = 0.5*(F[i + (j - 1)*NX][0] + F[i + j*NX][0])- speed*(U[i + j*NX][0] - U[i + (j - 1)*NX][0]);
			FU[i + j*NX][0] = 0.5*(F[i + j*NX][0] + F[i + (j + 1)*NX][0])- speed*(U[i + (j + 1)*NX][0] - U[i + j*NX][0]);
			FD[i + j*NX][1] = 0.5*(F[i + (j - 1)*NX][1] + F[i + j*NX][1])- speed*(U[i + j*NX][1] - U[i + (j - 1)*NX][1]);
			FU[i + j*NX][1] = 0.5*(F[i + j*NX][1] + F[i + (j + 1)*NX][1])- speed*(U[i + (j + 1)*NX][1] - U[i + j*NX][1]);
			FD[i + j*NX][2] = 0.5*(F[i + (j - 1)*NX][2] + F[i + j*NX][2])- speed*(U[i + j*NX][2] - U[i + (j - 1)*NX][2]);
			FU[i + j*NX][2] = 0.5*(F[i + j*NX][2] + F[i + (j + 1)*NX][2])- speed*(U[i + (j + 1)*NX][2] - U[i + j*NX][2]);
			FD[i + j*NX][3] = 0.5*(F[i + (j - 1)*NX][3] + F[i + j*NX][3])- speed*(U[i + j*NX][3] - U[i + (j - 1)*NX][3]);
			FU[i + j*NX][3] = 0.5*(F[i + j*NX][3] + F[i + (j + 1)*NX][3])- speed*(U[i + (j + 1)*NX][3] - U[i + j*NX][3]);
		}
	}
#pragma omp barrier
}
void CalculateResult()
{
	int i, j;
#pragma omp for collapse(2)
	for (i = 1; i < (NX - 1); i++)
	{
		for (j = 1; j < (NY - 1); j++)
		{
			U_new[i + j*NX][0] = U[i + j*NX][0] - (dt / dx)*(FR[i + j*NX][0] - FL[i + j*NX][0]) -
				(dt / dy)*(FU[i + j*NX][0] - FD[i + j*NX][0]);
			U_new[i + j*NX][1] = U[i + j*NX][1] - (dt / dx)*(FR[i + j*NX][1] - FL[i + j*NX][1]) - 
				(dt / dy)*(FU[i + j*NX][1] - FD[i + j*NX][1]);
			U_new[i + j*NX][2] = U[i + j*NX][2] - (dt / dx)*(FR[i + j*NX][2] - FL[i + j*NX][2]) - 
				(dt / dy)*(FU[i + j*NX][2] - FD[i + j*NX][2]);
			U_new[i + j*NX][3] = U[i + j*NX][3] - (dt / dx)*(FR[i + j*NX][3] - FL[i + j*NX][3]) - 
				(dt / dy)*(FU[i + j*NX][3] - FD[i + j*NX][3]);
		}
	}
#pragma omp barrier

	//Renew up and down boundary condition
#pragma omp for
	for (i = 1; i < (NX - 1); i++)
	{
		U_new[i][0] = U_new[i + NX][0];
		U_new[i][1] = U_new[i + NX][1];
		U_new[i][2] = U_new[i + NX][2];
		U_new[i][3] = U_new[i + NX][3];
		U_new[i + (NY - 1)*NX][0] = U_new[i + (NY - 2)*NX][0];
		U_new[i + (NY - 1)*NX][1] = U_new[i + (NY - 2)*NX][1];
		U_new[i + (NY - 1)*NX][2] = U_new[i + (NY - 2)*NX][2];
		U_new[i + (NY - 1)*NX][3] = U_new[i + (NY - 2)*NX][3];
	}
#pragma omp barrier

	//Renew left and right boundary condition
#pragma omp for
	for (i = 0; i < NY; i++)
	{
		U_new[i*NX][0] = U_new[i*NX + 1][0];
		U_new[i*NX][1] = U_new[i*NX + 1][1];
		U_new[i*NX][2] = U_new[i*NX + 1][2];
		U_new[i*NX][3] = U_new[i*NX + 1][3];
		U_new[(NX - 1) + i*NX][0] = U_new[(NX - 2) + i*NX][0];
		U_new[(NX - 1) + i*NX][1] = U_new[(NX - 2) + i*NX][1];
		U_new[(NX - 1) + i*NX][2] = U_new[(NX - 2) + i*NX][2];
		U_new[(NX - 1) + i*NX][3] = U_new[(NX - 2) + i*NX][3];
	}
#pragma omp barrier

#pragma omp for collapse(2)
	for (i = 0; i < NX; i++)
	{
		for (j = 0; j < NY; j++)
		{
			dens[i + j*NX] = U_new[i + j*NX][0];
			xv[i + j*NX] = U_new[i + j*NX][1] / U_new[i + j*NX][0];
			yv[i + j*NX] = U_new[i + j*NX][2] / U_new[i + j*NX][0];
			press[i + j*NX] = (GAMA - 1) * (U_new[i + j*NX][3] - 0.5 * dens[i + j*NX] *(xv[i + j*NX]*xv[i + j*NX] + yv[i + j*NX]*yv[i + j*NX]) );
			U[i + j*NX][0] = U_new[i + j*NX][0];
			U[i + j*NX][1] = U_new[i + j*NX][1];
			U[i + j*NX][2] = U_new[i + j*NX][2];
			U[i + j*NX][3] = U_new[i + j*NX][3];
		}
	}
#pragma omp barrier
}
void Free()
{
	free(dens);
	free(xv);
	free(yv);
	free(press);

}

void Save_Results() {
	FILE *pFile;
	int i, j;
	printf("Saving...");
	pFile = fopen("Results.txt", "w");
	for (j = 0; j < NY; j++) {
		for (i = 0; i < NX; i++) {
			fprintf(pFile, "%d %d %g %g %g %g\n", i + 1, j + 1, dens[i + j*NX], xv[i + j*NX], yv[i + j*NX], press[i + j*NX]);
		}
	}
	fclose(pFile);
	printf("Done.\n");
}
