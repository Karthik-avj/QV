#include<math.h>
#include<stdio.h>
#include<time.h>
#include<complex.h>
#include "cuda_runtime.h"

#define n 2
#define tot_samples 200000
#define fc 100000000.0
#define c 300000000.0
#define A0 1.0
#define PRF 10000.0
#define tp 0.00001
#define alpha 4000000000000.0
#define fs 1000000000.0
#define pi 3.141592654
#define h 1.0
#define theta1 30.0 
#define theta2 60.0
#define l 2.0
#define tot_time 0.0002
#define N 1024
#define M 512
#define pos 1.15

__global__ void compute_r(float *b);
__global__ void compute_c(float *b);

void func_init(float f[tot_samples]);
void compute_sum(float b[n][tot_samples], float s[tot_samples]);

int main(){
	
	float b_r[n][tot_samples];
	float b_c[n][tot_samples];
	float f[tot_samples], s_r[tot_samples], s_c[tot_samples];
	int i,j;

	float *dev_br, *dev_pos, *dev_bc;
	int size_s = tot_samples*sizeof(float);
	int size_b = n*tot_samples*sizeof(float);
	int size_pos = n*sizeof(float);

	clock_t start, end;
	double elapsed;

	start = clock();
	cudaMalloc((void **)&dev_br, size_b);
	compute_r<<<N,M>>>(dev_br);
	cudaMemcpy(&b_r, dev_br, size_b, cudaMemcpyDeviceToHost);

	cudaMalloc((void **)&dev_bc, size_b);
	compute_c<<<N,M>>>(dev_bc);
	cudaMemcpy(&b_c, dev_bc, size_b, cudaMemcpyDeviceToHost);

	end = clock();
	elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;   
    printf(" \n Time taken is %f \n",elapsed);

	compute_sum(b_r,s_r);
	compute_sum(b_c,s_c);

	FILE *fptr;

  	fptr = fopen("values.csv","w");
    for(i=0; i<tot_samples; i++){
    	fprintf(fptr,"%d,%lf,%lf\n",i,s_r[i],s_c[i]);
    }
    fclose(fptr);

    cudaFree(dev_br);
    cudaFree(dev_bc);
}

void compute_sum(float b[n][tot_samples], float s[tot_samples]){
	
	for(int j=0; j<tot_samples; j++){
		s[j] = 0.0;
	}
	for(int i=0; i<n; i++){
		for(int j=0; j<tot_samples; j++){
			s[j] += b[i][j];
		}
	}	
}

__global__ void compute_r(float *b){

	if (blockIdx.x*blockDim.x+threadIdx.x<tot_samples){
		if ((pos>h*tan(pi*theta1/180)) && (pos<h*tan(pi*theta2/180))){
			float t = remainder((double)(((blockIdx.x*blockDim.x+threadIdx.x)*tot_time/tot_samples)-(sqrt(h*h+pow(pos,2)))/(c)),(double)(1/PRF));
			if ((((t-tp/2)/tp)<=0.5) && (((t-tp/2)/tp)>=-0.5)){
				b[blockIdx.x*blockDim.x+threadIdx.x] = A0*cos(pi*(2*fc*(t-tp/2)+alpha*pow(t-tp/2,2)));
			}
			else{
				b[blockIdx.x*blockDim.x+threadIdx.x] = 0.0;
			}
		}
		else{
			b[blockIdx.x*blockDim.x+threadIdx.x] = 0.0;
		}
	}
}

__global__ void compute_c(float *b){

	if (blockIdx.x*blockDim.x+threadIdx.x<tot_samples){
		if ((pos>h*tan(pi*theta1/180)) && (pos<h*tan(pi*theta2/180))){
			float t = remainder((double)(((blockIdx.x*blockDim.x+threadIdx.x)*tot_time/tot_samples)-(sqrt(h*h+pow(pos,2)))/(c)),(double)(1/PRF));
			if ((((t-tp/2)/tp)<=0.5) && (((t-tp/2)/tp)>=-0.5)){
				b[blockIdx.x*blockDim.x+threadIdx.x] = A0*sin(pi*(2*fc*(t-tp/2)+alpha*pow(t-tp/2,2)));
			}
			else{
				b[blockIdx.x*blockDim.x+threadIdx.x] = 0.0;
			}
		}
		else{
			b[blockIdx.x*blockDim.x+threadIdx.x] = 0.0;
		}
	}
}
