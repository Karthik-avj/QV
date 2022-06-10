#include<math.h>
#include<stdio.h>
#include<time.h>
#include "cuda_runtime.h"

#define n 1
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
#define N 512
#define M 512
#define pos 1.15

__global__ void compute(float *b);

void func_init(float f[tot_samples]);
void compute_sum(float b[n][tot_samples], float s[tot_samples]);

int main(){
	
	float b[n][tot_samples];
	float f[tot_samples], s[tot_samples];
	int i,j;

	float *dev_b, *dev_pos;
	int size_s = tot_samples*sizeof(float);
	int size_b = n*tot_samples*sizeof(float);
	int size_pos = n*sizeof(float);

	clock_t start, end;
	double elapsed;

	start = clock();
	cudaMalloc((void **)&dev_b, size_b);
	compute<<<N,M>>>(dev_b);
	cudaMemcpy(&b, dev_b, size_b, cudaMemcpyDeviceToHost);

	end = clock();
	elapsed = ((double) (end - start)) / CLOCKS_PER_SEC;   
    printf(" \n Time taken is %f \n",elapsed);

	compute_sum(b,s);

	FILE *fptr;

  	fptr = fopen("values.csv","w");
    for(i=0; i<tot_samples; i++){
    	fprintf(fptr,"%d,%lf\n",i,s[i]);
    }
    fclose(fptr);

    cudaFree(dev_b);
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

__global__ void compute(float *b){

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
