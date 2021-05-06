/*****Implemented first layer of convolution using global memory*******/
/**Implemented First Maxpool Layer**/
/**Measuring time**/
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"
// #include<time.h>
#include<float.h>

__constant__ int FIL[32*5*5];

__global__ void conv1(unsigned int *pich, int *resulth, int xsize, int numfilters, int filterdim){
  int i,j,k,l;
  int sum;
  int height;
  i = threadIdx.y;
  j = threadIdx.x;
  l = blockIdx.x;
  k=0;
  sum =0;
  // height = blockIdx.x*(xsize -filterdim +1)*(xsize -filterdim +1);
  if(i<(xsize -filterdim +1)&& j<(xsize -filterdim +1)){
    sum = (FIL[l*(filterdim*filterdim) + k])*pich[ xsize * (i) + j ] + (FIL[l*(filterdim*filterdim) + k+1])*pich[ xsize*(i) + (j+1) ]
      + FIL[l*(filterdim*filterdim)+ k+2]*pich[ xsize * (i)+(j+2)] + FIL[l*(filterdim*filterdim) +k+3]*pich[xsize * (i)+(j+3)]
      + FIL[l*(filterdim*filterdim) +k+4]*pich[ xsize * (i)+(j+4)]+ FIL[l*(filterdim*filterdim) + k+5]*pich[ xsize*(i+1)+(j) ]
      + FIL[l*(filterdim*filterdim) +k+6]*pich[ xsize * (i+1) + (j+1) ] + FIL[l*(filterdim*filterdim) + k+7]*pich[ xsize*(i+1) + (j+2) ] +
      FIL[l*(filterdim*filterdim) +k+8]*pich[ xsize*(i+1) + (j+3) ] + FIL[l*(filterdim*filterdim) +k+9]*pich[ xsize*(i+1) + (j+4) ] +
      FIL[l*(filterdim*filterdim) +k+10]*pich[ xsize*(i+2) + (j) ]	+ FIL[l*(filterdim*filterdim) +k+11]*pich[ xsize * (i+2) + (j+1) ] +
      FIL[l*(filterdim*filterdim) +k+12]*pich[ xsize*(i+2) + (j+2)] + FIL[l*(filterdim*filterdim) +k+13]*pich[ xsize*(i+2) + (j+3)]
      +FIL[l*(filterdim*filterdim) +k+14]*pich[ xsize*(i+2) + (j+4)] + FIL[l*(filterdim*filterdim) +k+15]*pich[ xsize*(i+3) + (j)]
      + FIL[l*(filterdim*filterdim) +k+16]*pich[ xsize*(i+3) + (j+1)] + FIL[l*(filterdim*filterdim) +k+17]*pich[ xsize*(i+3) + (j+2)]
      + FIL[l*(filterdim*filterdim) +k+18]*pich[ xsize*(i+3) + (j+3)] + FIL[l*(filterdim*filterdim) +k+19]*pich[ xsize*(i+3) + (j+4)]
      + FIL[l*(filterdim*filterdim) +k+20]*pich[ xsize*(i+4) + (j)] +FIL[l*(filterdim*filterdim) +k+21]*pich[ xsize*(i+3) + (j+1)]
      + FIL[l*(filterdim*filterdim) +k+22]*pich[ xsize*(i+4) + (j+2)] + FIL[l*(filterdim*filterdim) +k+23]*pich[ xsize*(i+4) + (j+3)]
      + FIL[l*(filterdim*filterdim) + k+24]*pich[ xsize*(i+4) + (j+4)];

      resulth[l*(xsize -filterdim +1)*(xsize -filterdim +1) + i*(xsize - filterdim +1)+j] = sum;
      printf("resultgpu[%d][%d]=%d\n",l,i*(xsize - filterdim +1)+j,resulth[l*(xsize -filterdim +1)*(xsize -filterdim +1) + i*(xsize - filterdim +1)+j]);
  }
}

__global__ void maxpooling(int *resulth, int *maxpoolh, int xsize, int filterdim, int numfilters){

  int i,j,l;
  int temp;
  i = threadIdx.y;
  j = threadIdx.x;
  l = blockIdx.x;
  if(i<((xsize-filterdim+1)/2)&&(j<((xsize-filterdim+1)/2))){
    int a,b,c,d,index, max1, max2;
    index = l*((xsize -filterdim +1)*(xsize -filterdim +1))+ threadIdx.x*2 + threadIdx.y*2*(xsize -filterdim +1);
    a = resulth[index];
    b = resulth[index +1];
    c = resulth[index+(xsize-filterdim+1)];
    d = resulth[index + (xsize-filterdim+2)];
    if(a>b){
      max1 = a;
    }
    else{
      max1 = b;
    }
    if(c>d){
      max2 = c;
    }
    else{
      max2 = d;
    }
    if(max1>max2){
      maxpoolh[l*(xsize -filterdim +1)*(xsize -filterdim +1)/4 + i*(xsize - filterdim +1)/2+j]=max1;
    }
    else{
      maxpoolh[l*(xsize -filterdim +1)*(xsize -filterdim +1)/4 + i*(xsize - filterdim +1)/2+j] = max2;
    }
  }
}
int main( int argc, char **argv )
{

  int xsize;
  int filterdim;
  int numfilters;
  xsize = 28;
  filterdim = 5;
  numfilters =32;

 int numbytes = xsize*xsize*sizeof(int);
 int numbytes2 = (xsize-filterdim+1)*(xsize-filterdim+1)*sizeof(int);
 /**Numbytes required for output of first maxpool layer**/
 int numbytes3 = ((xsize-filterdim+1)*(xsize-filterdim+1)/4)*sizeof(int);

 unsigned int *pic = (unsigned int *)malloc(numbytes);
 unsigned int filter[numfilters*filterdim*filterdim];
 int *result;
 int *maxpool;

 result = (int *)malloc(numfilters*numbytes2);
 maxpool = (int *)malloc(numfilters*numbytes3);

 unsigned int *pich;
 int *resulth;
 int *maxpoolh;

 cudaMalloc(&pich, numbytes);
 cudaMalloc(&resulth, numfilters*numbytes2);
 cudaMalloc(&maxpoolh, numfilters*numbytes3);

 int i,j,k,l,count,dimx;
 for (i=0; i<xsize; i++) {
   for (j=0; j<xsize; j++) {
     pic[i*xsize + j] = 1;
     //printf("pic[%d][%d] : %d\t",i,j,pic[i*xsize + j]);
   }
   //  printf("\n");
 }

 for(int k=0;k<numfilters;k++){
   for (int i=0; i<filterdim; i++) {
     for (int j=0; j<filterdim; j++){
       filter[k*(filterdim*filterdim) + i*filterdim + j] = 1;
			 // printf("filter[%d][%d]: %d\n",k, i*filterdim + j, filter[k*(filterdim*filterdim) + i*filterdim + j]);

     }
   }
 }

 // int blocksize, gridsize;
 dim3 dimGrid (32);
 dim3 dimBlock (32,32);
 // gridsize = numfilters;
 // blocksize = (24,24);
 cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);

 cudaMemcpy(pich,pic,numbytes, cudaMemcpyHostToDevice);
 cudaMemcpyToSymbol(FIL, filter, numfilters*filterdim*filterdim*sizeof(int));

 conv1<<<dimGrid, dimBlock>>>(pich, resulth, xsize, numfilters, filterdim);

 cudaMemcpy(result,resulth,numfilters*numbytes2,cudaMemcpyDeviceToHost);

 dim3 dimBlock1 (16,16);
 cudaMemcpy(resulth, result,numfilters*numbytes2, cudaMemcpyHostToDevice);

 maxpooling<<<dimGrid, dimBlock1>>>(resulth, maxpoolh, xsize, filterdim, numfilters);

 cudaMemcpy(maxpool, maxpoolh, numfilters*numbytes3, cudaMemcpyDeviceToHost);


 cudaEventRecord(stop,0);
 cudaEventSynchronize(stop);
 float time = 0;
 cudaEventElapsedTime(&time, start, stop);
 cudaEventDestroy(start);
 cudaEventDestroy(stop);
 printf("Time taken on GPU: %f ms\n", time);
}
