#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"
// #include<time.h>
#include<float.h>

__constant__ int FIL[32*5*5];

__global__ void conv1(unsigned int *picd, int *resultd){
  int i,j,k,l;
  int sum, offset;
  i = threadIdx.y;
  j = threadIdx.x;
  l = blockIdx.x;
  offset = l*25;
  int xsize = 28;
  int filterdim = 5;
  k=0;
  sum =0;
  if(i<(xsize -filterdim +1)&& j<(xsize -filterdim +1)){
    sum = FIL[offset + k]*picd[ xsize * (i) + j ] + FIL[offset+ k+1]*picd[ xsize*(i) + (j+1) ]
      + FIL[offset+ k+2]*picd[ xsize * (i)+(j+2)] + FIL[offset+k+3]*picd[xsize * (i)+(j+3)]
      + FIL[offset+k+4]*picd[ xsize * (i)+(j+4)]+ FIL[offset+ k+5]*picd[ xsize*(i+1)+(j) ]
      + FIL[offset+k+6]*picd[ xsize * (i+1) + (j+1) ] + FIL[offset+ k+7]*picd[ xsize*(i+1) + (j+2) ] +
      FIL[offset+k+8]*picd[ xsize*(i+1) + (j+3) ] + FIL[offset+k+9]*picd[ xsize*(i+1) + (j+4) ] +
      FIL[offset+k+10]*picd[ xsize*(i+2) + (j) ]	+ FIL[offset+k+11]*picd[ xsize * (i+2) + (j+1) ] +
      FIL[offset+k+12]*picd[ xsize*(i+2) + (j+2)] + FIL[offset+k+13]*picd[ xsize*(i+2) + (j+3)]
      +FIL[offset+k+14]*picd[ xsize*(i+2) + (j+4)] + FIL[offset +k+15]*picd[ xsize*(i+3) + (j)]
      + FIL[offset+k+16]*picd[ xsize*(i+3) + (j+1)] + FIL[offset+k+17]*picd[ xsize*(i+3) + (j+2)]
      + FIL[offset+k+18]*picd[ xsize*(i+3) + (j+3)] + FIL[offset+k+19]*picd[ xsize*(i+3) + (j+4)]
      + FIL[offset+k+20]*picd[ xsize*(i+4) + (j)] +FIL[offset+k+21]*picd[ xsize*(i+3) + (j+1)]
      + FIL[offset+k+22]*picd[ xsize*(i+4) + (j+2)] + FIL[offset+k+23]*picd[ xsize*(i+4) + (j+3)]
      + FIL[offset+ k+24]*picd[ xsize*(i+4) + (j+4)];

      resultd[l*(xsize -filterdim +1)*(xsize -filterdim +1) + i*(xsize - filterdim +1)+j] = sum;
      //printf("resultgpu[%d][%d]=%d\n",l,i*(xsize - filterdim +1)+j,resulth[l*(xsize -filterdim +1)*(xsize -filterdim +1) + i*(xsize - filterdim +1)+j]);
  }
}

__global__ void maxpooling(int *maxip1d, int *maxop1d){

  int i,j,l;
  i = threadIdx.y;
  j = threadIdx.x;
  l = blockIdx.x;
  int xsize = 28;
  int filterdim = 5;
  if(i<((xsize-filterdim+1)/2)&&(j<((xsize-filterdim+1)/2))){
    int a,b,c,d,index, max1, max2;
    index = l*((xsize -filterdim +1)*(xsize -filterdim +1))+ threadIdx.x*2 + threadIdx.y*2*(xsize -filterdim +1);
    a = maxip1d[index];
    b = maxip1d[index +1];
    c = maxip1d[index+(xsize-filterdim+1)];
    d = maxip1d[index + (xsize-filterdim+2)];
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
      maxop1d[l*(xsize -filterdim +1)*(xsize -filterdim +1)/4 + i*(xsize - filterdim +1)/2+j]=max1;
    }
    else{
      maxop1d[l*(xsize -filterdim +1)*(xsize -filterdim +1)/4 + i*(xsize - filterdim +1)/2+j] = max2;
    }
  }
}

__global__ void conv2(int *cip2d, int *filter2d, int *cop2d){
  int i,j,l,sum;
  i = threadIdx.y;
  j = threadIdx.x;
  l = blockIdx.x;
  int lstar;
  lstar = l*800;
  sum = 0;
  int k =0;
  int di = 12;
  int disquare = di*di;
  int m;
  if(i<8 && j<8){
    for(m = 0; m<32; m++){
      sum = sum + filter2d[lstar + k]*cip2d[(m*disquare)+ (di*i) + j] + filter2d[lstar + k+1]*cip2d[(m*disquare)+ di*(i) + (j+1)]
        + filter2d[lstar+ k+2]*cip2d[(m*disquare)+ di*(i)+(j+2)] + filter2d[lstar +k+3]*cip2d[(m*disquare)+ di*(i)+(j+3)]
        + filter2d[lstar+k+4]*cip2d[(m*disquare)+ di*(i)+(j+4)]+ filter2d[lstar+ k+5]*cip2d[(m*disquare)+ di*(i+1)+(j)]
        + filter2d[lstar +k+6]*cip2d[(m*disquare)+ di* (i+1) + (j+1) ] + filter2d[lstar+ k+7]*cip2d[(m*disquare)+ di*(i+1)+(j+2)]
        + filter2d[lstar+k+8]*cip2d[(m*disquare)+ di*(i+1) + (j+3) ] + filter2d[lstar +k+9]*cip2d[(m*disquare)+ di*(i+1) +(j+4)]
        + filter2d[lstar+k+10]*cip2d[(m*disquare)+ di*(i+2) +(j)]	+ filter2d[lstar+k+11]*cip2d[(m*disquare)+ di* (i+2) + (j+1)]
        + filter2d[lstar+k+12]*cip2d[(m*disquare)+ di*(i+2) + (j+2)] +filter2d[lstar+k+13]*cip2d[(m*disquare)+ di*(i+2)+(j+3)]
        + filter2d[lstar+k+14]*cip2d[(m*disquare)+ di*(i+2)+(j+4)]+filter2d[lstar+k+15]*cip2d[(m*disquare)+ di*(i+3)+(j)]
        + filter2d[lstar+k+16]*cip2d[(m*disquare)+ di*(i+3)+(j+1)]+filter2d[lstar+k+17]*cip2d[(m*disquare)+ di*(i+3)+(j+2)]
        + filter2d[lstar+k+18]*cip2d[(m*disquare)+ di*(i+3)+(j+3)] + filter2d[lstar+k+19]*cip2d[(m*disquare)+di*(i+3)+(j+4)]
        + filter2d[lstar+k+20]*cip2d[(m*disquare)+ di*(i+4)+(j)] +filter2d[lstar+k+21]*cip2d[(m*disquare)+ di*(i+3)+(j+1)]
        + filter2d[lstar +k+22]*cip2d[(m*disquare)+ di*(i+4)+(j+2)] + filter2d[lstar+k+23]*cip2d[(m*disquare)+ di*(i+4)+(j+3)]
        + filter2d[lstar+ k+24]*cip2d[(m*disquare)+ di*(i+4) + (j+4)];

      k+=25;
    }
    cop2d[l*64+i*8+j] = sum;
    // printf("resultdevice[%d][%d]:%d\n",l,i*8+j,cop2d[l*64+i*8+j]);
  }
}

__global__ void maxpool(int *maxip2d, int *maxop2d){

  int i,j,l;
  i = threadIdx.y;
  j = threadIdx.x;
  l = blockIdx.x;
  int xsize = 12;
  int filterdim = 5;
  if(i<((xsize-filterdim+1)/2)&&(j<((xsize-filterdim+1)/2))){
    int a,b,c,d,index, max1, max2;
    index = l*((xsize -filterdim +1)*(xsize -filterdim +1))+ threadIdx.x*2 + threadIdx.y*2*(xsize -filterdim +1);
    a = maxip2d[index];
    b = maxip2d[index +1];
    c = maxip2d[index+(xsize-filterdim+1)];
    d = maxip2d[index + (xsize-filterdim+2)];
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
      maxop2d[l*(xsize -filterdim +1)*(xsize -filterdim +1)/4 + i*(xsize - filterdim +1)/2+j]=max1;
    }
    else{
      maxop2d[l*(xsize -filterdim +1)*(xsize -filterdim +1)/4 + i*(xsize - filterdim +1)/2+j] = max2;
    }
  }
}

int main(int argc, char **argv){
  int xsize;
  int filterdim;
  int numfilters;
  int numfilters1;
  xsize = 28;
  filterdim = 5;
  numfilters = 32;
  numfilters1 = 64;

  /*Numbytes required for initial image*/
  int numbytes = xsize*xsize*sizeof(int);
  /*Numbytes require for the output of first convolution layer*/
  int numbytes2 = (xsize-filterdim+1)*(xsize-filterdim+1)*sizeof(int); //24x24
  /**Numbytes required for output of first maxpool layer**/
  int numbytes3 = ((xsize-filterdim+1)*(xsize-filterdim+1)/4)*sizeof(int); //12x12
  /*Numbytes required for the output of second convolution layer*/
  int numbytes4 = ((xsize-filterdim+1)/2 - filterdim + 1)*((xsize-filterdim+1)/2 - filterdim + 1)*sizeof(int);//8x8
  /*Numbytes required for the output of second maxpool layer*/
  int numbytes5 = (numbytes4/4)*sizeof(int);//4x4

  /*Image on host side*/
  /*Ip and op to first conv layer*/
  unsigned int *pic = (unsigned int *)malloc(numbytes);
  int *result;
  int filter[numfilters*filterdim*filterdim];
  /*Ip and op to first maxpool layer*/
  int *maxip1;
  int *maxop1;
  /*Ip and op of second conv layer*/
  int *cip2;
  int *cop2;
  int *filter2;
  /*ip and op to second maxpool layer*/
  int *maxip2;
  int *maxop2;

  /*Device side variables*/
  unsigned int *picd;
  int *resultd;
  /*Ip and op to first maxpool layer*/
  int *maxip1d;
  int *maxop1d;
  /*Ip and op of second conv layer*/
  int *cip2d;
  int *cop2d;
  int *filter2d;
  /*ip and op to second maxpool layer*/
  int *maxip2d;
  int *maxop2d;

  result = (int *)malloc(numfilters*numbytes2);
  maxip1 = (int *)malloc(numfilters*numbytes2);
  maxop1 = (int *)malloc(numfilters*numbytes3);
  cip2 = (int *)malloc(numfilters*numbytes3);
  cop2 = (int *)malloc(numfilters1*numbytes4);
  filter2 = (int *)malloc(numfilters1*numfilters*filterdim*filterdim*sizeof(int));
  maxip2 = (int *)malloc(numfilters1*numbytes4);
  maxop2 = (int *)malloc(numfilters1*numbytes5);

  cudaMalloc(&picd, numbytes);
  cudaMalloc(&resultd, numfilters*numbytes2);
  cudaMalloc(&maxip1d, numfilters*numbytes2);
  cudaMalloc(&maxop1d, numfilters*numbytes3);
  cudaMalloc(&cip2d, numfilters*numbytes3);
  cudaMalloc(&cop2d, numfilters1*numbytes4);
  cudaMalloc(&filter2d, numfilters1*numfilters*filterdim*filterdim*sizeof(int));
  cudaMalloc(&maxip2d, numfilters1*numbytes4);
  cudaMalloc(&maxop2d, numfilters1*numbytes5);

  /*Initializing the image on host side*/
  /*Should modify to later on read in image*/
  int i,j,k,l,count,dimx;
  for (i=0; i<xsize; i++) {
   for (j=0; j<xsize; j++) {
     pic[i*xsize + j] = 1;
     //printf("pic[%d][%d] : %d\t",i,j,pic[i*xsize + j]);
   }
   //  printf("\n");
  }

  /*Initializing the filter for first conv layer to a value*/
  /*TO DO : Read in filter from a file */
  for(int k=0;k<numfilters;k++){
   for (int i=0; i<filterdim; i++) {
     for (int j=0; j<filterdim; j++){
       filter[k*(filterdim*filterdim) + i*filterdim + j] = 1;
       // printf("filter[%d][%d]: %d\n",k, i*filterdim + j, filter[k*(filterdim*filterdim) + i*filterdim + j]);
     }
   }
 }

 /*Initializing the filter for second conv layer to a value*/
 /*TO DO : Read in filter from a file */
 for(int k=0;k<numfilters1;k++){
   for(int m= 0; m<numfilters;m++){
     for (int i=0; i<filterdim; i++) {
       for (int j=0; j<filterdim; j++){
         filter2[k*(numfilters*filterdim*filterdim)+ m*filterdim*filterdim + i*filterdim + j] = 1;
         // printf("filter2[%d][%d]: %d\t",k, m*filterdim*filterdim+i*filterdim + j, filter2[k*(numfilters*filterdim*filterdim)+ m*filterdim*filterdim + i*filterdim + j]);
       }
     }
   }
   // printf("\n");
 }

 dim3 dimGrid (32);
 dim3 dimBlock (32,32);

 cudaMemcpy(picd,pic,numbytes, cudaMemcpyHostToDevice);
 cudaMemcpyToSymbol(FIL, filter, numfilters*filterdim*filterdim*sizeof(int));

 conv1<<<dimGrid, dimBlock>>>(picd,resultd);

 cudaMemcpy(result,resultd,numfilters*numbytes2,cudaMemcpyDeviceToHost);

 dim3 dimBlock1 (16,16);
 cudaMemcpy(maxip1d, result,numfilters*numbytes2, cudaMemcpyHostToDevice);

 maxpooling<<<dimGrid, dimBlock1>>>(maxip1d, maxop1d);

 cudaMemcpy(maxop1, maxop1d, numfilters*numbytes3, cudaMemcpyDeviceToHost);

 cudaMemcpy(cip2d, maxop1,numfilters*numbytes3,cudaMemcpyHostToDevice);
 cudaMemcpy(filter2d, filter2,numfilters1*numfilters*filterdim*filterdim*sizeof(int), cudaMemcpyHostToDevice);

 dim3 dimGrid2(64);
 dim3 dimBlock2(8,8);

 conv2<<<dimGrid2, dimBlock2>>>(cip2d, filter2d, cop2d);

 cudaMemcpy(cop2, cop2d,numfilters1*numbytes4,cudaMemcpyDeviceToHost);

 cudaMemcpy(maxip2d, cop2,numfilters1*numbytes4,cudaMemcpyHostToDevice);

 maxpool<<<dimGrid2, dimBlock2>>>(maxip2d, maxop2d);

 cudaMemcpy(maxop2, maxop2d, numfilters*numbytes5, cudaMemcpyDeviceToHost);

 for(k=0;k<numfilters1;k++){
   for(i=0;i<4;i++){
     for(j=0;j<4;j++){
       printf("maxpool[%d][%d]:%d\t",k,i*4+j, maxop2[k*16+i*4+j]);
     }
     printf("\n");
   }
   printf("\n\n");
 }

}
