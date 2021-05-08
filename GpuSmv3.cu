#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"
#include<time.h>
#include<float.h>

__constant__ int PIC[28*28];

__global__ void conv1(int *filterd, int *resultd){

  int xsize = 28;
  int filterdim = 5;

  __shared__ int fil[25];
  int i,j,l;
  int sum, offset;
  i = threadIdx.y;
  j = threadIdx.x;
  l = blockIdx.x;
  offset = l*25;
  sum =0;
  if(i<filterdim && j<filterdim){
    fil[i*filterdim+j] = filterd[offset + i*filterdim+j];
    // printf("offset: %d, \t fil[%d][%d]:%d\n",offset,i,j,fil[i*filterdim+j]);
  }
  __syncthreads();
  if(i<(xsize -filterdim +1)&& j<(xsize -filterdim +1)){
    sum = fil[0]*PIC[ xsize * (i) + j ] + fil[1]*PIC[ xsize*(i) + (j+1) ]
      + fil[2]*PIC[ xsize * (i)+(j+2)] + fil[3]*PIC[xsize * (i)+(j+3)]
      + fil[4]*PIC[ xsize * (i)+(j+4)]+ fil[5]*PIC[ xsize*(i+1)+(j) ]
      + fil[6]*PIC[ xsize * (i+1) + (j+1) ] + fil[7]*PIC[ xsize*(i+1) + (j+2) ] +
      fil[8]*PIC[ xsize*(i+1) + (j+3) ] + fil[9]*PIC[ xsize*(i+1) + (j+4) ] +
      fil[10]*PIC[ xsize*(i+2) + (j) ]	+ fil[11]*PIC[ xsize * (i+2) + (j+1) ] +
      fil[12]*PIC[ xsize*(i+2) + (j+2)] + fil[13]*PIC[ xsize*(i+2) + (j+3)]
      +fil[14]*PIC[ xsize*(i+2) + (j+4)] + fil[15]*PIC[ xsize*(i+3) + (j)]
      + fil[16]*PIC[ xsize*(i+3) + (j+1)] + fil[17]*PIC[ xsize*(i+3) + (j+2)]
      + fil[18]*PIC[ xsize*(i+3) + (j+3)] + fil[19]*PIC[ xsize*(i+3) + (j+4)]
      + fil[20]*PIC[ xsize*(i+4) + (j)] +fil[21]*PIC[ xsize*(i+3) + (j+1)]
      + fil[22]*PIC[ xsize*(i+4) + (j+2)] + fil[23]*PIC[ xsize*(i+4) + (j+3)]
      + fil[24]*PIC[ xsize*(i+4) + (j+4)];

      resultd[l*(xsize -filterdim +1)*(xsize -filterdim +1) + i*(xsize - filterdim +1)+j] = sum;
      // printf("offset2 : %d \t resultgpu[%d][%d]=%d\n",offset,l,i*(xsize - filterdim +1)+j,resultd[l*(xsize -filterdim +1)*(xsize -filterdim +1) + i*(xsize - filterdim +1)+j]);
  }
}

__global__ void maxpooling(int *maxip1d, int *maxop1d){

  int i,j,l,offset;
  i = threadIdx.y;
  j = threadIdx.x;
  l = blockIdx.x;
  int xsize = 24;
  int filterdim = 5;
  offset = l*xsize*xsize;

  __shared__ int max[576];

  if(i<12 && j<12){
    max[i*2*xsize + j*2] = maxip1d[offset + i*2*xsize + j*2];
    max[i*2*xsize+j*2+1] = maxip1d[offset + i*2*xsize +j*2+1];
    max[i*2*xsize+j*2+24] = maxip1d[offset + i*2*xsize +j*2+24];
    max[i*2*xsize+j*2+25] = maxip1d[offset + i*2*xsize +j*2+25];
    // printf("i: %d,\t j: %d,\t l: %d,\t max1: %d,\t max2: %d,\t max3: %d,\t max4: %d\n",i,j,l,max[i*xsize + j],max[i*xsize+1],max[i*xsize+24],max[i*xsize+25]);

  }

  __syncthreads();

  if(i<12 && j<12){
    int max1, max2;
    if(max[i*xsize + j]>=max[i*xsize + j+1]){
      max1 = max[i*xsize + j];
    }
    else{
      max1 = max[i*xsize + j+1];
    }
    if(max[i*xsize + j+24]>=max[i*xsize + j+25]){
      max2 = max[i*xsize + j+24];
    }
    else{
      max2 = max[i*xsize + j+25];
    }
    if(max1>=max2){
      maxop1d[l*144 + i*12+j]=max1;
      // printf("Max1 : %d\t l: %d \t i: %d\t j: %d\n",max1,l,i,j);
    }
    else{
      maxop1d[l*144 + i*12+j] = max2;
      // printf("Max2 : %d\n",max2);
    }
    // printf("Maxpool1d[%d][%d]:%d\n",l,i*12+j,maxop1d[l*144 + i*12+j]);
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
  int offset;
  offset = l*64;
  int xsize = 12;
  __shared__ int max2[64];

  if(i<8 && j<8){
    max2[i*8 + j] = maxop2d[offset + i*8 +j];
  }
  __syncthreads();
  if(i<4 && j<4){
    int a,b,c,d, m1, m2;
    // index = threadIdx.x*2 + threadIdx.y*2*8;
    a = max2[i*16 + j*2];
    b = max2[i*16 + j*2 +1];
    c = max2[i*16 + j*2+8];
    d = max2[i*16 + j*2 + 9];
    if(a>=b){
      m1 = a;
    }
    else{
      m1 = b;
    }
    if(c>=d){
      m2 = c;
    }
    else{
      m2 = d;
    }
    if(m1>=m2){
      maxop2d[l*16 + i*4+j]=m1;
    }
    else{
      maxop2d[l*16 + i*4+j] = m2;
    }
  }
}

__global__ void dense1(int *denseip1d, int *weight1d, int *denseop1d){
  int i;
  i=threadIdx.x;
  int k;
  int length;
  length = 64*4*4;
  for(k=0;k<length;k++){
    denseop1d[i] += weight1d[i*length + k]*denseip1d[k];
  }
}

__global__ void dense2(int *denseip2d, int *weight2d, int *denseop2d){
  int i;
  i = threadIdx.x;
  int k;
  int length;
  length =64;
  for(k=0;k<length;k++){
    denseop2d[i]+=weight2d[i*length + k]*denseip2d[k];
  }
  // printf("denseop2d[%d]:%d\n",i,denseop2d[i]);
}

int main(int argc, char **argv){
  int xsize;
  int filterdim;
  int numfilters;
  int numfilters1;
  int numunits;
  int numunits1;
  xsize = 28;
  filterdim = 5;
  numfilters = 32;
  numfilters1 = 64;
  numunits = 64;
  numunits1 =10;

  /*Numbytes required for initial image*/
  int numbytes = xsize*xsize*sizeof(int);
  /*Numbytes require for the output of first convolution layer*/
  int numbytes2 = (xsize-filterdim+1)*(xsize-filterdim+1)*sizeof(int); //24x24
  /**Numbytes required for output of first maxpool layer**/
  int numbytes3 = ((xsize-filterdim+1)*(xsize-filterdim+1)/4)*sizeof(int); //12x12
  /*Numbytes required for the output of second convolution layer*/
  int numbytes4 = ((xsize-filterdim+1)/2 - filterdim + 1)*((xsize-filterdim+1)/2 - filterdim + 1)*sizeof(int);//8x8
  /*Numbytes required for the output of second maxpool layer*/
  int numbytes5 = (numbytes4/4);//4x4
  /*Numbytes required for the weight matrix for the first dense layer*/
  int numbytes6 = (numunits*numfilters1*numbytes5);//64x64x4x4

  /*Image on host side*/
  /*Ip and op to first conv layer*/
  unsigned int *pic = (unsigned int *)malloc(numbytes);
  int *result;
  int *filter;
  /*op to first maxpool layer*/
  int *maxop1;
  /*op of second conv layer*/
  int *cop2;
  int *filter2;
  /*op to second maxpool layer*/
  int *maxop2;
  /*op of first dense layer*/
  int *denseop1;
  int *weight1;
  /*op of second dense layer*/
  int *denseop2;
  int *weight2;

  /*Device side variables*/
  int *filterd;
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
  /*ip and op of first dense layer*/
  int *denseip1d;
  int *denseop1d;
  int *weight1d;
  /*ip and op of second dense layer*/
  int *denseip2d;
  int *denseop2d;
  int *weight2d;

  filter = (int *)malloc( numfilters*filterdim*filterdim*sizeof(int));
  result = (int *)malloc(numfilters*numbytes2);
  maxop1 = (int *)malloc(numfilters*numbytes3);
  cop2 = (int *)malloc(numfilters1*numbytes4);
  filter2 = (int *)malloc(numfilters1*numfilters*filterdim*filterdim*sizeof(int));
  maxop2 = (int *)malloc(numfilters1*numbytes5);
  denseop1 = (int *)malloc(numunits*sizeof(int));
  weight1 = (int *)malloc(numbytes6);
  denseop2 = (int *)malloc(numunits1*sizeof(int));
  weight2 = (int *)malloc(numunits*numunits1*sizeof(int));

  cudaMalloc(&filterd,  numfilters*filterdim*filterdim*sizeof(int));
  cudaMalloc(&resultd, numfilters*numbytes2);
  cudaMalloc(&maxip1d, numfilters*numbytes2);
  cudaMalloc(&maxop1d, numfilters*numbytes3);
  cudaMalloc(&cip2d, numfilters*numbytes3);
  cudaMalloc(&cop2d, numfilters1*numbytes4);
  cudaMalloc(&filter2d, numfilters1*numfilters*filterdim*filterdim*sizeof(int));
  cudaMalloc(&maxip2d, numfilters1*numbytes4);
  cudaMalloc(&maxop2d, numfilters1*numbytes5);
  cudaMalloc(&denseip1d, numfilters1*4*4*sizeof(int));
  cudaMalloc(&denseop1d, numunits*sizeof(int));
  cudaMalloc(&weight1d, numbytes6);
  cudaMalloc(&denseip2d, numunits*sizeof(int));
  cudaMalloc(&denseop2d, numunits1*sizeof(int));
  cudaMalloc(&weight2d, numunits*numunits1*sizeof(int));

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

 /*Initializing the weight matrix for first dense layer*/
 int length = 64*16;
 for(l=0;l<numunits;l++){
   for(i=0;i<length;i++){
     weight1[l*length + i] = 1;
   }
 }

 /*Initializing the weight matrix for second dense layer*/
 for(l=0;l<numunits1;l++){
   for(i=0;i<numunits;i++){
     weight2[l*numunits + i] = 1;
   }
 }
 /******************Code that has everything to do with  kernels****************/
 cudaEvent_t start, stop;
 cudaEventCreate(&start);
 cudaEventCreate(&stop);

 cudaEventRecord(start,0);

 dim3 dimGrid (32);
 dim3 dimBlock (32,32);

 // cudaMemcpy(picd,pic,numbytes, cudaMemcpyHostToDevice);
 // cudaMemcpyToSymbol(FIL, filter, numfilters*filterdim*filterdim*sizeof(int));

 cudaMemcpyToSymbol(PIC, pic, numbytes);
 cudaMemcpy(filterd, filter, numfilters*filterdim*filterdim*sizeof(int), cudaMemcpyHostToDevice);

 conv1<<<dimGrid, dimBlock>>>(filterd,resultd);

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

 cudaMemcpy(maxop2, maxop2d, numfilters1*numbytes5, cudaMemcpyDeviceToHost);

 for(k=0;k<64;k++){
   for(i=0;i<4;i++){
     for(j=0;j<4;j++){
       printf("maxpool[%d][%d]:%d\t",k,i*4+j, maxop1[k*16+i*4+j]);
     }
     printf("\n");
   }
   printf("\n\n");
 }

 cudaMemcpy(denseip1d, maxop2, numfilters1*numbytes5, cudaMemcpyHostToDevice);
 cudaMemcpy(weight1d, weight1, numbytes6, cudaMemcpyHostToDevice);

 dim3 dimGrid3(1);
 dim3 dimBlock3(64);
 dense1<<<dimGrid3, dimBlock3>>>(denseip1d, weight1d, denseop1d);

 cudaMemcpy(denseop1, denseop1d,numunits*sizeof(int),cudaMemcpyDeviceToHost);

 dim3 dimGrid4(1);
 dim3 dimBlock4(10);

 cudaMemcpy(denseip2d, denseop1,numunits*sizeof(int),cudaMemcpyHostToDevice);
 cudaMemcpy(weight2d, weight2, numunits*numunits1*sizeof(int), cudaMemcpyHostToDevice);

 dense2<<<dimGrid4, dimBlock4>>>(denseip2d, weight2d, denseop2d);

 cudaMemcpy(denseop2, denseop2d, numunits1*sizeof(int), cudaMemcpyDeviceToHost);

 cudaEventRecord(stop,0);
 cudaEventSynchronize(stop);
 float milliseconds;
 cudaEventElapsedTime(&milliseconds, start, stop);
 cudaEventDestroy(start);
 cudaEventDestroy(stop);
 printf("Time taken : %f seconds", milliseconds/1000);
}
