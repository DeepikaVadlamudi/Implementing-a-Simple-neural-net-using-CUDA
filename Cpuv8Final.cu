/*Done with cpu version of convolution which can be scaled to any number of filters of
size 5x5. Maxpooling done with size 2x2. */
/*Implemented 2 layers of conv2d and maxpool using single 1D array */
/*implemented two dense layers*/
/*To do: measure time taken*/
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"
#include<time.h>

int Maxpooling(int a, int b, int c, int d){
	int temp,i;
	temp = a;
	if(b> c && b>d && b> temp){
		temp = b;
	//	printf("temp is b\n");
	}
	if(c>b && c>d && c> temp){
		temp = c;
		//printf("temp is c\n");
	}
	if(d> c && d>b && d> temp){
		temp = d;
	//	printf("temp is d\n");
	}
//	printf("temp is a\n");
	return temp;
}
int main( int argc, char **argv )
{

	int xsize, filterdim, numfilters, numfilters1, numweights, numweights1;
  xsize = 28;
  filterdim =5;
  numfilters=32;
	numfilters1=64;
	numweights = 64;
	numweights1 = 10;
	/******num bytes required for the initial input********/
	int numbytes =  xsize * xsize * sizeof( int );
	/*******num bytes required for input to conv1********/
  int numbytes2 =  (xsize-filterdim + 1) * (xsize - filterdim +1);
	/*********num bytes required for the input to conv2****/
	int numbytes3 = ((xsize-filterdim + 1)/2 -filterdim + 1) * ((xsize-filterdim + 1)/2 -filterdim + 1);
	/************num of bytes required for first weight matrix**************/
	int numbytes4 =  numfilters1*numweights*numbytes3/4;
	/************num of bytes required for second weight matrix************/
	int numbytes5 = numweights*numweights1;

	/*****Original input - pic*************/
  unsigned int *pic = (unsigned int *)malloc(numbytes);
	/*****filter of the first conv layer*****/
  unsigned int filter[numfilters*filterdim*filterdim];
	/*******filter for the second conv layer*******/
	unsigned int filter2[numfilters1*filterdim*filterdim];
	/*********weight matrix for the first dense layer**********/
	unsigned int weight1[numbytes4];
	/*********weight matrix for the second dense layer**********/
	unsigned int weight2[numbytes5];

	int result[numfilters*numbytes2];
	int result2[numfilters1*numbytes3];
	int maxpool[numfilters*(((xsize-filterdim + 1)*(xsize-filterdim + 1))/4)];
	int maxpool2[numfilters1*(numbytes3/4)];
	int dense1[numweights];
	int dense2[numweights1];

	int i, j;
  int count;
  int sum1,k,l;
	int dimx;
	dimx = numfilters1*(numbytes3/4);

	/*************Should read in input**********/
	for (i=0; i<xsize; i++) {
		for (j=0; j<xsize; j++) {
			pic[i*xsize + j] = 1;
    //  printf("pic[%d][%d] : %d\t",i,j,pic[i*xsize + j]);
		}
  //  printf("\n");
	}
	/******should read in filters*********/
  for(int k=0;k<numfilters;k++){
  	for (int i=0; i<filterdim; i++) {
    	for (int j=0; j<filterdim; j++){
        filter[k*(filterdim*filterdim) + i*filterdim + j] = 1;
	//			printf("filter[%d][%d]: %d\n",k, i*filterdim + j, filter[k*(filterdim*filterdim) + i*filterdim + j]);
    	}
  	 }
	}

	for(int k=0;k<numfilters1;k++){
  	for (int i=0; i<filterdim; i++) {
    	for (int j=0; j<filterdim; j++){
        filter2[k*(filterdim*filterdim) + i*filterdim + j] = 1;
	//			printf("filter2[%d][%d]: %d\n",k, i*filterdim + j, filter2[k*(filterdim*filterdim) + i*filterdim + j]);
    	}
  	 }
	}
	/*********First weight matrix**************/
	for(l=0;l<numweights;l++){
		for(i=0;i<dimx;i++){
			weight1[l*dimx+i] = 1;
			printf("element1 : %d\n", l*dimx+i);
		}
	}
	/**********Second weight matrix**************/
	for(l=0;l<numweights1;l++){
		for(i=0;i<numweights;i++){
			weight2[l*numweights+i] = 1;
			printf("element2 : %d\n", l*numweights+i);
		}
	}
	clock_t start, end;
	start  = clock();
	/*****Operations of first convolutional layer******/
	for(l=0; l < numfilters; l++){
	  count = 0;
		for (i = 0;  i < xsize - filterdim +1; i++){
			for (j = 0; j < xsize - filterdim+1; j++){


	      k =0;

	    sum1 =  (filter[l*(filterdim*filterdim) + k])*pic[ xsize * (i) + j ] + (filter[l*(filterdim*filterdim) + k+1])*pic[ xsize*(i) + (j+1) ]
				+ filter[l*(filterdim*filterdim)+ k+2]*pic[ xsize * (i)+(j+2)] + filter[l*(filterdim*filterdim) +k+3]*pic[xsize * (i)+(j+3)]
				+ filter[l*(filterdim*filterdim) +k+4]*pic[ xsize * (i)+(j+4)]+ filter[l*(filterdim*filterdim) + k+5]*pic[ xsize*(i+1)+(j) ]
				+ filter[l*(filterdim*filterdim) +k+6]*pic[ xsize * (i+1) + (j+1) ] + filter[l*(filterdim*filterdim) + k+7]*pic[ xsize*(i+1) + (j+2) ] +
				filter[l*(filterdim*filterdim) +k+8]*pic[ xsize*(i+1) + (j+3) ] + filter[l*(filterdim*filterdim) +k+9]*pic[ xsize*(i+1) + (j+4) ] +
	      filter[l*(filterdim*filterdim) +k+10]*pic[ xsize*(i+2) + (j) ]	+ filter[l*(filterdim*filterdim) +k+11]*pic[ xsize * (i+2) + (j+1) ] +
	      filter[l*(filterdim*filterdim) +k+12]*pic[ xsize*(i+2) + (j+2)] + filter[l*(filterdim*filterdim) +k+13]*pic[ xsize*(i+2) + (j+3)]
				+filter[l*(filterdim*filterdim) +k+14]*pic[ xsize*(i+2) + (j+4)] + filter[l*(filterdim*filterdim) +k+15]*pic[ xsize*(i+3) + (j)]
				+ filter[l*(filterdim*filterdim) +k+16]*pic[ xsize*(i+3) + (j+1)] + filter[l*(filterdim*filterdim) +k+17]*pic[ xsize*(i+3) + (j+2)]
				+ filter[l*(filterdim*filterdim) +k+18]*pic[ xsize*(i+3) + (j+3)] + filter[l*(filterdim*filterdim) +k+19]*pic[ xsize*(i+3) + (j+4)]
				+ filter[l*(filterdim*filterdim) +k+20]*pic[ xsize*(i+4) + (j)] +filter[l*(filterdim*filterdim) +k+21]*pic[ xsize*(i+3) + (j+1)]
				+ filter[l*(filterdim*filterdim) +k+22]*pic[ xsize*(i+4) + (j+2)] + filter[l*(filterdim*filterdim) +k+23]*pic[ xsize*(i+4) + (j+3)]
				+ filter[l*(filterdim*filterdim) + k+24]*pic[ xsize*(i+4) + (j+4)];


	      result[l*numbytes2 +count] = sum1;
	//      printf("result[%d][%d]=%d\t",l,count,result[l*numbytes2 + count]);
	      count+=1;
			}
//	  	printf("\n");
		}
//		printf("\n\n\n");
	}
	/***************Maxpool***************************/
	for(l=0; l<numfilters; l++){
		count =0;
		for(j=0;j<(xsize-filterdim + 1);j+=2){
				for(i=0;i<(xsize-filterdim + 1);i+=2){
					maxpool[l*(numbytes2/4) + count] =
					Maxpooling(result[l*(xsize - filterdim +1)*(xsize - filterdim +1) + j*(xsize-filterdim+1)+i],
				result[l*(xsize - filterdim +1)*(xsize - filterdim +1) + j*(xsize-filterdim+1)+i+1],
				result[l*(xsize - filterdim +1)*(xsize - filterdim +1) + j*(xsize-filterdim+1)+i+(xsize-filterdim + 1)],
			result[l*(xsize - filterdim +1)*(xsize - filterdim +1) + j*(xsize-filterdim+1)+i+(xsize-filterdim + 1)+1]);
		//	printf("Maxpool[%d][%d] : %d \t",l,count ,maxpool[l*(numbytes2/4) + count]);
			count+=1;
				}
	//			printf("\n");
		}
//		printf("\n\n\n");
	}
	/******Operations of second convolutional layer*******/
	int dim = (xsize-filterdim + 1)/2;
	dimx = (xsize-filterdim + 1)/2 -filterdim + 1;
	printf("dim: %d ; dimx: %d\n", dim,dimx);
	for(l=0; l < numfilters1; l++){
		count = 0;
		for (i = 0;  i < dimx; i++){
			for (j = 0; j < dimx; j++){

				k =0;

				sum1 =  (filter2[l*(filterdim*filterdim) + k])*maxpool[ dim * (i) + j ] + (filter2[l*(filterdim*filterdim) + k+1])*maxpool[ dim*(i) + (j+1) ]
				+ filter2[l*(filterdim*filterdim)+ k+2]*maxpool[ dim * (i)+(j+2)] + filter2[l*(filterdim*filterdim) +k+3]*maxpool[dim * (i)+(j+3)]
				+ filter2[l*(filterdim*filterdim) +k+4]*maxpool[ dim * (i)+(j+4)]+ filter2[l*(filterdim*filterdim) + k+5]*maxpool[ dim*(i+1)+(j) ]
				+ filter2[l*(filterdim*filterdim) +k+6]*maxpool[ dim * (i+1) + (j+1) ] + filter2[l*(filterdim*filterdim) + k+7]*maxpool[ dim*(i+1) + (j+2) ] +
				filter2[l*(filterdim*filterdim) +k+8]*maxpool[ dim*(i+1) + (j+3) ] + filter2[l*(filterdim*filterdim) +k+9]*maxpool[ dim*(i+1) + (j+4) ] +
				filter2[l*(filterdim*filterdim) +k+10]*maxpool[ dim*(i+2) + (j) ]	+ filter2[l*(filterdim*filterdim) +k+11]*maxpool[ dim * (i+2) + (j+1) ] +
				filter2[l*(filterdim*filterdim) +k+12]*maxpool[ dim*(i+2) + (j+2)] + filter2[l*(filterdim*filterdim) +k+13]*maxpool[ dim*(i+2) + (j+3)]
				+filter2[l*(filterdim*filterdim) +k+14]*maxpool[ dim*(i+2) + (j+4)] + filter2[l*(filterdim*filterdim) +k+15]*maxpool[ dim*(i+3) + (j)]
				+ filter2[l*(filterdim*filterdim) +k+16]*maxpool[ dim*(i+3) + (j+1)] + filter2[l*(filterdim*filterdim) +k+17]*maxpool[ dim*(i+3) + (j+2)]
				+ filter2[l*(filterdim*filterdim) +k+18]*maxpool[ dim*(i+3) + (j+3)] + filter2[l*(filterdim*filterdim) +k+19]*maxpool[ dim*(i+3) + (j+4)]
				+ filter2[l*(filterdim*filterdim) +k+20]*maxpool[ dim*(i+4) + (j)] +filter2[l*(filterdim*filterdim) +k+21]*maxpool[ dim*(i+3) + (j+1)]
				+ filter2[l*(filterdim*filterdim) +k+22]*maxpool[ dim*(i+4) + (j+2)] + filter2[l*(filterdim*filterdim) +k+23]*maxpool[ dim*(i+4) + (j+3)]
				+ filter2[l*(filterdim*filterdim) + k+24]*maxpool[ dim*(i+4) + (j+4)];


				result2[l*numbytes3 +count] = sum1;
		//		printf("result2[%d][%d]=%d\t",l,count,result2[l*numbytes3 + count]);
				count+=1;
			}
		//	printf("\n");
		}
	//	printf("\n\n\n");
	}
	/******Second Maxpool Layer******/
	dim =((xsize-filterdim + 1)/2 -filterdim + 1)/2;
	printf("dim: %d ;dimx: %d; numbytes3: %d", dim, dimx, numbytes3);
	for(l=0; l<numfilters1; l++){
		count =0;
		for(j=0;j<dimx;j+=2){
				for(i=0;i<dimx;i+=2){
					maxpool2[l*(numbytes3/4) + count] =
					(Maxpooling(result2[l*numbytes3 + j*dimx+i],
					result2[l*numbytes3 + j*dimx+i+1],
					result2[l*numbytes3 + j*dimx+i+dimx],
					result2[l*numbytes3 + j*dimx+i+dimx+1]))/625;

					printf("Maxpool2[%d][%d] : %d \t",l,count ,maxpool2[l*(numbytes3/4) + count]);
					count+=1;
				}
				printf("\n");
		}
		printf("\n\n\n");
	}
	/********************First Dense layer**********************/
	dimx = numfilters1*(numbytes3/4);

	for(l=0;l<numweights;l++){
		for(i=0;i<dimx;i++){
			dense1[l]+= weight1[l*dimx+i]*maxpool2[i];
		}
		printf("dense1[%d]:%d\n",l,dense1[l]);
	}

	/*************Second Dense Layer***************/
	dimx = numweights;
	for(l=0;l<numweights1;l++){
		for(i=0;i<dimx;i++){
			dense2[l]+= weight2[l*dimx+i]*dense1[i];
		}
		printf("dense2[%d]:%d\n",l,dense2[l]);
	}
	end = clock();
	printf("time taken by cpu : %f seconds",((double) (end - start)) );
}
