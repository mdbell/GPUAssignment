/*
Christopher Ginac

image.cpp
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <iostream>
#include "Image.cuh"
#include <cmath>
using namespace std;
const int ntpb = 1024;

__global__ void enlarge(int* a, int* b, int sz, int scale, int cols, int scols) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int x = idx / scols;
	int y = idx % scols;
	if (idx < sz) {
		a[idx] = b[(x / scale) * cols + (y / scale)];
		//a[idx] = 0xFFFFFFFF;
	}
}

__global__ void negate(int* a, int* b, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		a[idx] = -(b[idx]) + 255;
	}
}

__global__ void verticalReflect(int* a, int* b, int sz, int n, int m) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int x = idx / m;
	int y = idx % m;
	if (idx < sz) {
		//a[idx] = b[x * m + (m - y)];
		a[x * m + (m - y)] = b[idx];
	}
}

__global__ void horizontalReflect(int* a, int* b, int sz, int n, int m) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int x = idx / m;
	int y = idx % m;
	if (idx < sz) {
		a[(n - x) * m + y] = b[idx];
	}
}

Image::Image()
/* Creates an Image 0x0 */
{
	N = 0;
	M = 0;
	Q = 0;

	pixelVal = NULL;
}

Image::Image(int numRows, int numCols, int grayLevels)
/* Creates an Image of numRows x numCols and creates the arrays for it*/
{

	N = numRows;
	M = numCols;
	Q = grayLevels;
	cudaMalloc((void**)&pixelVal, N * M * sizeof(int));
	cudaMemset(pixelVal,0,N * M * sizeof(int));
	/*	pixelVal = new int[N * M];
	for (int i = 0; i < N; i++)
	{
	for (int j = 0; j < M; j++)
	pixelVal[i * M + j] = 0;
	}
	*/
}

Image::~Image()
/*destroy image*/
{
	if (pixelVal) {
		cudaFree(pixelVal);
	}
	//delete pixelVal;
}

Image::Image(const Image& oldImage)
/*copies oldImage into new Image object*/
{
	N = oldImage.N;
	M = oldImage.M;
	Q = oldImage.Q;
	int sz = M * N * sizeof(int);
	cudaMalloc((void**)&pixelVal, sz);
	cudaMemcpy(pixelVal, oldImage.pixelVal, sz, cudaMemcpyDeviceToDevice);
}

void Image::operator=(const Image& oldImage)
/*copies oldImage into whatever you = it to*/
{
	N = oldImage.N;
	M = oldImage.M;
	Q = oldImage.Q;

	if (pixelVal) {
		cudaFree(pixelVal);
	}

	int sz = M * N * sizeof(int);
	cudaMalloc((void**)&pixelVal, sz);
	cudaMemcpy(pixelVal, oldImage.pixelVal, sz, cudaMemcpyDeviceToDevice);
}

void Image::setImageInfo(int numRows, int numCols, int maxVal)
/*sets the number of rows, columns and graylevels*/
{
	N = numRows;
	M = numCols;
	Q = maxVal;
}

void Image::getImageInfo(int &numRows, int &numCols, int &maxVal)
/*returns the number of rows, columns and gray levels*/
{
	numRows = N;
	numCols = M;
	maxVal = Q;
}

int Image::getPixelVal(int row, int col)
/*returns the gray value of a specific pixel*/
{
	int i = 0;
	int* idx = pixelVal + (row * M + col);
	cudaMemcpy(&i, idx, sizeof(int), cudaMemcpyDeviceToHost);
	return i;
}


void Image::setPixelVal(int row, int col, int value)
/*sets the gray value of a specific pixel*/
{
	int* idx = pixelVal + (row * M + col);
	cudaMemcpy(idx, &value, sizeof(int), cudaMemcpyHostToDevice);
//	pixelVal[row * M + col] = value;
}

void Image::getPixels(int row, int col, int sz, int* out) {
	cudaMemcpy(out, pixelVal + (row * M + col), sz * sizeof(int), cudaMemcpyDeviceToHost);
}

void Image::setPixels(int row, int col, int sz, int* in) {
		cudaMemcpy(pixelVal + (row * M + col), in, sz * sizeof(int), cudaMemcpyHostToDevice);
}

void Image::enlargeImage(int value, Image& oldImage)
/*enlarges Image and stores it in tempImage, resizes oldImage and stores the
larger image in oldImage*/
{
	int rows, cols, gray;
	int pixel;
	int enlargeRow, enlargeCol;

	rows = oldImage.N * value;
	cols = oldImage.M * value;
	gray = oldImage.Q;

	Image tempImage(rows, cols, gray);


	int r = oldImage.N;
	int c = oldImage.M;

	int* d_temp = tempImage.pixelVal;
	int* d_img = oldImage.pixelVal;
	int size = rows * cols;
	int nblocks = size / ntpb;

	//cudaMalloc((void**)&d_temp, size * sizeof(int));
	//cudaMalloc((void**)&d_img, size * sizeof(int));

	//cudaMemcpy(d_temp, tempImage.pixelVal, size * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_img, oldImage.pixelVal, (r * c) * sizeof(int), cudaMemcpyHostToDevice);

	enlarge << <nblocks, ntpb >> >(d_temp, d_img, size, value, c, cols);

	cudaDeviceSynchronize();

	//set the image's data
	//cudaMemcpy(tempImage.pixelVal, d_temp, size * sizeof(int), cudaMemcpyDeviceToHost);

	//free device mem
	//cudaFree(d_temp);
	//cudaFree(d_img);
	/*

	for (int i = 0; i < oldImage.N; i++)
	{
	for (int j = 0; j < oldImage.M; j++)
	{
	pixel = oldImage.pixelVal[i * oldImage.M + j];
	enlargeRow = i * value;
	enlargeCol = j * value;
	for (int c = enlargeRow; c < (enlargeRow + value); c++)
	{
	for (int d = enlargeCol; d < (enlargeCol + value); d++)
	{
	tempImage.pixelVal[c * cols + d] = pixel;
	}
	}
	}
	}
	/**/
	oldImage = tempImage;
}

void Image::reflectImage(bool flag, Image& oldImage)
/*Reflects the Image based on users input*/
{
	int rows = oldImage.N;
	int cols = oldImage.M;
	Image tempImage(oldImage);

	int* d_temp = tempImage.pixelVal;
	int* d_img = oldImage.pixelVal;
	
	int size = rows * cols;
	int nblocks = size / ntpb;
	//cudaMalloc((void**)&d_temp, size * sizeof(int));
	//cudaMalloc((void**)&d_img, size * sizeof(int));
	//cudaMemcpy(d_temp, tempImage.pixelVal, size * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_img, oldImage.pixelVal, size * sizeof(int), cudaMemcpyHostToDevice);
	if (flag) {
		horizontalReflect << <nblocks, ntpb >> >(d_temp, d_img, size, rows, cols);
	}
	else {
		verticalReflect << <nblocks, ntpb >> >(d_temp, d_img, size, rows, cols);
	}
	cudaDeviceSynchronize();
	//cudaMemcpy(tempImage.pixelVal, d_temp, size * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaFree(d_temp);
	//cudaFree(d_img);

	oldImage = tempImage;
}

void Image::negateImage(Image& oldImage)
/*negates image*/
{
	Image tempImage(N, M, Q);

	/*for (int i = 0; i < rows; i++)
	{
	for (int j = 0; j < cols; j++)
	tempImage.pixelVal[i * cols + j] = -(pixelVal[i * cols + j]) + 255;
	}*/

	int* d_temp = tempImage.pixelVal;
	int* d_img = pixelVal;
	int size = N * M;
	int nblocks = size / ntpb;
	
	//cudaMalloc((void**)&d_temp, size * sizeof(int));
	//cudaMalloc((void**)&d_img, size * sizeof(int));
	//cudaMemcpy(d_temp, tempImage.pixelVal, size * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_img, pixelVal, size * sizeof(int), cudaMemcpyHostToDevice);

	negate << <nblocks, ntpb >> >(d_temp, d_img, size);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaDeviceSynchronize();
	//cudaMemcpy(tempImage.pixelVal, d_temp, size * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaFree(d_temp);
	//cudaFree(d_img);

	oldImage = tempImage;
}
