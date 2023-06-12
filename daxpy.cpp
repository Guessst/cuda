#include <iostream>

void daxpy(double scalarA, double* arrayX, double* arrayY, unsigned int length)
{
    for(unsigned int i = 0;i < length;i++)
    {
        arrayY[i] += scalarA * arrayX[i];
    }
}

template <typename T>
void printArray(T* array, unsigned int length)
{
    unsigned int i;
    for(i = 0;i < length - 1;i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << array[i] << " " << std::endl;
}

int main()
{
    double scalarA = 2;
    double arrayX[] = {1, 2, 3, 4, 5};
    unsigned int lengthOfX = sizeof(arrayX) / sizeof(*arrayX);
    
    double arrayY[] = {1, 1, 1, 1, 1};
    unsigned int lengthOfY = sizeof(arrayY) / sizeof(*arrayY);


    std::cout << "a: " << scalarA << std::endl;

    std::cout << "X (length=" << lengthOfX << "): ";
    printArray(arrayX, lengthOfX);
    
    std::cout << "Y (length=" << lengthOfY << "): ";
    printArray(arrayY, lengthOfY);

    std::cout << "-----------------------" << std::endl;

    daxpy(scalarA, arrayX, arrayY, lengthOfY);
    
    std::cout << "Y: ";
    printArray(arrayY, lengthOfY);

    return 0;
}