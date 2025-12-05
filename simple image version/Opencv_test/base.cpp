#include <iostream>
#include <opencv2/opencv.hpp>
#include <xfeatures2d.hpp>

using namespace std;
using namespace cv;
template<typename T>
class Stack {
private:
	//int size;
	int top;
	T stack[10];
public:
	void init();
	void push(T element);
	T pop();
};
template<class T>
void Stack<T>::init()
{
	top = 0;
	cout << "Õ»³õÊ¼»¯³É¹¦" << endl;
}
template<class T>
void Stack<T>::push(T element)
{
	if (top < 0) cout << "Õ»´íÎó" << endl;
	else if (top >= 10) cout << "Õ»Âú" << endl;
	else
	{
		stack[top] = element;
		top++;
	}
}
template<class T>
T Stack<T>::pop()
{
	if (top <= 0)
	{
		cout << "Õ»¿Õ" << endl;
		return -1;
	}
	else
	{
		top--;
		return stack[top];
	}
}
int main()
{
	Stack<int> s;
	s.init();
	s.push(1);
	s.push(2);
	s.push(3);
	for (int i = 0; i < 3; i++)
	{
		cout <<"Õ»¶¥ÔªËØÎª:"<< s.pop() << endl;
	}
	return 0;
}