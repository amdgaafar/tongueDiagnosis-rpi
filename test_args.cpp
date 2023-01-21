#include <iostream> 
using namespace std;

int main (int argc, char *argv[]){
    cout << "argc: " << argc << endl << endl;
    for(int i = 0; i < argc; i++)
    {
        cout << "argv: " << argv[i] << endl;
    }
}