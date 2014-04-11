
#include <iostream>
#include <armadillo>
#include <map>
#include <vector>

using namespace std;
using namespace arma;

void printF(int a, int b, int c = 1000, int d = -9) {
    cout << a << b << c << d << endl;
}

void testStd() {
    mat a;
    a << 1 << 2 << 3 << endr
            << 4 << 5 << 16 << endr
            << 7 << 8 << 9 << endr;
    rowvec b = stddev(a);
    b.print("std dev b");
    mat c = median(a);
    c.print("median a");
}

void testSqrt() {
    uword lol = 123;
    cout << "sqrt(lol)" << std::sqrt(lol) << endl;
}

void testFind() {
    //find the '%' operation

    icolvec c1 = randi<icolvec>(10, distr_param(0, +20));
    c1.print();
    icolvec c2 = c1 / 5 - 1;
    c2.print("c2");


    //cube r = y % y;     // element-wise cube multiplication
    imat c3 = randi<imat>(5, 5, distr_param(0, +50));
    c3.print("c3");
    ucolvec c4 = find(c3 > 25);
    c4.print("c4");
    ucolvec c5 = c4 / c3.n_rows;
    c5.print("c5");
    ucolvec c6;
    c6.copy_size(c4);
    c6.fill(c3.n_rows);
    c6.print("c6");
    //as_scalar(c4 % c6);
    //find remainder
    ucolvec c7 = c4 - c4 / 5 * 5;
    c7.print("c7");
}

void testString() {
    string abc = "abcdefg.txt";
    cout << "abc SIZE" << abc.size() << endl;
    cout << abc.substr(0, abc.size() - 1) << endl;
    ;
}

void testEPS() {
    mat A = randu<mat>(4, 5);
    mat B = eps(A);
    A.print("A");
    B.print("B");
}

//enable of LAPACK needed

void testRANK() {
    mat A = randu<mat>(4, 5);
    uword r = rank(A);
    A.print("A");
    cout << "r " << r << endl;

}

void testCOR() {
    mat X = randu<mat>(4, 5);
    mat Y = randu<mat>(4, 5);

    mat R = cor(X, Y);
    mat S = cor(X, Y, 1);
    X.print("X");
    Y.print("Y");
    R.print("R");
    S.print("S");
}

void testCOR2() {
    vec a = randu<vec>(10);
    a.print("a");
    vec b = cor(a, a, 0);
    b.print("b");
}

void testHIST() {
    vec v = randn<vec>(100); // Gaussian distribution

    uvec h1 = hist(v, 11);
    uvec h2 = hist(v, linspace<vec>(-2, 2, 11));
    h1.print("h1");
    h2.print("h2");
}

//fail

void testNewInLoop() {
    colvec a;
    for (int i = 0; i < 20; i++)
        a << i << endr;
    a.print("a ");
}

void findCorr() {
    colvec a(20);
    for (int i = 0; i < 20; i++)
        a(i) = i;
    double std = arma::stddev(a);
    double sum = 0;
    for (int i = 0; i < 19; i++) {
        sum += ((a(i) - 9.5) * (a((i + 1)) - 9.5));
    }
    double result = sum / 20 / std / std;
    cout << mean(a) << endl;
    cout << "result: " << result << endl;
}

void findCorr2() {
    colvec a(20);
    for (int i = 0; i < 20; i++)
        a(i) = i;
    double std = arma::stddev(a);
    double sum = 0;
    for (int i = 0; i < 19; i++) {
        sum += ((a(i) - 9.5) * (a((i + 1)) - 9.5));
    }
    double result = sum / 19 / std / std;
    cout << mean(a) << endl;
    cout << "result: " << result << endl;
}

//same result as matlab?

void findCorr3() {
    colvec a(20);
    for (int i = 0; i < 20; i++)
        a(i) = i;
    double std = arma::stddev(a);
    double sum = 0;
    for (int i = 0; i < 17; i++) {
        sum += ((a(i) - 9.5) * (a((i + 3)) - 9.5));
    }
    double result = sum / 19 / std / std;
    cout << mean(a) << endl;
    cout << "result: " << result << endl;
}

void findCorr4() {
    colvec a(20);
    for (int i = 0; i < 20; i++)
        a(i) = i + 1;
    double std = arma::stddev(a);
    double mean = arma::mean(a);
    double sum = 0;
    for (int i = 0; i < 1; i++) {
        sum += ((a(i) - mean) * (a((i + 19)) - mean));
    }
    double result = sum / 19 / std / std;
    cout << "result: " << result << endl;
}

void testStep() {
    arma::colvec a;
    //a(0:10,1);
    a.print("a");
}

void testForIni() {

    arma::colvec a(100000001);
    const double begin = (double) clock() / CLK_TCK;
    for (int i = 0; i < 100000001; i++)
        a(i) = i + 1;
    const double end = (double) clock() / CLK_TCK;
    cout << "ini time: " << end - begin << endl;
    //10000000 0.08s
    //100000001 0.803s
}

void testDot() {
    colvec a = linspace<colvec>(1, 3, 3);
    colvec b = linspace<colvec>(4, 6, 3);
    a.print("a");
    b.print("b");
    //double sum = as_scalar(a*b);
    double sum2 = dot(a, b);
    //cout << "sum " << sum << endl;
    cout << "sum2 " << sum2 << endl;
}

double corr2(arma::colvec data, int step) {
    int size = data.size();
    double std = arma::stddev(data);
    double mean = arma::mean(data);
    colvec temp1 = data.rows(0, data.size() - step - 1) - mean;
    temp1.print("temp1");
    colvec temp2 = data.rows(step, data.size() - 1) - mean;
    double sum = dot(temp1, temp2);
    cout << sum << endl;
    temp2.print("temp2");
    //double sum = arma::dot(data.rows(0,data.size() - step - 1) - mean, data.rows(step,data.size() - 1) - mean);
    //even without returning any values, still no error report
    return sum / (size - 1) / std / std;
}

void testSort() {
    vec a = randu<vec>(10);
    a.print("a");
    arma::uvec b = sort_index(a, "descend");
    b.print("b");
}

void testJoin() {
    vec a = randu<vec>(5);
    a.print("a");
    vec b = randu<vec>(5);
    b.print("b");
    a = arma::join_vert(a, b);
    a.print("a1");
}

void testVecCmpu(){
    vec a = linspace<vec>(0,20,5);
    vec b = linspace<vec>(10,50,5);
    vec c = a % b;
    c.print("c");
    a = pow(a,2);
    a.print("a");
}

//void testConstruct(){
//    mat44 G;
//    G.randn();
//    G.print("G");
//    mat a("1 2 3; 4 5 6; 7 7 7 ");
//    a.print("a");
//    mat55 c;
//    c.randu();
//    c.print("c");
//    
//    //c.print("c");
//}

//void testImbue(){
//    vec a = randu<vec>(5);
//    a.print("a");
//    vec c = linspace<vec> (1,5,5);
//    vec d = linspace<vec> (6,10,5);
//    a.imbue [&]() { return c + d; } ;
//}

template <class T>
T testTem(T val, int N) {
    return val * N;
}

void testComp() {
    vec a("1,2,3");
    vec b("4,5,6");
    uvec c = b > a;
    c.print("c");
    b - a > 0;
    //b - a creates a vector of 1 or 0
    //    if (b - a > 0)
    //        cout << "hello" << endl;
}

void testF() {
    vec a("1,2,3,2");
    cout << (a != 2) << endl;
}

// mat array can have variable size

void testArrayMat(int size) {
    vec m[size];
    for (int i = 0; i < size; i++) {
        m[i] = randu<vec> (10);
        m[i].fill(i + 1);
        cout << endl;
        m[i].print();
        char fileNo[5];
        itoa(i, fileNo, 10);
        string fileName = string("constant") + fileNo + string(".mat");
        m[i].save(fileName.c_str());
    }
}

void test3Cmp() {
    mat m[3];
    for (int i = 0; i < 3; i++) {
        m[i] = randu<mat>(5, 5);
        m[i].print("matrix");
    }
    uvec abc = find(m[0] + m[1] > m[2]);
    abc.print("rsus");
}

int add(int i, int j) { return i+j; }
int sub(int i, int j) { return i-j; }
void functionMap(string a, int b, int c){
    typedef int (*FnPtr)(int, int);
    std::map<std::string, FnPtr> myMap;
    myMap["add"] = add;
    myMap["sub"] = sub;
    
    cout << myMap[a](b,c) << endl;
}

void testMap(){
    map<int, string> map1;
    map1[2] = "hello";
    cout << map1[2] << endl;
    cout << map1[0] << endl;
}

void testVector(){
    vec a = linspace<vec>(-2, 2, 11);
    vec b = a;
    vec c = b;
    vector<vec> data;
    data.push_back(a);
    data[0].print("vectors of vec");
}

void testChangeType(){
    uvec a = linspace<uvec>(0, 200, 11);;
    // cannot just like normal just cast the type
    //vec b = (vec)a;
    a.print("a");
    vec b = conv_to<vec>::from(a);
    b.print("b");
}

void testString2(){
    string a = "abc lol";
    if (a.substr(0,3) == "abc")
        cout << "compare" << endl;
}

void testReadLongDouble(){
    ifstream myfile("a.txt");
    string line;
    getline(myfile,line);
    cout << "readl number " << line << endl;
    double l=acos(-1.0L) ;
    l = atof(line.c_str());
    double s = atof(line.c_str());
    
    cout.precision(50); 
    cout << "long:  " << l << endl;
    cout << "short: " << s << endl;
    
    myfile.close();
}

void testReadOp(){
    ifstream myfile("a.txt");
    string line;
    getline(myfile,line);
//    cout << line.find_first_of("+-*/^");
//    cout << line << endl;
//    line.erase(0,line.find_first_of("+-*/^") + 1);
    cout << line << endl;
//    string var = line.substr(0, line.find('+'));
//    cout << var << endl;
    
//    while(line.find_first_of("+-*/^;") > 0){
//        string var = line.substr(0, line.find_first_of("+-*/^;"));
//        if (var.at(var.size()-1) == ';')
//            break;
//        cout << var << endl;
//        line.erase(0, line.find_first_of("+-*/^;") + 1);
//        cout << line.find_first_of("+-*/^;") << endl;
//    }
    
    int index = 0;
    int counter = 0;
    do {
        cout << "couter " <<  ++counter << endl;
        index = line.find_first_of("+-*/^;");
        string var = line.substr(0, index );
        cout << var << endl;
        
        if (line.at(index) == ';'){
            break;
        }
        line.erase(0, index + 1);
    } while (index > 0);
}

bool isParam(string line){
    return isdigit(atoi(line.c_str()));
}

bool is_number(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}

void testDigit(){
    string a = "abc";
    string b = "123";
    if (is_number(a))
        cout << "abc" << endl;
    if (is_number(b))
        cout << b << endl;
}

void testIsdigit(){
    int i = 1;
    int a = 100;
    double b = 99.2;
    string abc = "sdf";
        if (std::isdigit(i))
        cout << i << endl;
    if (std::isdigit(a))
        cout << a << endl;
    if (std::isdigit(b))
        cout << b << endl;
//    if (std::isdigit(abc))
//        cout << abc << endl;    
}



main() {
//    int a = 5;
//    int b = 6;
//    int c = 7;
//    int d = 8;
//    printF(a, b);
//    printF(a, b, c);
//    printF(a, b, c, d);
//    //create a test find the usage of 'find'
//    mat A = randu<mat>(5, 5);
//    mat B = randu<mat>(5, 5);
//
//    A.print("mat A");
//    B.print("mat B");
//
//
//    uvec q1 = find(A >= B);
//    //q1.print("q1 = find(A > B)");
//    uvec q2 = find(A > 0.5);
//    uvec q3 = find(A > 0.5, 3, "last");
    //    testCOR2();
    //    testStd();
    //    testSqrt();
    //    testString();
    //    testEPS();
    //      testRANK();
    //    testCOR();
    //    testHIST();
    //    vec q4(4,fill::(5));
    //    q4.print("q4");
    //testNewInLoop();
    //    findCorr3();
    //    findCorr4();
    //    testForIni();
    //    testDot();
    //    arma::colvec te = arma::linspace<arma::colvec>(1,20,20);
    //    cout << corr2(te,1) << endl;
    //    
    //    testSort();
    //    testJoin();
    //    testConstruct();
    //    colvec a1 = arma::linspace<arma::colvec>(1,20,20);
    //    colvec a2 = testTem(a1,2);
    //    mat b1 = randu<mat>(5,5);
    //    mat b2 = testTem(b1,2);
    //    a2.print("a1");
    //    b2.print("b2");
    //    testComp();
    //    testF();
//    testArrayMat(4);
//    test3Cmp();
//    functionMap("sub",4,5);
//    cout << abs(-3) << endl;
//    std::abs();
//    testMap();
//    testVector();
//    testChangeType();
//    testString2();
//    testVecCmpu();
//    testReadLongDouble();
//    testReadOp();
//    string a = "sdfsdf";
//    string b = a;
//    cout << a << " " << b;
//    testDigit();
//    testArrayMat(4);
    testIsdigit();
}

