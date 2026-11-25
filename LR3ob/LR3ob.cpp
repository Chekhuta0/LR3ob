#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>
#include <windows.h>
using namespace std;
typedef long long ll;
typedef vector<vector<ll>> Mat;
typedef vector<vector<int>> IMat;
const ll INF = 1000000000000000000LL;

#define T(s) string((const char*)u8##s)

int readInt(string msg, int lo, int hi) {
    string s, extra;
    int v;
    while (true) {
        cout << msg;
        if (!getline(cin, s)) exit(0);
        istringstream in(s);
        if (in >> v && !(in >> extra) && v >= lo && v <= hi) return v;
        cout << T("Помилка: введіть ціле число від ") << lo << T(" до ") << hi << "\n";
    }
}

//Завдання 1
double poly(double a, double b) {
    return (3.0 * a * a - 1.0) * b + (5.0 * b * b - 1.0) * a + 0.2;
}

void task1() {
    int n = readInt(T("N (10000..1000000): "), 10000, 1000000);
    int th[2] = { readInt(T("Кількість потоків (варіант 1): "), 1, 64),
                 readInt(T("Кількість потоків (варіант 2): "), 1, 64) };
    vector<double> A(n), B(n), C(n);
    mt19937 gen(random_device{}());
    uniform_real_distribution<double> rnd(-1, 1);
    for (int i = 0; i < n; i++) { A[i] = rnd(gen); B[i] = rnd(gen); }

    double t = omp_get_wtime();
    for (int r = 0; r < 100; r++)
        for (int i = 0; i < n; i++)
            C[i] = poly(A[i], B[i]);
    double tSeq = (omp_get_wtime() - t) / 100;
    double sumSeq = 0;
    for (int i = 0; i < n; i++) sumSeq += C[i];

    cout << T("\nПерші елементи (i, A[i], B[i], C[i]):\n");
    for (int i = 0; i < 5; i++)
        cout << i << ": " << setprecision(6) << A[i] << "  " << B[i] << "  " << C[i] << "\n";
    cout << T("Контрольна сума (послідовно): ") << setprecision(6) << sumSeq << "\n\n";
    cout << setprecision(4);
    cout << T("Послідовно: ") << tSeq * 1000 << T(" мс\n");

    for (int v = 0; v < 2; v++) {
        fill(C.begin(), C.end(), 0.0);
        t = omp_get_wtime();
        for (int r = 0; r < 100; r++) {
#pragma omp parallel for num_threads(th[v])
            for (int i = 0; i < n; i++)
                C[i] = poly(A[i], B[i]);
        }
        double tPar = (omp_get_wtime() - t) / 100;
        double sumPar = 0;
        for (int i = 0; i < n; i++) sumPar += C[i];
        cout << T("Паралельно, ") << th[v] << T(" потоків: ") << tPar * 1000 << T(" мс, прискорення x")
            << tSeq / tPar << T(", контрольна сума: ") << setprecision(6) << sumPar << setprecision(4)
            << (sumPar == sumSeq ? T(" (збігається)") : T(" (НЕ ЗБІГАЄТЬСЯ!)")) << "\n";
    }
}

//Завдання 2 

string findFile(string name) {
    string dirs[] = { "", "tests/", "../", "../tests/", "../../", "../../tests/" };
    string exts[] = { "", ".txt" };
    for (string d : dirs)
        for (string e : exts) {
            ifstream f(d + name + e);
            if (f) return d + name + e;
        }
    return "";
}

bool load(string path, Mat& w, string& err) {
    ifstream f(path);
    vector<vector<string>> rows;
    string line, t;
    while (getline(f, line)) {
        istringstream in(line);
        vector<string> tok;
        while (in >> t) tok.push_back(t);
        if (!tok.empty()) rows.push_back(tok);
    }
    int n = rows.size();
    if (n == 0) { err = T("файл порожній."); return false; }
    w.assign(n, vector<ll>(n));
    for (int i = 0; i < n; i++) {
        if ((int)rows[i].size() != n) {
            err = T("матриця не квадратна: рядків ") + to_string(n) + T(", а в рядку ") + to_string(i + 1) +
                T(" елементів ") + to_string(rows[i].size()) + ".";
            return false;
        }
        for (int j = 0; j < n; j++) {
            string s = rows[i][j];
            char* end;
            ll x = strtoll(s.c_str(), &end, 10);
            if (s == "\xE2\x88\x9E" || s == "inf") w[i][j] = INF; 
            else if (*end != '\0' || x > 2147483647LL || x < -2147483648LL) {
                err = T("елемент [") + to_string(i + 1) + "][" + to_string(j + 1) + T("] = '") + s +
                    T("' не є цілим числом або inf.");
                return false;
            }
            else w[i][j] = x;
        }
        if (w[i][i] != 0) {
            err = T("діагональний елемент [") + to_string(i + 1) + "][" + to_string(i + 1) + T("] має бути 0.");
            return false;
        }
    }
    return true;
}

void step(int n, int i, int k, const vector<ll>& rk, Mat& d, IMat& nx) {
    if (d[i][k] >= INF) return;
    for (int j = 0; j < n; j++)
        if (rk[j] < INF && d[i][k] + rk[j] < d[i][j]) { d[i][j] = d[i][k] + rk[j]; nx[i][j] = nx[i][k]; }
}

void floydSeq(int n, Mat& d, IMat& nx) {
    for (int k = 0; k < n; k++) {
        vector<ll> rk = d[k];
        for (int i = 0; i < n; i++) step(n, i, k, rk, d, nx);
    }
}

void floydPar(int n, Mat& d, IMat& nx, int threads, vector<double>& tm) {
    tm.assign(threads, 0);
    vector<ll> rk(n);
#pragma omp parallel num_threads(threads)
    {
        double mine = 0;
        for (int k = 0; k < n; k++) {
#pragma omp single
            rk = d[k];
            double t = omp_get_wtime();
#pragma omp for nowait
            for (int i = 0; i < n; i++) step(n, i, k, rk, d, nx);
            mine += omp_get_wtime() - t;
#pragma omp barrier
        }
        tm[omp_get_thread_num()] = mine;
    }
}

int visLen(const string& s) {
    int c = 0;
    for (unsigned char ch : s) if ((ch & 0xC0) != 0x80) c++;
    return c;
}
string pad(const string& s, int w) { int l = visLen(s); return string(l < w ? w - l : 0, ' ') + s; }

void printTable(string title, const vector<vector<string>>& a) {
    int n = a.size(), w = to_string(n).size();
    for (auto& r : a) for (auto& s : r) w = max(w, visLen(s));
    w += 2;
    cout << title << "\n    |";
    for (int j = 1; j <= n; j++) cout << pad(to_string(j), w);
    cout << "\n----+" << string(n * w, '-') << "\n";
    for (int i = 0; i < n; i++) {
        cout << pad(to_string(i + 1), 3) << " |";
        for (int j = 0; j < n; j++) cout << pad(a[i][j], w);
        cout << "\n";
    }
    cout << "\n";
}

void task2() {
    cout << T("Ім'я файлу з матрицею (random - випадковий граф): ");
    string name;
    if (!getline(cin, name)) exit(0);
    Mat w;
    if (name == "random") {
        int n = readInt(T("Кількість вершин (2..2000): "), 2, 2000);
        w.assign(n, vector<ll>(n, INF));
        mt19937 gen(random_device{}());
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                if (i == j) w[i][j] = 0;
                else if (gen() % 10 < 3) w[i][j] = 1 + gen() % 100;
            }
    }
    else {
        string path = findFile(name), err;
        if (path.empty()) { cout << T("Помилка: файл не знайдено, спробуйте ще раз.\n"); return; }
        if (!load(path, w, err)) { cout << T("Помилка: ") << err << "\n"; return; }
    }

    int n = w.size(), a = 0, b = 0;
    int mode = readInt(T("Режим (1 - усі пари вершин, 2 - одна пара): "), 1, 2);
    if (mode == 2) {
        a = readInt(T("Початкова вершина (1..") + to_string(n) + "): ", 1, n) - 1;
        b = readInt(T("Кінцева вершина (1..") + to_string(n) + "): ", 1, n) - 1;
    }
    int th[2] = { readInt(T("Кількість потоків (варіант 1): "), 1, 64),
                 readInt(T("Кількість потоків (варіант 2): "), 1, 64) };

    IMat nx0(n, vector<int>(n, -1));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) nx0[i][j] = (i != j && w[i][j] < INF) ? j : -1;

    Mat d = w;
    IMat nx = nx0;
    double t = omp_get_wtime();
    floydSeq(n, d, nx);
    double tSeq = omp_get_wtime() - t;

    for (int i = 0; i < n; i++)
        if (d[i][i] < 0) { cout << T("У графі є цикл від'ємної ваги. Найкоротші шляхи не існують.\n"); return; }

    if (mode == 1 && n > 20) cout << T("(матриці не виводяться, бо їх розмір n > 20)\n");
    else if (mode == 1) {
        vector<vector<string>> dist(n, vector<string>(n)), paths(n, vector<string>(n));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                dist[i][j] = d[i][j] >= INF ? T("∞") : to_string(d[i][j]);
                paths[i][j] = i == j ? to_string(i + 1) : (nx[i][j] < 0 ? "-" : to_string(nx[i][j] + 1));
            }
        printTable(T("Матриця мінімальних відстаней:"), dist);
        printTable(T("Матриця найкоротших шляхів:"), paths);
    }
    else if (d[a][b] >= INF)
        cout << T("Шляху з вершини ") << a + 1 << T(" у вершину ") << b + 1 << T(" не існує.\n");
    else {
        cout << T("Мінімальна відстань ") << a + 1 << " -> " << b + 1 << ": " << d[a][b] << "\n";
        cout << T("Найкоротший шлях: ") << a + 1;
        for (int cur = a, s = 0; cur != b && s < n; s++) { cur = nx[cur][b]; cout << " -> " << cur + 1; }
        cout << "\n";
    }

    cout << T("\nПослідовно (основний потік): ") << tSeq * 1000 << T(" мс\n");
    for (int v = 0; v < 2; v++) {
        Mat d2 = w;
        IMat nx2 = nx0;
        vector<double> tm;
        t = omp_get_wtime();
        floydPar(n, d2, nx2, th[v], tm);
        double tPar = omp_get_wtime() - t;
        cout << T("Паралельно, ") << th[v] << T(" потоків:\n");
        for (int i = 0; i < th[v]; i++) cout << T("  потік ") << i << ": " << tm[i] * 1000 << T(" мс\n");
        cout << T("  загальний час: ") << tPar * 1000 << T(" мс, прискорення x") << tSeq / tPar
            << (d2 == d ? T(" (результат збігається)") : T(" (РЕЗУЛЬТАТ РІЗНИЙ!)")) << "\n";
    }
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    cout << fixed << setprecision(4);
    while (true) {
        cout << T("Завдання (1 - сума поліномів, 2 - Флойд-Уоршелл, exit - вихід): ");
        string s;
        if (!getline(cin, s) || s == "exit") break;
        if (s == "1") task1();
        else if (s == "2") task2();
        else cout << T("Помилка: введіть 1, 2 або exit.\n");
        cout << "\n";
    }
}
