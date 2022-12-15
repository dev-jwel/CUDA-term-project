#include <stdio.h>

int test_math();
int test_index();

void check(int (*test_func)(), char *func_name, int *ret) {
    printf("%s\n", func_name);
    *ret = test_func();
}

int main() {
    int ret = 0;

    printf("start test\n");
    
    check(test_math, "test_math", &ret);
    check(test_index, "test_index", &ret);

    return ret;
}