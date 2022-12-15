#ifndef _DEF_CUH_
#define _DEF_CUH_

#ifndef BLOCK_DIM
#define BLOCK_DIM 64
#endif

#define CEIL_DIV(a, b) ( (a)/(b) + ((a) % (b) > 0 ? 1 : 0) )
#define GRID_DIM(size) CEIL_DIV(size, BLOCK_DIM)

typedef struct {
	size_t src;
	size_t dst;
} Edge;

#endif // _DEF_CUH_
