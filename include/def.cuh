#ifndef _DEF_CUH_
#define _DEF_CUH_

#ifndef BLOCK_DIM
#define BLOCK_DIM 64
#endif

#define GRID_DIM(size) ( (size) / BLOCK_DIM + ((size) % BLOCK_DIM > 0 ? 1 : 0) )

typedef struct {
	size_t src;
	size_t dst;
} Edge;

#endif // _DEF_CUH_