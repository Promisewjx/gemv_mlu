#pragma once

#ifndef MIN
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#endif // !MIN

#ifndef MAX
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif // !MIN

#ifndef PAD_UP
#define PAD_UP(x, y) ((((x) + (y) - 1) / (y)) * (y))
#endif

#ifndef PAD_DOWN
#define PAD_DOWN(x, y) (((x) / (y)) * (y))
#endif