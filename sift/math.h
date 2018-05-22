#pragma once

#define MATH_PI 3.14159265358979323846264338327950288

#define MATH_SQRT2 1.41421356237309504880168872420969808   // ¸ùºÅ2


#define MATH_POW2(x) ((x) * (x))


/* Floating-point epsilon comparisons. */
#define MATH_EPSILON_EQ(x,v,eps) (((v - eps) <= x) && (x <= (v + eps)))