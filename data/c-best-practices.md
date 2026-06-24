# C Programming Best Practices

## Memory Management

### Always Check malloc Return Value

```c
int *arr = malloc(n * sizeof(int));
if (arr == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return -1;
}
```

### Use free() to Prevent Memory Leaks

Every `malloc()`, `calloc()`, or `realloc()` must have a corresponding `free()`:

```c
void process_data() {
    char *buffer = malloc(BUFFER_SIZE);
    if (buffer == NULL) return;

    // ... use buffer ...

    free(buffer);  // Always free allocated memory
}
```

### Avoid Dangling Pointers

Set pointer to NULL after freeing:

```c
free(ptr);
ptr = NULL;  // Prevents use-after-free bugs
```

## Pointer Safety

### Initialize Pointers to NULL

```c
int *ptr = NULL;  // Safe default, can be checked later
```

### Check for NULL Before Dereferencing

```c
if (ptr != NULL) {
    *ptr = value;
}
```

### Use const for Read-Only Data

```c
void print_string(const char *str) {
    // str cannot be modified inside this function
    printf("%s\n", str);
}
```

## Buffer Safety

### Use snprintf Instead of sprintf

```c
// Dangerous: no bounds checking
sprintf(buffer, "%s", input);

// Safe: limits written bytes
snprintf(buffer, sizeof(buffer), "%s", input);
```

### Use strncat Instead of strcat

```c
// Dangerous: may overflow
strcat(dest, src);

// Safe: limits copied bytes
strncat(dest, src, sizeof(dest) - strlen(dest) - 1);
```

### Always NUL-Terminate Strings

```c
char buffer[256];
strncpy(buffer, src, sizeof(buffer) - 1);
buffer[sizeof(buffer) - 1] = '\0';  // Ensure NUL termination
```

## Error Handling

### Check Return Values

```c
FILE *fp = fopen("file.txt", "r");
if (fp == NULL) {
    perror("fopen failed");
    return -1;
}
```

### Use errno for Error Information

```c
#include <errno.h>
#include <string.h>

FILE *fp = fopen("file.txt", "r");
if (fp == NULL) {
    fprintf(stderr, "Error: %s\n", strerror(errno));
    return -1;
}
```

### Clean Up Resources on Error

```c
int read_config(const char *path) {
    FILE *fp = fopen(path, "r");
    if (fp == NULL) return -1;

    char *buffer = malloc(BUFFER_SIZE);
    if (buffer == NULL) {
        fclose(fp);  // Clean up previously allocated resource
        return -1;
    }

    // ... process file ...

    free(buffer);
    fclose(fp);
    return 0;
}
```

## Code Organization

### Use Header Files for Interfaces

**math_utils.h:**
```c
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

int add(int a, int b);
int multiply(int a, int b);

#endif
```

**math_utils.c:**
```c
#include "math_utils.h"

int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }
```

### Minimize Global Variables

```c
// Bad: global state makes code hard to test and reason about
int counter = 0;

// Good: pass state explicitly
void increment(int *counter) {
    (*counter)++;
}
```

### Use Meaningful Variable Names

```c
// Bad: unclear purpose
int x = 0;
int y = 0;

// Good: self-documenting
int item_count = 0;
int max_retries = 0;
```

## Performance

### Prefer Pre-Increment in Loops

```c
// Less efficient for complex types
for (int i = 0; i < n; i++)

// More efficient (especially for iterators)
for (int i = 0; i < n; ++i)
```

### Avoid Repeated Calculations in Loops

```c
// Bad: strlen called every iteration
for (size_t i = 0; i < strlen(str); i++)

// Good: calculate once
size_t len = strlen(str);
for (size_t i = 0; i < len; i++)
```

### Use Appropriate Data Types

```c
// Use size_t for sizes and array indices
size_t array_size = 100;

// Use uint8_t for small integers (saves memory)
uint8_t small_value = 255;

// Use int32_t for portable 32-bit integers
int32_t portable_int = -1;
```

## Common Pitfalls

### Integer Overflow

```c
// Check for overflow before multiplication
int a = INT_MAX / 2;
int b = 3;

if (a > INT_MAX / b) {
    // Overflow would occur
    return ERROR;
}
int result = a * b;
```

### Signed/Unsigned Comparison

```c
// Dangerous: -1 promoted to large unsigned value
int i = -1;
size_t n = 10;
if (i < n)  // May not behave as expected!

// Safe: explicit cast
if (i < (int)n)
```

### Uninitialized Variables

```c
// Bad: uninitialized, contains garbage
int x;

// Good: initialized to known value
int x = 0;
```
