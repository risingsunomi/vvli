# `src/tensor.zig`

`tensor.zig` is VVLI's small CPU tensor layer. It owns contiguous `f32`
storage, keeps shapes inline, and provides the scalar/SIMD kernels needed by
the current language and vision inference paths.

The implementation is intentionally narrow:

- storage is always `f32`
- tensors are dense and row-major
- allocated tensor data is 64-byte aligned
- shape rank is capped at `max_rank`
- kernels operate on contiguous slices whenever possible
- SIMD is expressed through Zig vectors and lowered by the compiler for the
  target CPU

## Constants And Types

### `Allocator`

Alias for `std.mem.Allocator`.

Tensor allocation and deallocation are explicit. Functions that create a new
tensor take an allocator, and callers must later pass the same allocator to
`Tensor.deinit`.

### `max_rank`

Maximum tensor rank, currently `8`.

`Shape` stores dimensions inline in a fixed `[max_rank]usize` buffer. This
avoids heap allocation for shape metadata.

### `data_alignment_bytes` / `data_alignment`

Tensor data is allocated with 64-byte alignment:

```zig
const data = try allocator.alignedAlloc(f32, data_alignment, total);
```

The alignment is useful for cache-line friendly contiguous reads and gives the
compiler/runtime a better chance of using aligned vector memory operations.

### `simd_lanes`

The vector width used by SIMD kernels:

```zig
pub const simd_lanes: usize = std.simd.suggestVectorLength(f32) orelse 4;
```

On targets where Zig knows the preferred vector width, that value is used. The
fallback is four `f32` lanes.

### `Vec`

Private vector type:

```zig
const Vec = @Vector(simd_lanes, f32);
```

SIMD kernels load `Vec` values from contiguous slices, compute on whole
vectors, then handle any trailing elements with scalar loops.

### `AlignedData`

Private data slice type:

```zig
const AlignedData = []align(data_alignment_bytes) f32;
```

`Tensor.data` uses this type so the alignment guarantee stays visible in the
type system.

### `Error`

Tensor-specific errors:

- `InvalidShape`: shape rank, dimension size, or derived output size is invalid.
- `ShapeMismatch`: two tensors have incompatible shapes.
- `LengthMismatch`: a flat slice length does not match tensor element count.
- `OutOfBounds`: an index is outside a tensor dimension.

## `Shape`

`Shape` stores tensor dimensions inline:

```zig
pub const Shape = struct {
    rank_value: usize = 0,
    dims_buf: [max_rank]usize = [_]usize{0} ** max_rank,
};
```

Only `dims_buf[0..rank_value]` is active.

### `Shape.init(shape_dims)`

Validates and constructs a shape.

Checks:

- rank must be at least `1`
- rank must not exceed `max_rank`
- every dimension must be non-zero

Implementation detail: dimensions are copied into the inline buffer. No heap
allocation is needed.

### `Shape.rank()`

Returns `rank_value`.

### `Shape.asSlice()`

Returns the active dimensions as `[]const usize`.

This is the common comparison and allocation interface used by tensor methods.

### `Shape.dim(index)`

Returns one dimension by index.

Fails with `OutOfBounds` if `index >= rank`.

### `Shape.last()`

Returns the final dimension.

This is used by last-dimension operations such as softmax and RMSNorm.

### `Shape.elementCount()`

Returns the checked product of all active dimensions.

Implementation detail: delegates to `checkedProduct`, which rejects zero
dimensions and integer overflow.

### `Shape.eql(other)`

Returns true when both shapes have identical active dimensions.

## `Tensor`

`Tensor` is a dense row-major `f32` tensor:

```zig
pub const Tensor = struct {
    data: AlignedData,
    shape: Shape,
};
```

The flat index order is standard row-major order. For example, a 2D tensor with
shape `{ rows, cols }` stores row `r` at:

```zig
data[r * cols ..][0..cols]
```

### Allocation And Lifetime

#### `Tensor.init(allocator, shape_dims)`

Allocates a tensor and fills it with `0.0`.

Implementation:

1. Calls `initUndefined`.
2. Calls `fill(0.0)`.

#### `Tensor.initFilled(allocator, shape_dims, value)`

Allocates a tensor and fills every element with `value`.

#### `Tensor.initUndefined(allocator, shape_dims)`

Allocates aligned storage without initializing the elements.

Implementation:

1. Builds a validated `Shape`.
2. Computes element count with overflow checks.
3. Allocates `f32` data with 64-byte alignment.

Use this when the caller will immediately overwrite all elements.

#### `Tensor.fromSlice(allocator, shape_dims, values)`

Allocates a tensor and copies `values` into it.

The flat `values.len` must exactly match the element count implied by
`shape_dims`.

#### `Tensor.deinit(allocator)`

Frees tensor data and resets the tensor to an empty state.

Implementation detail: after freeing, the tensor is assigned:

```zig
.{ .data = emptyData(), .shape = .{} }
```

`emptyData` returns a zero-length aligned slice, allowing deinitialized tensors
to remain structurally valid without owning memory.

#### `Tensor.clone(allocator)`

Creates a new tensor with the same shape and copied data.

Implementation: delegates to `fromSlice`.

### Metadata And Views

#### `Tensor.dims()`

Returns active shape dimensions.

#### `Tensor.rank()`

Returns shape rank.

#### `Tensor.len()`

Returns the number of `f32` elements in `data`.

#### `Tensor.byteLen()`

Returns total data size in bytes:

```zig
self.data.len * @sizeOf(f32)
```

#### `Tensor.asSlice()`

Returns mutable flat data as `[]f32`.

#### `Tensor.constSlice()`

Returns immutable flat data as `[]const f32`.

#### `Tensor.row(index)`

Returns mutable row `index` from a rank-2 tensor.

Shape requirement: `{ rows, cols }`.

The returned slice is contiguous and has length `cols`.

#### `Tensor.constRow(index)`

Immutable version of `row`.

### Mutation And Copying

#### `Tensor.fill(value)`

Sets every element to `value` using `@memset`.

#### `Tensor.reshape(shape_dims)`

Changes the shape metadata without moving data.

The new shape must have the same element count as the current tensor.

#### `Tensor.copyFrom(source)`

Copies data from another tensor with the exact same shape.

#### `Tensor.copyFromSlice(values)`

Copies a flat slice into the tensor.

`values.len` must match `self.data.len`.

### Elementwise Math

Each allocating elementwise function creates an output tensor, then delegates to
the corresponding `Into` function.

#### `Tensor.add(allocator, a, b)`

Returns `a + b`.

Requires identical shapes.

#### `Tensor.addInto(out, a, b)`

Writes `a + b` into `out`.

Implementation:

1. Validates `a`, `b`, and `out` have the same shape.
2. Processes `simd_lanes` elements per loop with `Vec`.
3. Processes remaining tail elements scalar.

#### `Tensor.sub(allocator, a, b)`

Returns `a - b`.

#### `Tensor.subInto(out, a, b)`

Writes `a - b` into `out` using the same SIMD-plus-tail structure as
`addInto`.

#### `Tensor.mul(allocator, a, b)`

Returns elementwise `a * b`.

#### `Tensor.mulInto(out, a, b)`

Writes elementwise `a * b` into `out` using vector loads/stores and scalar tail
handling.

#### `Tensor.mulScalarInPlace(value)`

Multiplies every element by a scalar.

Implementation detail: splats the scalar into a `Vec`, processes vector chunks,
then handles the tail scalar.

#### `Tensor.addScaledInPlace(source, scale)`

Computes:

```zig
self = self + source * scale
```

Requires `source` to have the same shape.

Implementation detail: vector chunks use `dst + src * scale_v`; scalar tail
uses `@mulAdd` to express fused multiply-add where the target supports it.

### Activations

#### `Tensor.reluInPlace()`

Applies ReLU:

```text
max(x, 0)
```

The main loop uses vector `@max`; the tail loop uses scalar `@max`.

#### `Tensor.geluInPlace()`

Applies tanh-approximate GELU:

```text
0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

Current implementation is scalar because it uses `std.math.tanh` per element.

#### `Tensor.siluInPlace()`

Applies SiLU:

```text
x / (1 + exp(-x))
```

Current implementation is scalar because it uses `std.math.exp` per element.

### Transpose

#### `Tensor.transpose2d(allocator, input)`

Allocates and returns a transposed copy of a rank-2 tensor.

Input shape `{ rows, cols }` becomes `{ cols, rows }`.

#### `Tensor.transpose2dInto(out, input)`

Writes the 2D transpose into `out`.

Implementation:

```zig
out.data[c * rows + r] = input.data[r * cols + c];
```

This keeps both tensors dense and row-major.

### Matrix Multiplication

#### `Tensor.matmul(allocator, a, b)`

Returns matrix product `a * b`.

Requirements:

- `a` rank is `2`
- `b` rank is `2`
- `a.shape[1] == b.shape[0]`

For `a` shape `{ m, k }` and `b` shape `{ k, n }`, output shape is `{ m, n }`.

Implementation detail:

1. Allocates output `{ m, n }`.
2. Calls `matmulInto` with `scratch_b_transposed = null`.

The default path now avoids allocating a full transposed copy of `b`. It uses
the no-scratch vectorized row/panel kernel and writes each output value once.

#### `Tensor.matmulInto(out, a, b, scratch_b_transposed)`

Writes matrix product into an existing output tensor.

If `scratch_b_transposed` is provided:

1. Validates scratch shape `{ b_cols, b_rows }`.
2. Transposes `b` into scratch.
3. Calls `matmulWithTransposedB`.

If scratch is `null`, the implementation calls `matmulBlocked`, which avoids
allocating or requiring a transposed buffer. Despite the historical name, this
path is currently a vectorized row/panel kernel rather than the older scalar
blocked implementation.

### Softmax

#### `Tensor.softmaxLastDim(allocator, input)`

Returns a softmax copy of `input`, normalized independently across the last
dimension.

Implementation:

1. Clones `input`.
2. Calls `softmaxLastDimInPlace` on the clone.

#### `Tensor.softmaxLastDimInPlace()`

Normalizes each row implied by the last dimension.

For a tensor whose last dimension is `width`, the flat data is treated as:

```zig
rows = data.len / width
```

Each `width`-sized row is passed to `softmaxSlice`.

Implementation detail: `softmaxSlice` subtracts the row max before exponentials
for numerical stability.

### RMSNorm

#### `Tensor.rmsNormLastDim(allocator, input, weight, epsilon)`

Returns RMSNorm over the last dimension.

`weight` must be rank 1 and have length equal to the input's last dimension.

#### `Tensor.rmsNormLastDimInto(out, input, weight, epsilon)`

Writes RMSNorm into `out`.

For each row:

1. Compute mean square:

   ```text
   mean_square = sum(x_i^2) / width
   ```

2. Compute scale:

   ```text
   scale = 1 / sqrt(mean_square + epsilon)
   ```

3. Write:

   ```text
   out_i = input_i * weight_i * scale
   ```

Implementation detail: `sumSquares` and `mulWeightScaled` use vector loops and
scalar tails.

### NCHW Convolution

#### `Conv2DOptions`

Controls 2D convolution:

```zig
pub const Conv2DOptions = struct {
    stride_h: usize = 1,
    stride_w: usize = 1,
    pad_h: usize = 0,
    pad_w: usize = 0,
    dilation_h: usize = 1,
    dilation_w: usize = 1,
};
```

Padding is symmetric. Dilation and stride must be non-zero.

#### `Tensor.conv2dNchw(allocator, input, kernel, bias, options)`

Allocates and returns a 2D convolution result.

Shape requirements:

- `input`: `{ batch, in_channels, in_height, in_width }`
- `kernel`: `{ out_channels, in_channels, kernel_height, kernel_width }`
- optional `bias`: `{ out_channels }`

Output shape:

```text
{
  batch,
  out_channels,
  floor((in_height + 2 * pad_h - effective_kernel_h) / stride_h) + 1,
  floor((in_width  + 2 * pad_w - effective_kernel_w) / stride_w) + 1,
}
```

Where:

```text
effective_kernel = dilation * (kernel - 1) + 1
```

#### `Tensor.conv2dNchwInto(out, input, kernel, bias, options)`

Writes convolution into `out`.

Implementation:

1. Validates and computes expected output dimensions with `conv2dOutputDims`.
2. Iterates batch, output channel, output y, output x.
3. Starts each output accumulator with `bias[oc]` when bias exists.
4. Iterates input channels and kernel coordinates.
5. Uses `paddedIndex` to map output/kernel coordinates back to input indexes.
6. Skips coordinates that land in padding.
7. Uses `dotUnchecked` for contiguous horizontal kernel runs when
   `dilation_w == 1` and the full kernel row is inside the input.
8. Falls back to scalar `@mulAdd` for padded or dilated horizontal positions.

The optimized inner row path keeps common valid convolutions reading contiguous
input and kernel slices.

## Free Public Functions

### `dot(a, b)`

Computes the dot product of two flat `f32` slices.

Requires equal lengths.

Implementation: validates length, then calls `dotAssumeEqual`.

### `dotAssumeEqual(a, b)`

Computes the dot product without returning a length error.

This function keeps a debug assertion that the slice lengths match, then calls
the optimized unchecked dot kernel. It is useful for hot paths that have already
validated shapes, including the native VLM attention wrapper.

## Private Helpers

These are private to `tensor.zig` but are central to how the public API works.

### `checkedProduct(dims)`

Computes the product of dimensions with overflow checks.

Returns `InvalidShape` for zero dimensions or multiplication overflow.

### `emptyData()`

Returns a zero-length 64-byte-aligned `f32` slice.

This is used after `Tensor.deinit` so the tensor no longer owns memory but still
has a well-typed aligned data slice.

### `expectRank(tensor, rank)`

Validates exact tensor rank.

### `expectShape(tensor, dims)`

Validates exact shape dimensions.

### `expectSameShape(a, b)`

Validates that two tensors have identical dimensions.

### `loadVec(slice, index)`

Loads `simd_lanes` contiguous `f32` values from `slice[index..]` into a `Vec`.

Callers only use this after checking that a full vector-sized chunk is
available.

### `storeVec(slice, index, vec)`

Stores a `Vec` back into a flat slice.

Implementation detail: writes each lane explicitly in an `inline for` loop.

### `dotUnchecked(a, b)`

Computes a dot product without checking lengths.

Implementation:

1. Uses four independent SIMD accumulators over unrolled vector chunks.
2. Combines the accumulators to reduce dependency-chain stalls.
3. Processes remaining full vector chunks.
4. Reduces the vector accumulator with `@reduce(.Add, acc)`.
5. Handles the scalar tail with `@mulAdd`.

This is used in matrix multiplication and the convolution fast path.

### `dot2Unchecked(a, b0, b1)`

Computes two dot products that share the same left-hand slice.

This is used by the transposed matmul path for remaining column pairs. It loads
the `a` vector once and applies it to two transposed `b` rows.

### `dot4Unchecked(a, b0, b1, b2, b3)`

Computes four dot products that share the same left-hand slice.

This is the main transposed matmul helper. It reduces repeated `a` loads when
computing adjacent output columns.

### `sumSquares(values)`

Computes:

```text
sum(values_i * values_i)
```

Used by RMSNorm. It uses the same vector-then-tail pattern as `dotUnchecked`.

### `mulWeightScaled(out, input, weight, scale)`

Writes:

```text
out_i = input_i * weight_i * scale
```

Used by RMSNorm. The main path is vectorized; the tail path is scalar.

### `softmaxSlice(values)`

Applies in-place softmax to one flat row.

Implementation:

1. Find max value.
2. Replace each element with `exp(value - max_value)`.
3. Sum exponentials.
4. Multiply every element by `1 / sum`.

Subtracting the max improves numerical stability.

### `matmulWithTransposedB(out, a, b_transposed)`

Matrix multiplication where `b` has already been transposed.

For each output row, it computes groups of four output columns with
`dot4Unchecked`, then groups of two with `dot2Unchecked`, then any final column
with `dotUnchecked`.

Conceptually each output cell is:

```zig
out[row, col] = dot(a_row, b_transposed_row)
```

The transposed layout keeps both dot-product inputs contiguous.

### `matmulBlocked(out, a, b)`

No-scratch matrix multiplication that does not require transposed `b` storage.

Implementation:

1. Iterates each output row.
2. Processes output columns in panels of `simd_lanes * 4`.
3. Uses four SIMD accumulators per panel.
4. Broadcasts one `a[row, p]` value with `@splat`.
5. Loads contiguous vectors from `b[p, col..]`.
6. Accumulates the full `k` dimension in registers.
7. Stores each output vector once after accumulation.
8. Handles remaining vector-width columns, then scalar tail columns.

This removes the older inner-loop read-modify-write pattern on `out.data` and
keeps the common no-scratch path explicitly vectorized.

### `conv2dOutputDims(input, kernel, bias, options)`

Validates convolution inputs and computes output dimensions.

Checks:

- input and kernel are rank 4
- stride and dilation are non-zero
- input channels match kernel input channels
- optional bias has shape `{ out_channels }`
- derived height and width are valid

### `convOutputDim(input, kernel, pad, stride, dilation)`

Computes one output spatial dimension for convolution.

Rejects invalid kernel/stride/dilation values, arithmetic overflow, and cases
where the padded input is smaller than the effective kernel.

### `paddedIndex(out_index, stride, kernel_index, dilation, pad)`

Maps one output coordinate plus kernel coordinate back into input space:

```text
out_index * stride + kernel_index * dilation - pad
```

Returns `isize` so callers can detect negative positions that land in padding.

## Current Test Coverage

The file includes unit tests for:

- inline shape storage and 64-byte aligned contiguous data allocation
- 2D matmul over contiguous rows
- agreement between no-scratch matmul and transposed-scratch matmul across
  vector-width-sized dimensions
- last-dimension softmax
- last-dimension RMSNorm
- valid NCHW convolution

The tests exercise the main public behavior, but they do not yet cover every
error branch, scalar SIMD tails, padded/dilated convolution variants, or
allocation failure paths.
