#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>

#define COORD_BITS 5
#define BLOCK_BITS 10
#define COORD_MASK 31
#define ROW_MASK 1023u
#define PACK(x, y) ((uint64_t)(x) | ((uint64_t)(y) << COORD_BITS))
#define SHAPE(x0, y0, x1, y1, x2, y2, x3, y3) \
    (PACK(x0, y0) | (PACK(x1, y1) << 10) | (PACK(x2, y2) << 20) | (PACK(x3, y3) << 30))

__constant__ uint64_t BLOCKTYPES[7];

static uint32_t* gamefield = nullptr;
static uint64_t* blockProjection = nullptr;
static uint64_t* rngState = nullptr;
static uint32_t* episodeLength = nullptr;
static uint32_t* episodeRows = nullptr;
static unsigned long long* statsCounters = nullptr;
static uint64_t envsCount = 0;
static int runtimeEnvsPerThread = 1;

__device__ __forceinline__ uint32_t getCell(const uint32_t* field, int x, int y)
{
    int bitIndex = y * 10 + x;
    return (field[bitIndex >> 5] >> (bitIndex & 31)) & 1u;
}

__device__ __forceinline__ void setCell(uint32_t* field, int x, int y, uint32_t value)
{
    int bitIndex = y * 10 + x;
    uint32_t shift = (uint32_t)(bitIndex & 31);
    uint32_t mask = 1u << shift;
    uint32_t* word = field + (bitIndex >> 5);
    *word = (*word & ~mask) | ((value & 1u) << shift);
}

__device__ __forceinline__ int randomTileType(int env, uint64_t* rng)
{
    uint64_t state = rng[env];
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    rng[env] = state;
    return (int)((state >> 32) % 7);
}

__device__ __forceinline__ uint64_t generateNewTile(int env, uint64_t* rng)
{
    uint64_t rawProjection = BLOCKTYPES[randomTileType(env, rng)];
    uint64_t offset = 3ull | (4ull << COORD_BITS);
    uint64_t parallelOffset = offset | (offset << 10) | (offset << 20) | (offset << 30);
    return rawProjection + parallelOffset;
}

__global__ void setupKernel(uint64_t* __restrict__ projection, uint64_t* __restrict__ rng, uint64_t n)
{
    uint64_t env = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (env < n) {
        projection[env] = generateNewTile((int)env, rng);
    }
}

void start(int64_t envs, int64_t envsPerThread)
{
    if (gamefield != nullptr) {
        cudaFree(gamefield);
        cudaFree(blockProjection);
        cudaFree(rngState);
        cudaFree(episodeLength);
        cudaFree(episodeRows);
        cudaFree(statsCounters);
    }

    envsCount = (uint64_t)envs;
    runtimeEnvsPerThread = (int)envsPerThread;
    cudaMalloc(&gamefield, sizeof(uint32_t) * 8 * envsCount);
    cudaMalloc(&blockProjection, sizeof(uint64_t) * envsCount);
    cudaMalloc(&rngState, sizeof(uint64_t) * envsCount);
    cudaMalloc(&episodeLength, sizeof(uint32_t) * envsCount);
    cudaMalloc(&episodeRows, sizeof(uint32_t) * envsCount);
    cudaMalloc(&statsCounters, sizeof(unsigned long long) * 3);

    uint64_t blockTypeData[7] = {
        SHAPE(1, 0, 1, 1, 1, 2, 1, 3),
        SHAPE(1, 1, 2, 1, 1, 2, 2, 2),
        SHAPE(1, 1, 1, 2, 2, 2, 2, 3),
        SHAPE(2, 1, 1, 2, 2, 2, 1, 3),
        SHAPE(1, 1, 2, 1, 2, 2, 2, 3),
        SHAPE(2, 1, 1, 2, 2, 2, 2, 3),
        SHAPE(1, 1, 2, 1, 1, 2, 1, 3),
    };
    cudaMemcpyToSymbol(BLOCKTYPES, blockTypeData, sizeof(blockTypeData));

    auto* rngCpu = (uint64_t*)malloc(sizeof(uint64_t) * envsCount);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> distrib(0, std::numeric_limits<uint64_t>::max());
    for (uint64_t i = 0; i < envsCount; i++) {
        rngCpu[i] = distrib(gen);
    }

    cudaMemcpy(rngState, rngCpu, sizeof(uint64_t) * envsCount, cudaMemcpyHostToDevice);
    free(rngCpu);
    cudaMemset(gamefield, 0, sizeof(uint32_t) * 8 * envsCount);
    cudaMemset(blockProjection, 0, sizeof(uint64_t) * envsCount);
    cudaMemset(episodeLength, 0, sizeof(uint32_t) * envsCount);
    cudaMemset(episodeRows, 0, sizeof(uint32_t) * envsCount);
    cudaMemset(statsCounters, 0, sizeof(unsigned long long) * 3);

    const int blockSize = 256;
    int gridSize = (int)((envsCount + blockSize - 1) / blockSize);
    setupKernel<<<gridSize, blockSize>>>(blockProjection, rngState, envsCount);
}

__device__ __forceinline__ uint64_t horizontalMove(uint64_t proj, int dir, const uint32_t* field)
{
    uint64_t moved = proj;
    unsigned collision = 0;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int shift = i * BLOCK_BITS;
        int x = (proj >> shift) & COORD_MASK;
        int y = (proj >> (shift + COORD_BITS)) & COORD_MASK;
        int nx = x + dir;
        unsigned invalid = ((unsigned)nx >= 10u) | ((unsigned)y >= 20u);
        int sx = max(0, min(nx, 9));
        int sy = max(0, min(y, 19));
        int bitIndex = sy * 10 + sx;

        collision |= invalid | ((field[bitIndex >> 5] >> (bitIndex & 31)) & 1u);
        moved &= ~((uint64_t)COORD_MASK << shift);
        moved |= (uint64_t)(nx & COORD_MASK) << shift;
    }

    uint64_t mask = 0ull - (uint64_t)(collision == 0);
    return (moved & mask) | (proj & ~mask);
}

__device__ __forceinline__ uint64_t rotate(uint64_t proj, const uint32_t* field)
{
    uint64_t rotated = 0;
    unsigned collision = 0;
    int pivotX = proj & COORD_MASK;
    int pivotY = (proj >> COORD_BITS) & COORD_MASK;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int shift = i * BLOCK_BITS;
        int x = (proj >> shift) & COORD_MASK;
        int y = (proj >> (shift + COORD_BITS)) & COORD_MASK;
        int relX = x - pivotX;
        int relY = y - pivotY;
        int nx = pivotX - relY;
        int ny = pivotY + relX;
        unsigned invalid = ((unsigned)nx >= 10u) | ((unsigned)ny >= 20u);
        int sx = max(0, min(nx, 9));
        int sy = max(0, min(ny, 19));
        int bitIndex = sy * 10 + sx;

        collision |= invalid | ((field[bitIndex >> 5] >> (bitIndex & 31)) & 1u);
        rotated |= ((uint64_t)(nx & COORD_MASK) << shift) |
                   ((uint64_t)(ny & COORD_MASK) << (shift + COORD_BITS));
    }

    uint64_t mask = 0ull - (uint64_t)(collision == 0);
    return (rotated & mask) | (proj & ~mask);
}

__device__ __forceinline__ uint64_t moveDown(uint64_t proj, const uint32_t* field, unsigned int* collision)
{
    uint64_t moved = proj;
    unsigned hit = 0;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int shift = i * BLOCK_BITS;
        int x = (proj >> shift) & COORD_MASK;
        int y = (proj >> (shift + COORD_BITS)) & COORD_MASK;
        int ny = y + 1;
        unsigned invalid = ((unsigned)x >= 10u) | ((unsigned)ny >= 20u);
        int sx = max(0, min(x, 9));
        int sy = max(0, min(ny, 19));
        int bitIndex = sy * 10 + sx;

        hit |= invalid | ((field[bitIndex >> 5] >> (bitIndex & 31)) & 1u);
        moved &= ~((uint64_t)COORD_MASK << (shift + COORD_BITS));
        moved |= (uint64_t)(ny & COORD_MASK) << (shift + COORD_BITS);
    }

    *collision = hit;
    return moved;
}

__device__ __forceinline__ bool isFullLine(const uint32_t* field, int y)
{
    int bitIndex = y * 10;
    int word = bitIndex >> 5;
    int shift = bitIndex & 31;
    uint32_t row = field[word] >> shift;
    if (shift > 22) {
        row |= field[word + 1] << (32 - shift);
    }
    return (row & ROW_MASK) == ROW_MASK;
}

__device__ __forceinline__ uint32_t getRow(const uint32_t* field, int y)
{
    int bitIndex = y * 10;
    int word = bitIndex >> 5;
    int shift = bitIndex & 31;
    uint32_t row = field[word] >> shift;
    if (shift > 22) {
        row |= field[word + 1] << (32 - shift);
    }
    return row & ROW_MASK;
}

__device__ __forceinline__ void setRow(uint32_t* field, int y, uint32_t row)
{
    int bitIndex = y * 10;
    int word = bitIndex >> 5;
    int shift = bitIndex & 31;
    uint32_t mask = ROW_MASK << shift;
    field[word] = (field[word] & ~mask) | ((row & ROW_MASK) << shift);
    if (shift > 22) {
        uint32_t highBits = shift + 10 - 32;
        uint32_t highMask = (1u << highBits) - 1u;
        field[word + 1] = (field[word + 1] & ~highMask) | ((row & ROW_MASK) >> (32 - shift));
    }
}

__device__ __forceinline__ int clearLines(uint32_t* field)
{
    int cleared = 0;
    int writeY = 19;

    for (int readY = 19; readY >= 0; readY--) {
        uint32_t row = getRow(field, readY);
        if (row == ROW_MASK) {
            cleared++;
            continue;
        }

        if (writeY != readY) {
            setRow(field, writeY, row);
        }
        writeY--;
    }

    for (int y = writeY; y >= 0; y--) {
        setRow(field, y, 0u);
    }

    return cleared;
}

__device__ __forceinline__ bool isGameOver(uint64_t newProjection, const uint32_t* field)
{
    bool gameOver = false;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int shift = i * BLOCK_BITS;
        int x = (newProjection >> shift) & COORD_MASK;
        int y = (newProjection >> (shift + COORD_BITS)) & COORD_MASK;
        gameOver |= getCell(field, x, y) != 0u;
    }
    return gameOver;
}

__device__ __forceinline__ void clearGameField(uint32_t* field)
{
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        field[i] = 0u;
    }
}

__device__ __forceinline__ void imageObservation(const uint32_t* field, float* observation)
{
    for (int i = 0; i < 200; i++) {
        observation[i] = (float)((field[i >> 5] >> (i & 31)) & 1u);
    }
}

__device__ __forceinline__ void topProjectionObservation(const uint32_t* field, float* observation)
{
    #pragma unroll
    for (int x = 0; x < 10; x++) {
        float firstHit = 20.0f;
        for (int y = 0; y < 20; y++) {
            if (getCell(field, x, y) != 0u) {
                firstHit = (float)y;
                break;
            }
        }
        observation[x] = firstHit;
    }
}

__global__ void stepKernel(const uint32_t* __restrict__ actions,
                           int n,
                           uint32_t* __restrict__ fields,
                           uint64_t* __restrict__ projections,
                           uint64_t* __restrict__ rng,
                           bool imageObs,
                           int envsPerThread,
                           uint32_t* __restrict__ episodeLengths,
                           uint32_t* __restrict__ episodeRowCounts,
                           unsigned long long* __restrict__ counters,
                           float* __restrict__ observations,
                           float* __restrict__ rewards,
                           bool* __restrict__ done)
{
    int firstEnv = (blockIdx.x * blockDim.x + threadIdx.x) * envsPerThread;
    for (int loopEnv = 0; loopEnv < envsPerThread; loopEnv++) {
    int env = firstEnv + loopEnv;
    if (env >= n) {
        return;
    }

    uint32_t* field = fields + 8 * env;
    uint64_t proj = projections[env];
    uint32_t action = actions[env];

    uint64_t leftMoved = horizontalMove(proj, -1, field);
    uint64_t rotated = rotate(proj, field);
    uint64_t rightMoved = horizontalMove(proj, 1, field);
    uint64_t leftMask = 0ull - (uint64_t)(action == 0);
    uint64_t rotateMask = 0ull - (uint64_t)(action == 1);
    uint64_t rightMask = 0ull - (uint64_t)(action == 2);
    uint64_t keepMask = ~(leftMask | rotateMask | rightMask);
    proj = (leftMoved & leftMask) | (rotated & rotateMask) | (rightMoved & rightMask) | (proj & keepMask);

    float reward = 0.0f;
    bool isDone = false;
    uint32_t currentEpisodeLength = episodeLengths[env] + 1u;
    uint32_t currentEpisodeRows = episodeRowCounts[env];
    unsigned int collision = 0;
    uint64_t movedDown = moveDown(proj, field, &collision);

    if (collision) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int shift = i * BLOCK_BITS;
            int x = (proj >> shift) & COORD_MASK;
            int y = (proj >> (shift + COORD_BITS)) & COORD_MASK;
            setCell(field, x, y, 1u);
        }

        uint64_t newProjection = generateNewTile(env, rng);
        if (isGameOver(newProjection, field)) {
            clearGameField(field);
            reward = -1.0f;
            isDone = true;
            atomicAdd(counters + 0, 1ull);
            atomicAdd(counters + 1, (unsigned long long)currentEpisodeLength);
            atomicAdd(counters + 2, (unsigned long long)currentEpisodeRows);
            currentEpisodeLength = 0u;
            currentEpisodeRows = 0u;
        } else {
            int cleared = clearLines(field);
            reward = (float)cleared;
            currentEpisodeRows += (uint32_t)cleared;
        }
        proj = newProjection;
    } else {
        proj = movedDown;
    }

    projections[env] = proj;
    episodeLengths[env] = currentEpisodeLength;
    episodeRowCounts[env] = currentEpisodeRows;
    rewards[env] = reward;
    done[env] = isDone;

    if (imageObs) {
        imageObservation(field, observations + env * 200);
    } else {
        topProjectionObservation(field, observations + env * 10);
    }
    }
}

__global__ void statsKernel(const unsigned long long* __restrict__ counters, float* __restrict__ out)
{
    unsigned long long episodes = counters[0];
    if (episodes == 0ull) {
        out[0] = 0.0f;
        out[1] = 0.0f;
        return;
    }
    out[0] = (float)((double)counters[1] / (double)episodes);
    out[1] = (float)((double)counters[2] / (double)episodes);
}

torch::Tensor step(torch::Tensor actions, torch::Tensor observations, torch::Tensor rewards, torch::Tensor done, bool imageObservation)
{
    const uint32_t* actionData = actions.const_data_ptr<uint32_t>();
    float* observationData = observations.mutable_data_ptr<float>();
    float* rewardData = rewards.mutable_data_ptr<float>();
    bool* doneData = done.mutable_data_ptr<bool>();
    torch::Tensor stats = torch::empty({2}, actions.options().dtype(at::kFloat));
    float* statsData = stats.mutable_data_ptr<float>();

    const int blockSize = 256;
    int gridSize = (int)((envsCount + blockSize * runtimeEnvsPerThread - 1) / (blockSize * runtimeEnvsPerThread));
    stepKernel<<<gridSize, blockSize>>>(actionData,
                                        (int)envsCount,
                                        gamefield,
                                        blockProjection,
                                        rngState,
                                        imageObservation,
                                        runtimeEnvsPerThread,
                                        episodeLength,
                                        episodeRows,
                                        statsCounters,
                                        observationData,
                                        rewardData,
                                        doneData);
    statsKernel<<<1, 1>>>(statsCounters, statsData);
    return stats;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}

TORCH_LIBRARY(TetrisEnvBranchless, m) {
    m.def("start(int envs, int envs_per_thread) -> ()");
    m.def("step(Tensor actions, Tensor(a!) observations, Tensor(b!) rewards, Tensor(c!) done, bool image_observation) -> Tensor");
}

TORCH_LIBRARY_IMPL(TetrisEnvBranchless, CatchAll, m) {
    m.impl("start", &start);
}

TORCH_LIBRARY_IMPL(TetrisEnvBranchless, CUDA, m) {
    m.impl("step", &step);
}


