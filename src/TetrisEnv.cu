#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <tuple>

#define COORD_BITS 5
#define BLOCK_BITS 10
#define COORD_MASK 31
#define ROW_MASK 1023u
#define ROLLING_STATS_WINDOW 16384ull
#define PACK(x, y) ((uint64_t)(x) | ((uint64_t)(y) << COORD_BITS))
#define SHAPE(x0, y0, x1, y1, x2, y2, x3, y3) \
    (PACK(x0, y0) | (PACK(x1, y1) << 10) | (PACK(x2, y2) << 20) | (PACK(x3, y3) << 30))

// Per-episode accumulators (all uint32, summed within an episode, rolled over a window of
// completed episodes). Index order is shared by episodeAccum, rollingData and rollingSums.
#define M_LENGTH 0
#define M_ROWS 1
#define M_PIECES 2
#define M_HOLES 3
#define M_BUMP 4
#define M_AGGH 5
#define M_MAXH 6
#define M_SINGLE 7
#define M_DOUBLE 8
#define M_TRIPLE 9
#define M_TETRIS 10
#define NUM_EPISODE_METRICS 11
#define NUM_STATS_OUT 13

__constant__ uint64_t BLOCKTYPES[7];
__constant__ float REWARD_MULTIPLIERS[3];   // [0]=survival, [1]=placedBlock, [2]=gameOver
__constant__ float LINE_CLEAR_TABLE[5];     // [k] = reward for clearing k lines at once; [0]=0

static uint32_t* gamefield = nullptr;
static uint64_t* blockProjection = nullptr;
static uint64_t* rngState = nullptr;
static uint32_t* frameCounter = nullptr;
static uint32_t* episodeAccum = nullptr;                 // envsCount * NUM_EPISODE_METRICS
static uint32_t* rollingData = nullptr;                  // NUM_EPISODE_METRICS * ROLLING_STATS_WINDOW (metric-major)
static unsigned long long* rollingCount = nullptr;       // 1: number of completed episodes
static unsigned long long* rollingSums = nullptr;        // NUM_EPISODE_METRICS: windowed sums
static uint32_t* lastTileType = nullptr;                 // per env: previous piece type (NES-style single re-roll)
static uint64_t envsCount = 0;
static int runtimeEnvsPerThread = 1;
static int runtimeDropSpeed = 1;

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

// Board-quality metrics scanned column by column. holes = empty cells below the topmost filled
// cell of a column (raycast down); heights drive bumpiness / aggregate / max height.
__device__ __forceinline__ void boardMetrics(const uint32_t* field, int* holesOut, int* bumpOut, int* aggHOut, int* maxHOut)
{
    int heights[10];
    int aggH = 0;
    int maxH = 0;
    int holes = 0;
    #pragma unroll
    for (int x = 0; x < 10; x++) {
        int top = 20;
        for (int y = 0; y < 20; y++) {
            if (getCell(field, x, y) != 0u) {
                top = y;
                break;
            }
        }
        int height = 20 - top;
        heights[x] = height;
        aggH += height;
        maxH = max(maxH, height);
        for (int y = top + 1; y < 20; y++) {
            holes += (getCell(field, x, y) == 0u) ? 1 : 0;
        }
    }
    int bump = 0;
    #pragma unroll
    for (int x = 0; x < 9; x++) {
        bump += abs(heights[x] - heights[x + 1]);
    }
    *holesOut = holes;
    *bumpOut = bump;
    *aggHOut = aggH;
    *maxHOut = maxH;
}

__device__ __forceinline__ int randomTileType(int env, uint64_t* rng)
{
    uint64_t state = rng[env];
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    rng[env] = state;
    return (int)((state >> 32) % 7);
}

__device__ __forceinline__ uint64_t generateNewTile(int env, uint64_t* rng, uint32_t* lastType)
{
    // NES-style randomizer: roll a piece; if it repeats the previous one, re-roll ONCE and keep
    // whatever comes. Immediate repeats become rare (~1/49) but not impossible, and droughts stay.
    int t = randomTileType(env, rng);
    if ((uint32_t)t == lastType[env]) {
        t = randomTileType(env, rng);
    }
    lastType[env] = (uint32_t)t;
    uint64_t rawProjection = BLOCKTYPES[t];
    uint64_t offset = 4ull;
    uint64_t parallelOffset = offset | (offset << 10) | (offset << 20) | (offset << 30);
    return rawProjection + parallelOffset;
}

__global__ void setupKernel(uint64_t* __restrict__ projection, uint64_t* __restrict__ rng, uint32_t* __restrict__ lastType, uint64_t n)
{
    uint64_t env = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (env < n) {
        projection[env] = generateNewTile((int)env, rng, lastType);
    }
}

void start(int64_t envs,
           int64_t envsPerThread,
           int64_t dropSpeed,
           double survivalReward,
           double placedBlockReward,
           double clearReward1,
           double clearReward2,
           double clearReward3,
           double clearReward4,
           double gameOverReward)
{
    if (gamefield != nullptr) {
        cudaFree(gamefield);
        cudaFree(blockProjection);
        cudaFree(rngState);
        cudaFree(frameCounter);
        cudaFree(episodeAccum);
        cudaFree(rollingData);
        cudaFree(rollingCount);
        cudaFree(rollingSums);
        cudaFree(lastTileType);
    }

    envsCount = (uint64_t)envs;
    runtimeEnvsPerThread = (int)envsPerThread;
    runtimeDropSpeed = dropSpeed > 1 ? (int)dropSpeed : 1;
    cudaMalloc(&gamefield, sizeof(uint32_t) * 8 * envsCount);
    cudaMalloc(&blockProjection, sizeof(uint64_t) * envsCount);
    cudaMalloc(&rngState, sizeof(uint64_t) * envsCount);
    cudaMalloc(&frameCounter, sizeof(uint32_t) * envsCount);
    cudaMalloc(&episodeAccum, sizeof(uint32_t) * NUM_EPISODE_METRICS * envsCount);
    cudaMalloc(&rollingData, sizeof(uint32_t) * NUM_EPISODE_METRICS * ROLLING_STATS_WINDOW);
    cudaMalloc(&rollingCount, sizeof(unsigned long long) * 1);
    cudaMalloc(&rollingSums, sizeof(unsigned long long) * NUM_EPISODE_METRICS);
    cudaMalloc(&lastTileType, sizeof(uint32_t) * envsCount);

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
    float rewardMultiplierData[3] = {
        (float)survivalReward,
        (float)placedBlockReward,
        (float)gameOverReward,
    };
    cudaMemcpyToSymbol(REWARD_MULTIPLIERS, rewardMultiplierData, sizeof(rewardMultiplierData));
    float lineClearTableData[5] = {
        0.0f,
        (float)clearReward1,
        (float)clearReward2,
        (float)clearReward3,
        (float)clearReward4,
    };
    cudaMemcpyToSymbol(LINE_CLEAR_TABLE, lineClearTableData, sizeof(lineClearTableData));

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
    cudaMemset(frameCounter, 0, sizeof(uint32_t) * envsCount);
    cudaMemset(episodeAccum, 0, sizeof(uint32_t) * NUM_EPISODE_METRICS * envsCount);
    cudaMemset(rollingData, 0, sizeof(uint32_t) * NUM_EPISODE_METRICS * ROLLING_STATS_WINDOW);
    cudaMemset(rollingCount, 0, sizeof(unsigned long long) * 1);
    cudaMemset(rollingSums, 0, sizeof(unsigned long long) * NUM_EPISODE_METRICS);
    cudaMemset(lastTileType, 0xFF, sizeof(uint32_t) * envsCount);

    const int blockSize = 256;
    int gridSize = (int)((envsCount + blockSize - 1) / blockSize);
    setupKernel<<<gridSize, blockSize>>>(blockProjection, rngState, lastTileType, envsCount);
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

__device__ __forceinline__ bool projectionHasCell(uint64_t proj, int cell)
{
    bool hit = false;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int shift = i * BLOCK_BITS;
        int x = (proj >> shift) & COORD_MASK;
        int y = (proj >> (shift + COORD_BITS)) & COORD_MASK;
        hit |= (y * 10 + x) == cell;
    }
    return hit;
}

__device__ __forceinline__ void imageObservation(const uint32_t* field, uint64_t proj, float* observation)
{
    for (int i = 0; i < 200; i++) {
        uint32_t boardCell = (field[i >> 5] >> (i & 31)) & 1u;
        observation[i] = (float)(boardCell | (uint32_t)projectionHasCell(proj, i));
    }
}

__device__ __forceinline__ void topProjectionObservation(const uint32_t* field, uint64_t proj, float* observation)
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

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int shift = i * BLOCK_BITS;
        observation[10 + i * 2] = (float)((proj >> shift) & COORD_MASK);
        observation[11 + i * 2] = (float)((proj >> (shift + COORD_BITS)) & COORD_MASK);
    }
}

__global__ void stepKernel(const uint32_t* __restrict__ actions,
                           int n,
                           uint32_t* __restrict__ fields,
                           uint64_t* __restrict__ projections,
                           uint64_t* __restrict__ rng,
                           bool imageObs,
                           int envsPerThread,
                           int dropSpeed,
                           uint32_t* __restrict__ frames,
                           uint32_t* __restrict__ episodeAccumulators,
                           uint32_t* __restrict__ rolling,
                           unsigned long long* __restrict__ rollCount,
                           unsigned long long* __restrict__ rollSums,
                           uint32_t* __restrict__ lastType,
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
    uint32_t* acc = episodeAccumulators + (uint64_t)env * NUM_EPISODE_METRICS;
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

    float reward = REWARD_MULTIPLIERS[0];
    bool isDone = false;
    acc[M_LENGTH] += 1u;
    uint32_t frame = frames[env];
    // Action 4 = hard drop: slam the piece straight to its resting position and force a lock this
    // step, so placement speed no longer depends on how high the stack is built.
    bool hardDrop = (action == 4u);
    if (hardDrop) {
        unsigned int dropCollision = 0u;
        uint64_t below = moveDown(proj, field, &dropCollision);
        while (dropCollision == 0u) {
            proj = below;
            below = moveDown(proj, field, &dropCollision);
        }
    }
    bool shouldDrop = hardDrop || (dropSpeed <= 1) || ((frame % (uint32_t)dropSpeed) == 0u);
    unsigned int collision = hardDrop ? 1u : 0u;
    uint64_t movedDown = hardDrop ? proj : (shouldDrop ? moveDown(proj, field, &collision) : proj);

    if (shouldDrop && collision) {
        int holesBefore, bumpBefore, aggHBefore, maxHBefore;
        boardMetrics(field, &holesBefore, &bumpBefore, &aggHBefore, &maxHBefore);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int shift = i * BLOCK_BITS;
            int x = (proj >> shift) & COORD_MASK;
            int y = (proj >> (shift + COORD_BITS)) & COORD_MASK;
            setCell(field, x, y, 1u);
        }
        acc[M_PIECES] += 1u;

        uint64_t newProjection = generateNewTile(env, rng, lastType);
        if (isGameOver(newProjection, field)) {
            clearGameField(field);
            reward = REWARD_MULTIPLIERS[2];
            isDone = true;

            unsigned long long idx = atomicAdd(rollCount, 1ull);
            unsigned long long slot = idx % ROLLING_STATS_WINDOW;
            bool full = idx >= ROLLING_STATS_WINDOW;
            #pragma unroll
            for (int m = 0; m < NUM_EPISODE_METRICS; m++) {
                uint32_t val = acc[m];
                uint32_t old = rolling[(uint64_t)m * ROLLING_STATS_WINDOW + slot];
                rolling[(uint64_t)m * ROLLING_STATS_WINDOW + slot] = val;
                atomicAdd(rollSums + m, (unsigned long long)val);
                if (full) {
                    atomicAdd(rollSums + m, 0ull - (unsigned long long)old);
                }
                acc[m] = 0u;
            }
        } else {
            int cleared = clearLines(field);
            reward += REWARD_MULTIPLIERS[1] + LINE_CLEAR_TABLE[cleared];
            acc[M_ROWS] += (uint32_t)cleared;
            if (cleared == 1) {
                acc[M_SINGLE] += 1u;
            } else if (cleared == 2) {
                acc[M_DOUBLE] += 1u;
            } else if (cleared == 3) {
                acc[M_TRIPLE] += 1u;
            } else if (cleared == 4) {
                acc[M_TETRIS] += 1u;
            }

            int holesAfter, bumpAfter, aggHAfter, maxHAfter;
            boardMetrics(field, &holesAfter, &bumpAfter, &aggHAfter, &maxHAfter);
            int created = holesAfter - holesBefore;
            if (created < 0) {
                created = 0;
            }
            acc[M_HOLES] += (uint32_t)created;
            acc[M_BUMP] += (uint32_t)bumpAfter;
            acc[M_AGGH] += (uint32_t)aggHAfter;
            acc[M_MAXH] += (uint32_t)maxHAfter;
        }
        proj = newProjection;
    } else {
        proj = movedDown;
        frames[env] = frame + 1u;
    }

    projections[env] = proj;
    rewards[env] = reward;
    done[env] = isDone;

    if (imageObs) {
        imageObservation(field, proj, observations + env * 200);
    } else {
        topProjectionObservation(field, proj, observations + env * 18);
    }
    }
}

__global__ void statsKernel(const unsigned long long* __restrict__ rollCount,
                            const unsigned long long* __restrict__ rollSums,
                            float* __restrict__ out)
{
    unsigned long long episodes = rollCount[0];
    if (episodes == 0ull) {
        for (int i = 0; i < NUM_STATS_OUT; i++) {
            out[i] = 0.0f;
        }
        return;
    }
    unsigned long long divisor = episodes < ROLLING_STATS_WINDOW ? episodes : ROLLING_STATS_WINDOW;
    double dEp = (double)divisor;
    double sLen = (double)rollSums[M_LENGTH];
    double sRows = (double)rollSums[M_ROWS];
    double sPieces = (double)rollSums[M_PIECES];
    double lenDiv = sLen > 0.0 ? sLen : 1.0;
    double pieceDiv = sPieces > 0.0 ? sPieces : 1.0;

    out[0] = (float)(sLen / dEp);                               // avg episode length
    out[1] = (float)(sRows / dEp);                              // avg rows cleared / episode
    out[2] = (float)(sRows / lenDiv);                           // clears per step
    out[3] = (float)((double)rollSums[M_HOLES] / pieceDiv);     // holes created / placement
    out[4] = (float)((double)rollSums[M_BUMP] / pieceDiv);      // bumpiness / placement
    out[5] = (float)((double)rollSums[M_AGGH] / pieceDiv);      // aggregate height / placement
    out[6] = (float)((double)rollSums[M_MAXH] / pieceDiv);      // max height / placement
    out[7] = (float)(sPieces / dEp);                            // pieces / episode
    out[8] = (float)(sRows / pieceDiv);                         // lines per piece
    out[9] = (float)((double)rollSums[M_SINGLE] / dEp);         // singles / episode
    out[10] = (float)((double)rollSums[M_DOUBLE] / dEp);        // doubles / episode
    out[11] = (float)((double)rollSums[M_TRIPLE] / dEp);        // triples / episode
    out[12] = (float)((double)rollSums[M_TETRIS] / dEp);        // tetrises / episode
}

torch::Tensor step(torch::Tensor actions, torch::Tensor observations, torch::Tensor rewards, torch::Tensor done, bool imageObservation)
{
    const uint32_t* actionData = actions.const_data_ptr<uint32_t>();
    float* observationData = observations.mutable_data_ptr<float>();
    float* rewardData = rewards.mutable_data_ptr<float>();
    bool* doneData = done.mutable_data_ptr<bool>();
    torch::Tensor stats = torch::empty({NUM_STATS_OUT}, actions.options().dtype(at::kFloat));
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
                                        runtimeDropSpeed,
                                        frameCounter,
                                        episodeAccum,
                                        rollingData,
                                        rollingCount,
                                        rollingSums,
                                        lastTileType,
                                        observationData,
                                        rewardData,
                                        doneData);
    statsKernel<<<1, 1>>>(rollingCount, rollingSums, statsData);
    return stats;
}

// Returns (data, valid) where data is an int32 [NUM_EPISODE_METRICS, ROLLING_STATS_WINDOW] GPU
// tensor holding the per-episode values of the last completed episodes (metric-major, same index
// order as the M_* macros), and valid is how many leading slots are filled. Consumers slice
// data[metric, :valid] and run torch.quantile on-GPU for distribution stats.
std::tuple<torch::Tensor, int64_t> rolling_episode_data()
{
    torch::Tensor data = torch::empty({(long)NUM_EPISODE_METRICS, (long)ROLLING_STATS_WINDOW},
                                      at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
    if (rollingData != nullptr) {
        cudaMemcpy(data.mutable_data_ptr<int>(),
                   rollingData,
                   sizeof(uint32_t) * NUM_EPISODE_METRICS * ROLLING_STATS_WINDOW,
                   cudaMemcpyDeviceToDevice);
    } else {
        data.zero_();
    }
    unsigned long long countHost = 0ull;
    if (rollingCount != nullptr) {
        cudaMemcpy(&countHost, rollingCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    }
    int64_t valid = (int64_t)(countHost < ROLLING_STATS_WINDOW ? countHost : ROLLING_STATS_WINDOW);
    return std::make_tuple(data, valid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}

TORCH_LIBRARY(TetrisEnvBranchless, m) {
    m.def("start(int envs, int envs_per_thread, int drop_speed, float survival_reward, float placed_block_reward, float clear_reward_1, float clear_reward_2, float clear_reward_3, float clear_reward_4, float game_over_reward) -> ()");
    m.def("step(Tensor actions, Tensor(a!) observations, Tensor(b!) rewards, Tensor(c!) done, bool image_observation) -> Tensor");
    m.def("rolling_episode_data() -> (Tensor, int)");
}

TORCH_LIBRARY_IMPL(TetrisEnvBranchless, CatchAll, m) {
    m.impl("start", &start);
    m.impl("rolling_episode_data", &rolling_episode_data);
}

TORCH_LIBRARY_IMPL(TetrisEnvBranchless, CUDA, m) {
    m.impl("step", &step);
}
