#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <climits>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <sys/resource.h>     // For getrusage() to measure peak CPU RAM usage
#include <filesystem>         // C++17 for std::filesystem::path, create_directories

// Uncomment for debug prints
// #define DEBUG_PRINT

// For GPU error-checking convenience
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while(0)

// -----------------------------------------------------------------------------
// Global variables to track peak GPU VRAM usage
// -----------------------------------------------------------------------------
static size_t g_totalGPUMem   = 0;      
static size_t g_minFreeMem    = SIZE_MAX; 
static bool   g_memInitialized = false;

void updateVRAMUsage() {
    cudaDeviceSynchronize();
    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);

    if (!g_memInitialized) {
        g_totalGPUMem   = totalMem;
        g_memInitialized = true;
    }
    if (freeMem < g_minFreeMem) {
        g_minFreeMem = freeMem;
    }
}

template <typename T>
inline void myCudaMalloc(T** ptr, size_t size) {
    CUDA_CHECK(cudaMalloc((void**)ptr, size));
    updateVRAMUsage();
}

inline void myCudaFree(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
    updateVRAMUsage();
}

// -----------------------------------------------------------------------------
// Graph Structure (adjacency list in CSR-like form)
// -----------------------------------------------------------------------------
struct Graph {
    int numVertices;
    int numEdges;
    std::vector<int> adjacencyList; 
    std::vector<int> edgesOffset;  
    std::vector<int> edgesSize;    
};

// -----------------------------------------------------------------------------
// Load the graph from an edge list file: each line "src dst"
// -----------------------------------------------------------------------------
void loadGraph(const std::string &filename, Graph &G)
{
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cerr << "Could not open file: " << filename << "\n";
        exit(EXIT_FAILURE);
    }

    std::vector<std::pair<int,int>> edges;
    int maxNodeId = -1;
    {
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            int s, t;
            ss >> s >> t;
            edges.push_back({s, t});
            maxNodeId = std::max(maxNodeId, std::max(s, t));
        }
    }
    in.close();

    G.numVertices = maxNodeId + 1;
    G.numEdges    = static_cast<int>(edges.size());

    G.edgesOffset.resize(G.numVertices, 0);
    G.edgesSize.resize(G.numVertices,   0);

    // Count outdegree for each vertex
    for (auto &e : edges) {
        int s = e.first;
        G.edgesSize[s]++;
    }

    // Compute prefix sums for edgesOffset
    for (int i = 1; i < G.numVertices; i++) {
        G.edgesOffset[i] = G.edgesOffset[i-1] + G.edgesSize[i-1];
    }

    // Make a copy of offset as a "fill pointer"
    std::vector<int> fillPtr = G.edgesOffset;
    G.adjacencyList.resize(G.numEdges);

    // Fill adjacencyList
    for (auto &e : edges) {
        int s = e.first;
        int t = e.second;
        int pos = fillPtr[s]++;
        G.adjacencyList[pos] = t;
    }
}

// -----------------------------------------------------------------------------
// CUDA kernel: expand the front layer in parallel
// -----------------------------------------------------------------------------
__global__
void kernel_expand_front(int nFront,
                         const int *frontNodes,
                         const int *adjList,
                         const int *offsets,
                         const int *sizes,
                         int *outDegrees,
                         int *neighbors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nFront) return;

    int node = frontNodes[i];
    int start = offsets[node];
    int sz    = sizes[node];

    outDegrees[i] = sz;

    // Copy adjacency into the output neighbors array
    for (int j = 0; j < sz; j++) {
        neighbors[i + j * nFront] = adjList[start + j];
    }
}

// -----------------------------------------------------------------------------
// A custom compressed "visited" bitset
// -----------------------------------------------------------------------------
struct Bitset {
    std::vector<uint64_t> blocks;  // each block is 64 bits

    Bitset() = default;
    explicit Bitset(int nVertices) {
        size_t nBlocks = (nVertices + 63) / 64;
        blocks.resize(nBlocks, 0ull);
    }
    
    // Check if vertex v is set
    inline bool test(int v) const {
        int blockIdx = v / 64;
        int bitPos   = v % 64;
        return (blocks[blockIdx] & (1ULL << bitPos)) != 0ULL;
    }
    
    // Set vertex v
    inline void set(int v) {
        int blockIdx = v / 64;
        int bitPos   = v % 64;
        blocks[blockIdx] |= (1ULL << bitPos);
    }
};

// -----------------------------------------------------------------------------
// A "PathItem" that stores the partial path and a compressed visited bitset
// -----------------------------------------------------------------------------
struct PathItem {
    // The partial path of nodes
    std::vector<int> path;  
    // Which nodes are visited in path
    Bitset visited;         

    PathItem() = default;
    PathItem(int nVertices) : visited(nVertices) {}
};

// -----------------------------------------------------------------------------
// Write a PathItem to disk in a binary format
// -----------------------------------------------------------------------------
void writePathItem(std::ofstream &out, const PathItem &item) {
    // 1) Write path size + path data
    size_t pathLen = item.path.size();
    out.write(reinterpret_cast<const char*>(&pathLen), sizeof(pathLen));
    out.write(reinterpret_cast<const char*>(item.path.data()),
              pathLen * sizeof(int));
    // 2) Write visited bitset blocks
    size_t nBlocks = item.visited.blocks.size();
    out.write(reinterpret_cast<const char*>(&nBlocks), sizeof(nBlocks));
    out.write(reinterpret_cast<const char*>(item.visited.blocks.data()),
              nBlocks * sizeof(uint64_t));
}

// -----------------------------------------------------------------------------
// Read a PathItem from disk
// -----------------------------------------------------------------------------
void readPathItem(std::ifstream &in, PathItem &item) {
    // 1) Read path length + path
    size_t pathLen;
    in.read(reinterpret_cast<char*>(&pathLen), sizeof(pathLen));
    item.path.resize(pathLen);
    in.read(reinterpret_cast<char*>(item.path.data()), pathLen * sizeof(int));

    // 2) Read visited bitset
    size_t nBlocks = 0;
    in.read(reinterpret_cast<char*>(&nBlocks), sizeof(nBlocks));
    item.visited.blocks.resize(nBlocks, 0ULL);
    in.read(reinterpret_cast<char*>(item.visited.blocks.data()),
            nBlocks * sizeof(uint64_t));
}

// -----------------------------------------------------------------------------
// Flush all items to disk (overwrites the file)
// -----------------------------------------------------------------------------
void flushToDisk(const std::vector<PathItem> &items, const std::string &filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Cannot open cache file " << filename << " for writing.\n";
        return;
    }

    // Write how many items
    size_t count = items.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // Write each PathItem
    for (const auto &it : items) {
        writePathItem(out, it);
    }
    out.close();
}

// -----------------------------------------------------------------------------
// Read all PathItem from disk
// -----------------------------------------------------------------------------
void loadFromDisk(std::vector<PathItem> &items, const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Cannot open cache file " << filename << " for reading.\n";
        return;
    }

    // Read how many
    size_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));

    items.clear();
    items.reserve(count);
    for (size_t i = 0; i < count; i++) {
        PathItem temp;
        readPathItem(in, temp);
        items.push_back(std::move(temp));
    }
    in.close();
}

// -----------------------------------------------------------------------------
// BFS-like enumeration of all simple paths from start->end
// with disk caching each BFS wave
// -----------------------------------------------------------------------------
void findAllPathsBFS_GPU(const Graph &G, int start, int end, 
                         std::vector<std::vector<int>> &allPaths,
                         const std::string &outDir)
{
    // We'll store expansions for each BFS wave in "wave_cache.bin" under outDir
    std::filesystem::path cacheFilePath = 
        std::filesystem::path(outDir) / "wave_cache.bin";

    // Prepare initial wave with a single path
    std::vector<PathItem> queue;
    {
        PathItem item(G.numVertices);
        item.path.push_back(start);
        item.visited.set(start);
        queue.push_back(std::move(item));
    }

    // GPU adjacency
    int *d_adjList   = nullptr;
    int *d_offsets   = nullptr;
    int *d_sizes     = nullptr;

    myCudaMalloc(&d_adjList, G.adjacencyList.size() * sizeof(int));
    myCudaMalloc(&d_offsets, G.numVertices         * sizeof(int));
    myCudaMalloc(&d_sizes,   G.numVertices         * sizeof(int));

    CUDA_CHECK(cudaMemcpy(d_adjList, G.adjacencyList.data(),
                          G.adjacencyList.size()*sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, G.edgesOffset.data(),
                          G.numVertices*sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sizes,   G.edgesSize.data(),
                          G.numVertices*sizeof(int),
                          cudaMemcpyHostToDevice));

    // BFS expansions in "waves"
    // Each iteration expands the 'queue' into 'nextQueue', then flushes nextQueue to disk.
    // Then we read it back from disk to become the next wave's queue.

    int waveIdx = 0;
    while (!queue.empty()) {
        waveIdx++;

        // 1) Gather last nodes
        int nFront = static_cast<int>(queue.size());
        std::vector<int> frontNodes(nFront);
        for (int i = 0; i < nFront; i++) {
            frontNodes[i] = queue[i].path.back();
        }

        // 2) Copy frontNodes to GPU
        int *d_frontNodes = nullptr;
        myCudaMalloc(&d_frontNodes, nFront * sizeof(int));
        CUDA_CHECK(cudaMemcpy(d_frontNodes, frontNodes.data(),
                              nFront*sizeof(int), cudaMemcpyHostToDevice));

        // 3) outDegrees array
        int *d_outDegrees = nullptr;
        myCudaMalloc(&d_outDegrees, nFront * sizeof(int));

        // 4) find maxOutDegree among these front nodes
        int maxOutDegree = 0;
        for (int fn : frontNodes) {
            maxOutDegree = std::max(maxOutDegree, G.edgesSize[fn]);
        }

        // 5) neighbors
        int *d_neighbors = nullptr;
        if (maxOutDegree > 0) {
            myCudaMalloc(&d_neighbors, nFront * maxOutDegree * sizeof(int));
        }

        // 6) GPU kernel
        {
            int blockSize = 128;
            int gridSize  = (nFront + blockSize - 1) / blockSize;
            kernel_expand_front<<<gridSize, blockSize>>>(
                nFront, d_frontNodes, d_adjList, d_offsets, d_sizes,
                d_outDegrees, d_neighbors
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // 7) Copy back, build nextWave
        std::vector<int> outDegrees(nFront);
        std::vector<PathItem> nextWave; 
        // We'll store expansions from this wave in nextWave, then flush it to disk

        if (maxOutDegree > 0) {
            std::vector<int> neighborsCPU(nFront * maxOutDegree);

            CUDA_CHECK(cudaMemcpy(outDegrees.data(), d_outDegrees,
                                  nFront*sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(neighborsCPU.data(), d_neighbors,
                                  nFront*maxOutDegree*sizeof(int), 
                                  cudaMemcpyDeviceToHost));

            nextWave.reserve(nFront * 2); 
            for (int i = 0; i < nFront; i++) {
                const PathItem &pitem = queue[i];
                int deg = outDegrees[i];
                for (int j = 0; j < deg; j++) {
                    int nbr = neighborsCPU[i + j*nFront];
                    if (pitem.visited.test(nbr)) {
                        // skip cycles
                        continue;
                    }
                    // build new path
                    PathItem newItem = pitem; // copy
                    newItem.path.push_back(nbr);
                    newItem.visited.set(nbr);

                    if (nbr == end) {
                        allPaths.push_back(newItem.path);
                    } else {
                        nextWave.push_back(std::move(newItem));
                    }
                }
            }
        }

        // Cleanup for this wave
        myCudaFree(d_frontNodes);
        myCudaFree(d_outDegrees);
        if (maxOutDegree > 0) {
            myCudaFree(d_neighbors);
        }

        // 8) Flush nextWave to disk, clear it from RAM
        flushToDisk(nextWave, cacheFilePath.string());
        nextWave.clear();

        // 9) Now we've fully expanded the current queue -> nextWave on disk.
        //    Clear the old queue from RAM:
        queue.clear();

        // 10) Read from disk to load the expansions as the next BFS wave
        loadFromDisk(queue, cacheFilePath.string());

        // optionally remove the cache file (it'll be overwritten each wave)
        // std::filesystem::remove(cacheFilePath);
        // But it's fine to leave it for debug
    }

    // Free adjacency
    myCudaFree(d_adjList);
    myCudaFree(d_offsets);
    myCudaFree(d_sizes);
}

// -----------------------------------------------------------------------------
// Save all final paths to JSON
// -----------------------------------------------------------------------------
void saveAllPaths(const std::vector<std::vector<int>> &allPaths,
                  const std::string &outFilename)
{
    std::ofstream out(outFilename);
    if (!out.is_open()) {
        std::cerr << "Cannot open output file: " << outFilename << "\n";
        return;
    }

    out << "[\n";
    for (size_t i = 0; i < allPaths.size(); i++) {
        out << "  [";
        for (size_t j = 0; j < allPaths[i].size(); j++) {
            out << allPaths[i][j];
            if (j + 1 < allPaths[i].size()) {
                out << ", ";
            }
        }
        out << "]";
        if (i + 1 < allPaths.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "]\n";
    out.close();
    std::cout << "Saved " << allPaths.size() 
              << " paths to " << outFilename << "\n";
}

// -----------------------------------------------------------------------------
// Main
// Usage: ./bfs_paths <edges.txt> <start> <end> <out_directory>
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    auto tStart = std::chrono::high_resolution_clock::now();

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <edges.txt> <start> <end> <out_directory>\n";
        return 1;
    }

    std::string filename   = argv[1];
    int         start      = std::stoi(argv[2]);
    int         end        = std::stoi(argv[3]);
    std::string outDirArg  = argv[4];

    // Ensure the output directory exists
    std::filesystem::path outDir(outDirArg);
    try {
        std::filesystem::create_directories(outDir);
    } catch (const std::exception &ex) {
        std::cerr << "Error creating output directory: " << ex.what() << "\n";
        return 1;
    }

    // Load graph
    Graph G;
    loadGraph(filename, G);

    std::cout << "Graph loaded with " << G.numVertices 
              << " vertices and " << G.numEdges << " edges.\n";

    if (start < 0 || start >= G.numVertices ||
        end   < 0 || end   >= G.numVertices)
    {
        std::cerr << "Start/end node out of range [0.."
                  << (G.numVertices - 1) << "]\n";
        return 2;
    }

    // Enumerate all paths, using BFS + disk-caching
    std::vector<std::vector<int>> allPaths;
    findAllPathsBFS_GPU(G, start, end, allPaths, outDirArg);

    // Construct output filenames
    std::string baseFilename = std::filesystem::path(filename).stem().string();

    std::ostringstream outFile;
    outFile << "bfs_paths_" << start << "_" << end << "_" << baseFilename << ".json";
    std::filesystem::path outJsonPath = outDir / outFile.str();

    // Save final paths
    saveAllPaths(allPaths, outJsonPath.string());
    std::cout << "Total paths found: " << allPaths.size() << "\n";

    // Timing and resource usage
    auto tEnd = std::chrono::high_resolution_clock::now();
    double elapsedSec = std::chrono::duration<double>(tEnd - tStart).count();

    // Peak CPU RAM (KB, on Linux/Unix)
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long peakRamKB = usage.ru_maxrss;

    // Peak GPU usage
    size_t peakGpuBytes = 0;
    if (g_memInitialized) {
        peakGpuBytes = g_totalGPUMem - g_minFreeMem;
    }

    // Write stats
    std::ostringstream statsFilename;
    statsFilename << "bfs_stats_" << start << "_" << end << "_" << baseFilename << ".csv";
    std::filesystem::path outCsvPath = outDir / statsFilename.str();

    std::ofstream statsOut(outCsvPath);
    if (!statsOut.is_open()) {
        std::cerr << "Cannot open " << outCsvPath.string() << " for writing.\n";
    } else {
        statsOut << "Execution Time (seconds),Peak CPU RAM (MB),Peak GPU VRAM (MB)\n";
        double peakRamMB = static_cast<double>(peakRamKB) / 1024.0;
        double peakGpuMB = static_cast<double>(peakGpuBytes) / (1024.0 * 1024.0);
        statsOut << elapsedSec << "," 
                 << peakRamMB << "," 
                 << peakGpuMB << "\n";
        statsOut.close();
        std::cout << "Saved execution stats to " << outCsvPath.string() << "\n";
    }

    return 0;
}
