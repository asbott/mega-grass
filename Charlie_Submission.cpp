/*
	
				Charlie's Interview Submission
				
	Features:
		- Height-map terrain rendering
		- Dense grass rendering
		
	Major techniques used:
		- Instanced drawing of grass meshes and terrain. Completely procedural.
		- Mesh LOD's
		- Tile-based grass density LOD's
		- Distance-based widening of grass models, to improve aliasing in far-away grass
		- Wind simulation based of perlin-noise
		
	Note:
	
		Submitting draw commands to a command buffer is 600x slower in Debug, and we do a lot
		off them, so I recommend you to run in Release. See "@DebugPerformance" for more discussion.
		
		Due to a short deadline on an ambitious goal, I have heavily prioritized result over
		code quality. I have however tried to make sure it's well-commented and divided into
		clear sections.
		
		If your colors look wrong, see "@SwapchainFormat".
*/


#include "The-Forge/Common_3/Application/Interfaces/IApp.h"
#include "The-Forge/Common_3/Graphics/Interfaces/IGraphics.h"
#include "The-Forge/Common_3/Application/Interfaces/IFont.h"
#include "The-Forge/Common_3/Application/Interfaces/IProfiler.h"
#include "The-Forge/Common_3/Application/Interfaces/IUI.h"

#include "The-Forge/Common_3/Utilities/RingBuffer.h"
#include "The-Forge/Common_3/Utilities/Math/Random.h"

#include "The-Forge/Common_3/Application/Interfaces/ICameraController.h"

#include "terrain_config.h"

#define TAU (PI*2)

///
// Structures
//
typedef struct LodLevelInfo {
	float mThreshold;
	uint32_t mIndexCount;
	
	float pad1;
	float pad2;
} LodLevelInfo;
typedef struct LodSettings {
	LodLevelInfo mLevels[NUMBER_OF_GRASS_LOD] = {
		{ 0.0f, 0, 0, 0 },
    	{ 60.0f, 0, 0, 0 },
    	{ 450.0f, 0, 0, 0 },
    	{ 750.0f, 0, 0, 0 },
	};
	float mDensityFadeStartPercent = 0.3f;
	float mMinDensityPercent = 0.4f;
	float mLowestDetailDistance = 935.0f;
} LodSettings;

///
// Structures reflected in shaders
//
typedef struct GrassVertex {
	float3 mPosition;
	uint32_t mNormal;
} GrassVertex;

typedef struct SceneUniformData {
	CameraMatrix mCameraToClip;
	Vector3 mSunDirection;
	Vector2 mTerrainSize;
	float mDaylightFactor = 1.0;
	float mMaxFloorY = 80;
	
	float mTime;
	float mWindStrength = 0.9f;
	float mMaxWindLeanAngle = TAU*0.25f;
	float mWindSpeed = 80.0;
	float mMaxNaturalAngle = TAU*0.1;
	float mMinGrassWidth = 0.2f;
	float mMaxGrassWidth = 0.8f;
	float mMinGrassHeight = 3.4f;
	float mMaxGrassHeight = 18.5f;
	uint32_t mMaxInstancesPerTile;
	float mSampleGranularity = 8;
	float pad3;
	Vector3 mViewDir;
	Vector3 mCameraPos;
	Vector3 mGrassBaseColor = Vector3(0.05f, 0.3f, 0.01f);
	Vector3 mGrassTipColor = Vector3(0.3f, 0.5f, 0.1f);
	Vector3 mWindDir = Vector3(1, 0, 0.2f);
} SceneUniformData;

typedef struct TileEntry {
	uint32_t mXTile;
	uint32_t mYTile;
	uint32_t mTileSeed;
	uint32_t pad;
} TileEntry;
typedef struct GrassTileData {
	TileEntry mTiles[GRASS_TILE_COUNT];
} GrassTileData;

typedef struct GrassDrawArgument {
	uint32_t mIndexCount;
    uint32_t mInstanceCount;
    uint32_t mStartIndex;
    uint32_t mVertexOffset;
    uint32_t mStartInstance;
} GrassDrawArgument;

typedef struct GrassDrawUniformData {
	float3 mViewPosition;
	uint32_t mPerceivedNumberOfGrass = 10000000;
	LodSettings mLod;
	
	// Frustum planes
	Vector4 rcp;
	Vector4 lcp;
	Vector4 tcp;
	Vector4 bcp;
	Vector4 fcp;
	Vector4 ncp;
	Vector4 fhp; // forward horizontal plane
	Vector4 fvp; // forward vertical plane
} GrassDrawUniformData;

typedef struct SkyboxUniformData {
	Matrix4 mView;
	CameraMatrix mProjection;
} SkyboxUniformData;

const uint32_t gNumberOfFrames = 3;

///
// Base graphics resources
SwapChain         *pSwapChain              = NULL;
Renderer          *pRenderer               = NULL;
Queue             *pGraphicsQueue          = NULL;
GpuCmdRing        gGraphicsCmdRing         = {};
Semaphore         *pImageAcquiredSemaphore = NULL;
RootSignature     *pRootSignature          = NULL;

//
// Terrain resources 
Shader             *pTerrainShader                = NULL;
Pipeline           *pTerrainPipeline              = NULL;
DescriptorSet      *pDescriptorSetTerrainUbo      = { NULL };

///
// Grass resources
Shader           *pGrassShader                    = NULL;
Pipeline         *pGrassPipeline                  = NULL;
Buffer           *pGrassTileBuffer                = NULL; // Readonly, we only need one
Buffer           *pGrassInstanceVbo               = NULL;
// We need to get a single integer index to each draw call of each grass tile.
// I REALLY wanted to use vulkan push constant/d3d12 root signature constant here.
// I couldn't figure out a way to do so with the The Forge, so now I'm making 100's
// of tiny ubo's instead...
DescriptorSet    *pDescriptorSetGrass             = { NULL };
GrassTileData    gGrassTileData                   = {};
VertexLayout     gGrassVertexLayoutForLoading     = {}; // We need per instance layout, but model loader will be unhappy about that.
VertexLayout     gGrassVertexLayoutForDrawing     = {};
Geometry         *pGrassGeoms[NUMBER_OF_GRASS_LOD]     = { NULL }; // 4 LOD levels
GeometryData     *pGrassGeomDatas[NUMBER_OF_GRASS_LOD] = { NULL };
Buffer           *pGrassVbo = NULL;
Buffer           *pGrassIbo = NULL;
Buffer           *pGrassDrawBuffer                = NULL;
Shader           *pGrassDrawShader                = NULL;
RootSignature    *pGrassDrawRootSignature         = NULL;
Pipeline         *pGrassDrawComputePipeline       = NULL;
DescriptorSet    *pDescriptorSetGrassDrawCompute  = NULL;
Buffer           *pGrassDrawUbos[gNumberOfFrames] = {};
GrassDrawUniformData gGrassDrawUniformData        = {};


///
// Skybox resources 
Shader             *pSkyboxShader                = NULL;
Pipeline           *pSkyboxPipeline              = NULL;
Buffer             *pSkyboxUbos[gNumberOfFrames] = {};
SkyboxUniformData  gSkyboxUniformData            = {};
DescriptorSet      *pDescriptorSetSkyboxUbos     = { NULL };
DescriptorSet      *pDescriptorSetSkyboxTextures = { NULL };
// 0: back -> 1: left -> 2: front -> 3: right -> 4: bottom -> 5: top
Texture            *pSkyboxTextures[6]           = { NULL }; 

///
// Shared resources
DescriptorSet     *pDescriptorSetHeightMap  = { NULL };
DescriptorSet     *pDescriptorSetHeightMapDrawCompute  = { NULL };
Texture           *pHeightMap               = NULL;
Sampler           *pSampler                 = NULL;
ICameraController *pCameraController        = NULL;
RenderTarget      *pDepthBuffer             = NULL;
UIComponent       *pGuiWindow              = NULL;
Buffer            *pSceneUbos[gNumberOfFrames] = {};
SceneUniformData  gSceneUniformData;

uint32_t          gFrameIndex;
uint32_t          gFontID = 0;
ProfileToken      gGpuProfileToken = PROFILE_INVALID_TOKEN;
ProfileToken      pGrassUpdateToken = PROFILE_INVALID_TOKEN;
ProfileToken      gQueueSubmitToken = PROFILE_INVALID_TOKEN;

///
// Utility
//

///
// Temporary storage
//
// Very simple and low-cost "garbage collection" to avoid small temporary malloc()'s
// and free()'s when I don't really care what happens to the memory after I'm done,
// with no real overhead.

void *pTemporaryStorage = NULL;
void *pTemporaryStorageNext = NULL;
const size_t pTemporaryStorageSize = 1024*500; // 500kib

void initTemporaryStorage() {
	pTemporaryStorage = tf_malloc(pTemporaryStorageSize);
	pTemporaryStorageNext = pTemporaryStorage;
}
void exitTemporaryStorage() {
	tf_free(pTemporaryStorage);
	pTemporaryStorage = NULL;
	pTemporaryStorageNext = NULL;
}
// Call this at the start of each frame
void resetTemporaryStorage() {
	pTemporaryStorageNext = pTemporaryStorage;
}
void *tempAlloc(size_t size) {
	void *p = pTemporaryStorageNext;
	
	size = (size+15) & ~15; // Align to 16
	
	pTemporaryStorageNext = (uint8_t*)pTemporaryStorageNext+size;
	
	// In a real world scenario I would probably make this allocate contiguous pages 
	// in virtual memory space, and never free them (within a reasonable limit).
	// For now I will just assert.
	ASSERT(pTemporaryStorageNext <= (uint8_t*)pTemporaryStorage+pTemporaryStorageSize);
	
	return p;
}
// Useful string functions to make string stuff less painful
// without needing to care about the memory
char* tempCopyString(const char *pStr) {
	size_t len = strlen(pStr);
	
	char *newStr = (char*)tempAlloc(len);
	
	memcpy(newStr, pStr, len+1);
	
	return newStr;
}
char* tempPrint(const char *fmt, ...) {
    va_list args1;
    va_list args2;
    va_start(args1, fmt);
    
    // First, figure out how large the buffer needs to be
    int bufferSize = vsnprintf(NULL, 0, fmt, args1) + 1;
    
    va_end(args1);
    
    if (bufferSize <= 0) {
        return NULL;
    }
    
    va_start(args2, fmt);
    
    // Then, temp allocate the buffer and printf to it
    char *buffer = (char*)tempAlloc(bufferSize);
    if (buffer == NULL) {
        va_end(args2);
        return NULL;
    }
    vsnprintf(buffer, bufferSize, fmt, args2);
    
    va_end(args2);
    
    return buffer;
}

void addUiWidgets();

class Charlie_Submission: public IApp
{
public:
    bool Init()
    {
    	initTemporaryStorage();
    
    	///
    	// Init Renderer
    	RendererDesc rendDesc {};
    	initGPUConfiguration(rendDesc.pExtendedSettings);
    #ifdef _DEBUG
    	rendDesc.mEnableGpuBasedValidation = true;
	#endif
    	initRenderer(GetName(), &rendDesc, &pRenderer);
    	if (!pRenderer) 
    	{
    		LOGF(LogLevel::eERROR, "Failed to initialize renderer.");
    		return false;
    	}
    	setupGPUConfigurationPlatformParameters(pRenderer, rendDesc.pExtendedSettings);
    	
    	///
    	// Get the Graphics queue
    	QueueDesc queueDesc = {};
        queueDesc.mType = QUEUE_TYPE_GRAPHICS;
        queueDesc.mFlag = QUEUE_FLAG_INIT_MICROPROFILE;
        initQueue(pRenderer, &queueDesc, &pGraphicsQueue);
    
    	///
    	// Make a command ring buffer
		GpuCmdRingDesc cmdRingDesc = {};
        cmdRingDesc.pQueue = pGraphicsQueue;
        cmdRingDesc.mPoolCount = gNumberOfFrames;
        cmdRingDesc.mCmdPerPoolCount = 1;
        cmdRingDesc.mAddSyncPrimitives = true;
        initGpuCmdRing(pRenderer, &cmdRingDesc, &gGraphicsCmdRing);
		
		
		initSemaphore(pRenderer, &pImageAcquiredSemaphore);
		
		initResourceLoaderInterface(pRenderer);
		
		AddCustomInputBindings();
		
		// Init Camera Controller
		CameraMotionParameters cmp{ 80.0f, 300.0f, 100.0f };
        Vector3 camPos{ TERRAIN_WIDTH/2, 48.0f, TERRAIN_HEIGHT/2 };
        Vector3 lookAt{ Vector3(0) };
		pCameraController = initFpsCameraController(camPos, lookAt);
        pCameraController->setMotionParameters(cmp);
        
        // Load fonts
        FontDesc font = {};
        font.pFontPath = "TitilliumText/TitilliumText-Bold.otf";
        fntDefineFonts(&font, 1, &gFontID);

		// Init font rendering
        FontSystemDesc fontRenderDesc = {};
        fontRenderDesc.pRenderer = pRenderer;
        if (!initFontSystem(&fontRenderDesc))
            return false;

        // Initialize Forge User Interface Rendering
        UserInterfaceDesc uiRenderDesc = {};
        uiRenderDesc.pRenderer = pRenderer;
        initUserInterface(&uiRenderDesc);

        // Initialize micro profiler and its UI.
        ProfilerDesc profiler = {};
        profiler.pRenderer = pRenderer;
        initProfiler(&profiler);
        
        // Init profile tokens
        gGpuProfileToken = initGpuProfiler(pRenderer, pGraphicsQueue, "Graphics");
        pGrassUpdateToken = getCpuProfileToken("CPU", "Grass update", 0xff00ffff);
        gQueueSubmitToken = getCpuProfileToken("CPU", "Queue submit", 0xff00ffff);

        return true;
    }
	
    void Exit()
    {
        waitQueueIdle(pGraphicsQueue);
        
        exitUserInterface();

        exitFontSystem();

        exitProfiler();
        
        exitCameraController(pCameraController);
        
        exitResourceLoaderInterface(pRenderer);
        
        exitSemaphore(pRenderer, pImageAcquiredSemaphore);
        
        exitGpuCmdRing(pRenderer, &gGraphicsCmdRing);
        
        exitQueue(pRenderer, pGraphicsQueue);

        exitRenderer(pRenderer);
        pRenderer = NULL;
        
        exitGPUConfiguration();
        
        exitTemporaryStorage();
    }

    bool Load(ReloadDesc *pReloadDesc)
    {
		
		///
		// Load UI
		loadProfilerUI(mSettings.mWidth, mSettings.mHeight);
		
		UIComponentDesc guiDesc = {};
        guiDesc.mStartPosition = vec2(mSettings.mWidth * 0.01f, mSettings.mHeight * 0.2f);
		uiAddComponent(GetName(), &guiDesc, &pGuiWindow);
		uiSetComponentFlags(pGuiWindow, GUI_COMPONENT_FLAGS_NONE);
		
		addUiWidgets();

		///
		// Init swapchain
		
        SwapChainDesc swapChainDesc = {};
		swapChainDesc.mWindowHandle = pWindow->handle;
		swapChainDesc.mPresentQueueCount = 1;
		swapChainDesc.ppPresentQueues = &pGraphicsQueue;
		swapChainDesc.mWidth = mSettings.mWidth;
		swapChainDesc.mHeight = mSettings.mHeight;
		swapChainDesc.mImageCount = getRecommendedSwapchainImageCount(pRenderer, &pWindow->handle);
		// @SwapchainFormat
		// "getSupportedSwapchainFormat" makes colors look incorrect on my monitor. I wasn't able
		// to figure this out, So I manually set it to B8G8R8A8_UNORM. 
		// If your colors are off, you can try changing this.
		//swapChainDesc.mColorFormat = getSupportedSwapchainFormat(pRenderer, &swapChainDesc, COLOR_SPACE_SDR_SRGB);
		swapChainDesc.mColorFormat = TinyImageFormat_B8G8R8A8_UNORM;
        swapChainDesc.mColorSpace = COLOR_SPACE_SDR_SRGB;
		Vector4 clearColor = {}; // Corn-flower blue
		swapChainDesc.mColorClearValue.r = clearColor.getX();
		swapChainDesc.mColorClearValue.g = clearColor.getY();
		swapChainDesc.mColorClearValue.b = clearColor.getZ();
		swapChainDesc.mColorClearValue.a = clearColor.getW();
		swapChainDesc.mEnableVsync = mSettings.mVSyncEnabled;
		swapChainDesc.mFlags = SWAP_CHAIN_CREATION_FLAG_ENABLE_FOVEATED_RENDERING_VR;
		addSwapChain(pRenderer, &swapChainDesc, &pSwapChain);
		
		if (pSwapChain == NULL) 
		{
    		LOGF(LogLevel::eERROR, "Failed to add swapchain.");
    		return false;
    	}
    	
    	///
	    // Init depth buffer
	    
    	RenderTargetDesc depthRT = {};
        depthRT.mArraySize = 1;
        depthRT.mClearValue.depth = 0.0f;
        depthRT.mClearValue.stencil = 0;
        depthRT.mDepth = 1;
        depthRT.mFormat = TinyImageFormat_D32_SFLOAT;
        depthRT.mStartState = RESOURCE_STATE_DEPTH_WRITE;
        depthRT.mHeight = mSettings.mHeight;
        depthRT.mSampleCount = SAMPLE_COUNT_1;
        depthRT.mSampleQuality = 0;
        depthRT.mWidth = mSettings.mWidth;
        depthRT.mFlags = TEXTURE_CREATION_FLAG_ON_TILE | TEXTURE_CREATION_FLAG_VR_MULTIVIEW;
        addRenderTarget(pRenderer, &depthRT, &pDepthBuffer);
		
		if (pDepthBuffer == NULL) 
		{
    		LOGF(LogLevel::eERROR, "Failed to add depth buffer render target.");
    		return false;
    	}

		///
	    // Init shaders

    	ShaderLoadDesc shaderDesc = {};
        shaderDesc.mVert.pFileName = "terrain.vert";
        shaderDesc.mFrag.pFileName = "terrain.frag";
        addShader(pRenderer, &shaderDesc, &pTerrainShader);
		if (!pTerrainShader) 
		{
    		LOGF(LogLevel::eERROR, "Failed to add shader.");
    		return false;
    	}
    	
        shaderDesc.mVert.pFileName = "grass.vert";
        shaderDesc.mFrag.pFileName = "grass.frag";
        addShader(pRenderer, &shaderDesc, &pGrassShader);
		if (!pGrassShader) 
		{
    		LOGF(LogLevel::eERROR, "Failed to add shader.");
    		return false;
    	}
    	
    	shaderDesc.mVert.pFileName = "skybox.vert";
        shaderDesc.mFrag.pFileName = "skybox.frag";
        addShader(pRenderer, &shaderDesc, &pSkyboxShader);
		if (!pSkyboxShader) 
		{
    		LOGF(LogLevel::eERROR, "Failed to add shader.");
    		return false;
    	}
    	shaderDesc = {};
    	shaderDesc.mComp.pFileName = "grass_draw.comp";
        addShader(pRenderer, &shaderDesc, &pGrassDrawShader);
		if (!pGrassDrawShader) 
		{
    		LOGF(LogLevel::eERROR, "Failed to add shader.");
    		return false;
    	}
    	
    	///
    	// Init root signature
    	
    	// (This should probably be divided into multiple root signatures)
    	
    	Shader *shaders[3];
        shaders[0] = pTerrainShader;
        shaders[1] = pGrassShader;
        shaders[2] = pSkyboxShader;
        RootSignatureDesc rootDesc = {};
        rootDesc.mShaderCount = sizeof(shaders)/sizeof(Shader*);
        rootDesc.ppShaders = shaders;
        addRootSignature(pRenderer, &rootDesc, &pRootSignature);
        
        rootDesc = {};
        rootDesc.mShaderCount = 1;
        rootDesc.ppShaders = &pGrassDrawShader;
        addRootSignature(pRenderer, &rootDesc, &pGrassDrawRootSignature);
        
        if (!pRootSignature) 
		{
    		LOGF(LogLevel::eERROR, "Failed to add root signature.");
    		return false;
    	}
        if (!pGrassDrawRootSignature) 
		{
    		LOGF(LogLevel::eERROR, "Failed to add grass draw root signature.");
    		return false;
    	}
    	
    	///
	    // Init buffers
	    
	    uint32_t *grassInstanceData = NULL;
    	
    	{ // Shared
    		BufferLoadDesc uboDesc = {};
		    uboDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		    uboDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
		    uboDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
		    uboDesc.mDesc.mSize = sizeof(SceneUniformData);
		    uboDesc.pData = NULL;
		    uboDesc.mDesc.pName = "SceneUniformData";
		    
		    for (uint32_t i = 0; i < gNumberOfFrames; i += 1) {
			    uboDesc.ppBuffer = &pSceneUbos[i];
			    addResource(&uboDesc, nullptr);
		    }
		
		    gSceneUniformData.mTerrainSize = Vector2(TERRAIN_WIDTH, TERRAIN_HEIGHT);
		    gSceneUniformData.mSunDirection = Vector3(-1.f, -0.6f,  0.2f);
    	}
    	
	    { // Grass
		    
		    for (uint32_t i = 0; i < GRASS_TILE_COUNT; i += 1) {
		    
		    	uint32_t xTile = i % GRASS_TILE_COUNT_X;
		    	uint32_t yTile = i / GRASS_TILE_COUNT_X;
		    
		    	gGrassTileData.mTiles[i].mTileSeed = rand();
		    	gGrassTileData.mTiles[i].mXTile = xTile;
		    	gGrassTileData.mTiles[i].mYTile = yTile;
		    }
            
	    	BufferLoadDesc tileDataDesc = {};
		    tileDataDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_BUFFER;
		    tileDataDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
		    tileDataDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_NONE;
		    tileDataDesc.mDesc.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
		    tileDataDesc.pData = &gGrassTileData;
		    tileDataDesc.mDesc.pName = "GrassTileData";
		    tileDataDesc.ppBuffer = &pGrassTileBuffer;
		    tileDataDesc.mDesc.mStructStride = sizeof(TileEntry);
		    tileDataDesc.mDesc.mElementCount = GRASS_TILE_COUNT;
		    tileDataDesc.mDesc.mSize = tileDataDesc.mDesc.mStructStride*tileDataDesc.mDesc.mElementCount;
		    addResource(&tileDataDesc, nullptr);
		    
		    grassInstanceData = (uint32_t*)tf_malloc(sizeof(uint32_t)*MAX_GRASS_CAP);
		    for (uint32_t i = 0; i < MAX_GRASS_CAP; i += 1) {
		    	grassInstanceData[i] = i;
		    }
		    
		    BufferLoadDesc vboDesc = {};
		    vboDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
		    vboDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
		    vboDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_NONE;
		    vboDesc.mDesc.mSize = sizeof(uint32_t)*MAX_GRASS_CAP;
		    vboDesc.pData = grassInstanceData;
		    vboDesc.mDesc.pName = "GrassInstanceData";
		    vboDesc.ppBuffer = &pGrassInstanceVbo;
		    addResource(&vboDesc, nullptr);
		    
		    BufferLoadDesc indirectDesc = {};
		    indirectDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_BUFFER | DESCRIPTOR_TYPE_RW_BUFFER | DESCRIPTOR_TYPE_INDIRECT_BUFFER;
		    indirectDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
		    indirectDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_NONE;
		    indirectDesc.mDesc.mStartState = RESOURCE_STATE_SHADER_RESOURCE;
            indirectDesc.mDesc.mSize = sizeof(GrassDrawArgument)*GRASS_TILE_COUNT;
		    indirectDesc.mDesc.pName = "GrassDrawBuffer";
		    indirectDesc.pData = NULL;
		    indirectDesc.ppBuffer = &pGrassDrawBuffer;
		    indirectDesc.mDesc.mElementCount = GRASS_TILE_COUNT;
    		indirectDesc.mDesc.mStructStride = sizeof(GrassDrawArgument);
		    
		    addResource(&indirectDesc, nullptr);
		    
		    BufferLoadDesc uboDesc = {};
		    uboDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		    uboDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
		    uboDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
		    uboDesc.mDesc.mSize = sizeof(GrassDrawUniformData);
		    uboDesc.pData = NULL;
		    uboDesc.mDesc.pName = "GrassDrawUniformData";
		    
		    for (uint32_t i = 0; i < gNumberOfFrames; i += 1) {
			    uboDesc.ppBuffer = &pGrassDrawUbos[i];
			    addResource(&uboDesc, nullptr);
		    }
		    
	    }
	    { // Skybox
	    	BufferLoadDesc uboDesc = {};
		    uboDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		    uboDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_CPU_TO_GPU;
		    uboDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_PERSISTENT_MAP_BIT;
		    uboDesc.mDesc.mSize = sizeof(SkyboxUniformData);
		    uboDesc.pData = NULL;
		    uboDesc.mDesc.pName = "SkyboxUniformData";
		    
		    for (uint32_t i = 0; i < gNumberOfFrames; i += 1) {
			    uboDesc.ppBuffer = &pSkyboxUbos[i];
			    addResource(&uboDesc, nullptr);
		    }
	    }
	    
	    ///
	    // Init pipelines
	    
		DepthStateDesc depthStateDesc = {};
        depthStateDesc.mDepthTest = true;
        depthStateDesc.mDepthWrite = true;
        depthStateDesc.mDepthFunc = CMP_GEQUAL;
        
		{ // Terrain
	        
	    	RasterizerStateDesc basicRasterizerStateDesc = {};
	        basicRasterizerStateDesc.mCullMode = CULL_MODE_FRONT;
	        
	        PipelineDesc pipelineDesc = {};
	        pipelineDesc.mType = PIPELINE_TYPE_GRAPHICS;
	        GraphicsPipelineDesc& pipelineSettings = pipelineDesc.mGraphicsDesc;
	        pipelineSettings.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
	        pipelineSettings.mRenderTargetCount = 1;
	        pipelineSettings.pColorFormats = &pSwapChain->ppRenderTargets[0]->mFormat;
	        pipelineSettings.mSampleCount = pSwapChain->ppRenderTargets[0]->mSampleCount;
	        pipelineSettings.mSampleQuality = pSwapChain->ppRenderTargets[0]->mSampleQuality;
	        pipelineSettings.pRootSignature = pRootSignature;
	        pipelineSettings.pShaderProgram = pTerrainShader;
	        pipelineSettings.pRasterizerState = &basicRasterizerStateDesc;
	        pipelineSettings.pDepthState = &depthStateDesc;
	        pipelineSettings.mDepthStencilFormat = pDepthBuffer->mFormat;
	        addPipeline(pRenderer, &pipelineDesc, &pTerrainPipeline);
	        
			if (!pTerrainPipeline) 
			{
	    		LOGF(LogLevel::eERROR, "Failed to add terrain pipeline.");
	    		return false;
	    	}
		}
		{ // Grass
	        
	        RasterizerStateDesc basicRasterizerStateDesc = {};
	        basicRasterizerStateDesc.mCullMode = CULL_MODE_NONE; // Grass straws are planes, and we want to be able to see both sides.
	        
	        // Vertex layout
	        
	        gGrassVertexLayoutForLoading.mBindingCount = 1;
	        gGrassVertexLayoutForLoading.mAttribCount = 2;
	        
	        gGrassVertexLayoutForLoading.mBindings[0].mStride = sizeof(GrassVertex);
	        gGrassVertexLayoutForLoading.mBindings[0].mRate = VERTEX_BINDING_RATE_VERTEX;
	        
	        gGrassVertexLayoutForLoading.mAttribs[0].mSemantic = SEMANTIC_POSITION;
	        gGrassVertexLayoutForLoading.mAttribs[0].mFormat = TinyImageFormat_R32G32B32_SFLOAT;
	        gGrassVertexLayoutForLoading.mAttribs[0].mBinding = 0;
	        gGrassVertexLayoutForLoading.mAttribs[0].mLocation = 0;
	        gGrassVertexLayoutForLoading.mAttribs[0].mOffset = offsetof(GrassVertex, mPosition);;
	        
	        gGrassVertexLayoutForLoading.mAttribs[1].mSemantic = SEMANTIC_NORMAL;
	        gGrassVertexLayoutForLoading.mAttribs[1].mFormat = TinyImageFormat_R32_UINT;
	        gGrassVertexLayoutForLoading.mAttribs[1].mBinding = 0;
	        gGrassVertexLayoutForLoading.mAttribs[1].mLocation = 1;
	        gGrassVertexLayoutForLoading.mAttribs[1].mOffset = offsetof(GrassVertex, mNormal);
	        
	        gGrassVertexLayoutForDrawing = gGrassVertexLayoutForLoading;
	        
	        gGrassVertexLayoutForDrawing.mBindingCount = 2;
	        gGrassVertexLayoutForDrawing.mAttribCount = 3;
	        
	        gGrassVertexLayoutForDrawing.mBindings[1].mStride = sizeof(uint32_t);
        	gGrassVertexLayoutForDrawing.mBindings[1].mRate = VERTEX_BINDING_RATE_INSTANCE;
	        
	        gGrassVertexLayoutForDrawing.mAttribs[2].mSemantic = SEMANTIC_CUSTOM;
            strcpy(gGrassVertexLayoutForDrawing.mAttribs[2].mSemanticName, "INSTANCEID");
	        gGrassVertexLayoutForDrawing.mAttribs[2].mSemanticNameLength = (uint32_t)strlen("INSTANCEID");
	        gGrassVertexLayoutForDrawing.mAttribs[2].mFormat = TinyImageFormat_R32_UINT;
	        gGrassVertexLayoutForDrawing.mAttribs[2].mBinding = 1;
	        gGrassVertexLayoutForDrawing.mAttribs[2].mLocation = 0;
	        gGrassVertexLayoutForDrawing.mAttribs[2].mOffset = 0;
	        
	        
	        PipelineDesc pipelineDesc = {};
	        pipelineDesc.mType = PIPELINE_TYPE_GRAPHICS;
	        GraphicsPipelineDesc& pipelineSettings = pipelineDesc.mGraphicsDesc;
	        pipelineSettings.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
	        pipelineSettings.mRenderTargetCount = 1;
	        pipelineSettings.pColorFormats = &pSwapChain->ppRenderTargets[0]->mFormat;
	        pipelineSettings.mSampleCount = pSwapChain->ppRenderTargets[0]->mSampleCount;
	        //pipelineSettings.mSampleCount = SAMPLE_COUNT_8;
	        pipelineSettings.mSampleQuality = pSwapChain->ppRenderTargets[0]->mSampleQuality;
	        pipelineSettings.pRootSignature = pRootSignature;
	        pipelineSettings.pShaderProgram = pGrassShader;
	        pipelineSettings.pRasterizerState = &basicRasterizerStateDesc;
	        pipelineSettings.pDepthState = &depthStateDesc;
	        pipelineSettings.mDepthStencilFormat = pDepthBuffer->mFormat;
	        pipelineSettings.pVertexLayout = &gGrassVertexLayoutForDrawing;
	        addPipeline(pRenderer, &pipelineDesc, &pGrassPipeline);
	        
			if (!pGrassPipeline) 
			{
	    		LOGF(LogLevel::eERROR, "Failed to add grass pipeline.");
	    		return false;
	    	}
	    	
	    	// Grass draw compute pipeline
	    	pipelineDesc = {};
		    pipelineDesc.mType = PIPELINE_TYPE_COMPUTE;
		    pipelineDesc.mComputeDesc.pRootSignature = pGrassDrawRootSignature;
		    pipelineDesc.mComputeDesc.pShaderProgram = pGrassDrawShader;
		    addPipeline(pRenderer, &pipelineDesc, &pGrassDrawComputePipeline);
		}
		{ // Skybox
	        
	        depthStateDesc.mDepthTest = false;
            depthStateDesc.mDepthWrite = false; // Everything should draw over the skybox
	        
	    	RasterizerStateDesc basicRasterizerStateDesc = {};
	        basicRasterizerStateDesc.mCullMode = CULL_MODE_BACK;
	        
	        PipelineDesc pipelineDesc = {};
	        pipelineDesc.mType = PIPELINE_TYPE_GRAPHICS;
	        GraphicsPipelineDesc& pipelineSettings = pipelineDesc.mGraphicsDesc;
	        pipelineSettings.mPrimitiveTopo = PRIMITIVE_TOPO_TRI_LIST;
	        pipelineSettings.mRenderTargetCount = 1;
	        pipelineSettings.pColorFormats = &pSwapChain->ppRenderTargets[0]->mFormat;
	        pipelineSettings.mSampleCount = pSwapChain->ppRenderTargets[0]->mSampleCount;
	        pipelineSettings.mSampleQuality = pSwapChain->ppRenderTargets[0]->mSampleQuality;
	        pipelineSettings.pRootSignature = pRootSignature;
	        pipelineSettings.pShaderProgram = pSkyboxShader;
	        pipelineSettings.pRasterizerState = &basicRasterizerStateDesc;
	        pipelineSettings.pDepthState = &depthStateDesc;
	        pipelineSettings.mDepthStencilFormat = pDepthBuffer->mFormat;
	        addPipeline(pRenderer, &pipelineDesc, &pSkyboxPipeline);
	        
			if (!pSkyboxPipeline) 
			{
	    		LOGF(LogLevel::eERROR, "Failed to add skybox pipeline.");
	    		return false;
	    	}
		}
		
		
		///
	    // Init samplers & load textures
	    
		SamplerDesc samplerDesc = { FILTER_LINEAR,
                                    FILTER_LINEAR,
                                    MIPMAP_MODE_NEAREST,
                                    ADDRESS_MODE_CLAMP_TO_EDGE,
                                    ADDRESS_MODE_CLAMP_TO_EDGE,
                                    ADDRESS_MODE_CLAMP_TO_EDGE };
        addSampler(pRenderer, &samplerDesc, &pSampler); 
		
		// Height map texture
		TextureLoadDesc heightMapDesc = {};
        heightMapDesc.pFileName = "height_map.tex";
        heightMapDesc.ppTexture = &pHeightMap;
        heightMapDesc.mCreationFlag = TEXTURE_CREATION_FLAG_SRGB;
        addResource(&heightMapDesc, NULL);
        
        // Skybox textures
        const char* skyboxNames[] = { "skybox_back.tex",  "skybox_left.tex",   "skybox_front.tex",
                                      "skybox_right.tex", "skybox_bottom.tex", "skybox_top.tex" };
        for (uint32_t i = 0; i < 6; i += 1) {
        	TextureLoadDesc skyboxDesc = {};
	        skyboxDesc.pFileName = skyboxNames[i];
	        skyboxDesc.ppTexture = &pSkyboxTextures[i];
	        skyboxDesc.mCreationFlag = TEXTURE_CREATION_FLAG_SRGB;
	        addResource(&skyboxDesc, NULL);
        }
        
        // Grass meshes
        for (uint32_t i = 0; i < TF_ARRAY_COUNT(pGrassGeoms); i += 1) {
	    
	    	char *filename = (char*)tempAlloc(128);
	    	sprintf(filename, "grass_lod_%i.bin", i);
	    
		    GeometryLoadDesc loadDesc = {};
	        loadDesc.pFileName = filename;
	        loadDesc.pVertexLayout = &gGrassVertexLayoutForLoading;
	        loadDesc.ppGeometry = &pGrassGeoms[i];
	        loadDesc.ppGeometryData = &pGrassGeomDatas[i];
	        loadDesc.mFlags = GEOMETRY_LOAD_FLAG_SHADOWED;
	        addResource(&loadDesc, NULL);
	    }
        
        waitForAllResourceLoads();
        
        tf_free(grassInstanceData);
        
        ///
        // Check resource loading result
        if (!pHeightMap) 
        {
        	LOGF(LogLevel::eERROR, "Failed to load height map.");
        	return false;
        }
        for (uint32_t i = 0; i < TF_ARRAY_COUNT(pGrassGeoms); i += 1) {
			if (!pGrassGeoms[i] || !pGrassGeomDatas[i]) {
				LOGF(LogLevel::eERROR, "Failed to load grass.");
	        	return false;
			}
	    }
	    for (uint32_t i = 0; i < 6; i += 1) {
	    	if (!pSkyboxTextures[i]) {
	    		LOGF(LogLevel::eERROR, "Failed to load skybox texture.");
	        	return false;
	    	}
        }

        // Combine grash meshes into one vbo
        
        uint32_t totalVertexCount = 0;
        uint32_t totalIndexCount = 0;
        for (uint32_t i = 0; i < NUMBER_OF_GRASS_LOD; i += 1)
        {
        	totalVertexCount += pGrassGeoms[i]->mVertexCount;
        	totalIndexCount += pGrassGeoms[i]->mIndexCount;
        	
        	gGrassDrawUniformData.mLod.mLevels[i].mIndexCount = pGrassGeoms[i]->mIndexCount;
        }
        
        GrassVertex *vertices = (GrassVertex*)tempAlloc(sizeof(GrassVertex)*totalVertexCount);
        uint32_t *indices = (uint32_t*)tempAlloc(sizeof(uint32_t)*totalIndexCount);

		uint32_t nextBaseVertex = 0;        
		uint32_t nextBaseIndex = 0;        
        for (uint32_t i = 0; i < NUMBER_OF_GRASS_LOD; i += 1)
        {
        	
        	for (uint32_t j = 0; j < pGrassGeoms[i]->mVertexCount; j += 1)
            {
        		vertices[nextBaseVertex+j].mPosition
        			= ((float3*)pGrassGeomDatas[i]->pShadow->pAttributes[SEMANTIC_POSITION])[j];
        		vertices[nextBaseVertex+j].mNormal
        			= ((uint32_t*)pGrassGeomDatas[i]->pShadow->pAttributes[SEMANTIC_NORMAL])[j];
        	}
            for (uint32_t j = 0; j < pGrassGeoms[i]->mIndexCount; j += 1)
            {
        		indices[nextBaseIndex+j] = ((uint16_t*)pGrassGeomDatas[i]->pShadow->pIndices)[j] + nextBaseVertex;
        	}
        	
        	nextBaseVertex += pGrassGeoms[i]->mVertexCount;
        	nextBaseIndex += pGrassGeoms[i]->mIndexCount;
        }
        
        BufferLoadDesc vboDesc = {};
	    vboDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_VERTEX_BUFFER;
	    vboDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
	    vboDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_NONE;
	    vboDesc.mDesc.mSize = totalVertexCount*sizeof(GrassVertex);
	    vboDesc.pData = vertices;
	    vboDesc.ppBuffer = &pGrassVbo;
	    addResource(&vboDesc, nullptr);
        BufferLoadDesc iboDesc = {};
	    vboDesc.mDesc.mDescriptors = DESCRIPTOR_TYPE_INDEX_BUFFER;
	    vboDesc.mDesc.mMemoryUsage = RESOURCE_MEMORY_USAGE_GPU_ONLY;
	    vboDesc.mDesc.mFlags = BUFFER_CREATION_FLAG_NONE;
	    vboDesc.mDesc.mSize = totalIndexCount*sizeof(uint32_t);
	    vboDesc.pData = indices;
	    vboDesc.ppBuffer = &pGrassIbo;
	    addResource(&vboDesc, nullptr);
	    
	    waitForAllResourceLoads();
	    
	    
        { // terrain ubo descriptor set
	    	DescriptorSetDesc setDesc = { pRootSignature, DESCRIPTOR_UPDATE_FREQ_PER_FRAME, gNumberOfFrames };
	        addDescriptorSet(pRenderer, &setDesc, &pDescriptorSetTerrainUbo);
		    for (uint32_t i = 0; i < gNumberOfFrames; i += 1) {
		    	DescriptorData params[1] = {};
		    	params[0].mCount = 1;
	            params[0].pName = "scene";
	            params[0].ppBuffers = &pSceneUbos[i];
	            updateDescriptorSet(pRenderer, i, pDescriptorSetTerrainUbo, 1, params);
		    }
	    }
	    
        { // grass ubo descriptor set
	    	DescriptorSetDesc setDesc = { pRootSignature, DESCRIPTOR_UPDATE_FREQ_PER_FRAME, gNumberOfFrames };
    		addDescriptorSet(pRenderer, &setDesc, &pDescriptorSetGrass);
    		DescriptorData params[2] = {};
            for (uint32_t i = 0; i < gNumberOfFrames; i += 1)
            {
	    	    params[0].mCount = 1;
                params[0].pName = "tileData";
                params[0].ppBuffers = &pGrassTileBuffer;
            
                params[1].mCount = 1;
                params[1].pName = "scene";
                params[1].ppBuffers = &pSceneUbos[i];
            
                updateDescriptorSet(pRenderer, i, pDescriptorSetGrass, 2, params);
            }
	    }
        { // grass draw call compute set
	    	DescriptorSetDesc setDesc = { pGrassDrawRootSignature, DESCRIPTOR_UPDATE_FREQ_PER_FRAME, gNumberOfFrames };
    		addDescriptorSet(pRenderer, &setDesc, &pDescriptorSetGrassDrawCompute);
    		
    		for (uint32_t i = 0; i < gNumberOfFrames; i += 1) {
	    		DescriptorData params[3] = {};
	    	    params[0].mCount = 1;
	            params[0].pName = "scene";
	            params[0].ppBuffers = &pSceneUbos[i];
	            
	    	    params[1].mCount = 1;
	            params[1].pName = "drawBuffer";
	            params[1].ppBuffers = &pGrassDrawBuffer;
	        
	    	    params[2].mCount = 1;
	            params[2].pName = "drawInfo";
	            params[2].ppBuffers = &pGrassDrawUbos[i];
	        
	            updateDescriptorSet(pRenderer, i, pDescriptorSetGrassDrawCompute, 3, params);
    		}
    		
	    }
	    
        { // skybox ubo descriptor set
	    	DescriptorSetDesc setDesc = { pRootSignature, DESCRIPTOR_UPDATE_FREQ_PER_FRAME, gNumberOfFrames };
	        addDescriptorSet(pRenderer, &setDesc, &pDescriptorSetSkyboxUbos);
		    for (uint32_t i = 0; i < gNumberOfFrames; i += 1) {
		    	DescriptorData params[2] = {};
		    	params[0].mCount = 1;
	            params[0].pName = "skyboxData";
	            params[0].ppBuffers = &pSkyboxUbos[i];
	            params[1].mCount = 1;
	            params[1].pName = "scene";
	            params[1].ppBuffers = &pSceneUbos[i];
	            updateDescriptorSet(pRenderer, i, pDescriptorSetSkyboxUbos, 2, params);
		    }
		    
		    setDesc = { pRootSignature, DESCRIPTOR_UPDATE_FREQ_NONE, 1 };
	        addDescriptorSet(pRenderer, &setDesc, &pDescriptorSetSkyboxTextures);
		    
			const char* skyboxDescNames[] = { "SkyboxBack",  "SkyboxLeft",   "SkyboxFront",
                                              "SkyboxRight", "SkyboxBottom", "SkyboxTop" };
	    	DescriptorData params[7] = {};
	    	params[0].pName = "Sampler";
	        params[0].ppSamplers = &pSampler;
	        params[0].mCount = 1;
		    for (uint32_t i = 0; i < 6; i += 1) {
			    params[i+1].pName = skyboxDescNames[i];
		        params[i+1].ppTextures = &pSkyboxTextures[i];
		        params[i+1].mCount = 1;
	        }
            updateDescriptorSet(pRenderer, 0, pDescriptorSetSkyboxTextures, 7, params);
	    }
        
	    { // textures descriptor set (never updated)
	    	DescriptorSetDesc setDesc = { pRootSignature, DESCRIPTOR_UPDATE_FREQ_NONE, 1 };
	        addDescriptorSet(pRenderer, &setDesc, &pDescriptorSetHeightMap);
		    DescriptorData params[2] = {};
		    params[0].pName = "HeightMap";
	        params[0].ppTextures = &pHeightMap;
	        params[0].mCount = 1;
		    params[1].pName = "Sampler";
	        params[1].ppSamplers = &pSampler;
	        params[1].mCount = 1;
	        
	        updateDescriptorSet(pRenderer, 0, pDescriptorSetHeightMap, 2, params);
	    }
	    
    	DescriptorSetDesc setDesc = { pGrassDrawRootSignature, DESCRIPTOR_UPDATE_FREQ_NONE, 1 };
        addDescriptorSet(pRenderer, &setDesc, &pDescriptorSetHeightMapDrawCompute);
	    DescriptorData  params[1] = {};
	    params[0].pName = "HeightMap";
        params[0].ppTextures = &pHeightMap;
        params[0].mCount = 1;
        
        updateDescriptorSet(pRenderer, 0, pDescriptorSetHeightMapDrawCompute, 1, params);
		
		///
		// Load UI
		UserInterfaceLoadDesc uiLoad = {};
        uiLoad.mColorFormat = pSwapChain->ppRenderTargets[0]->mFormat;
        uiLoad.mHeight = mSettings.mHeight;
        uiLoad.mWidth = mSettings.mWidth;
        uiLoad.mLoadType = pReloadDesc->mType;
        loadUserInterface(&uiLoad);

		///
		// Load font system
        FontSystemLoadDesc fontLoad = {};
        fontLoad.mColorFormat = pSwapChain->ppRenderTargets[0]->mFormat;
        fontLoad.mHeight = mSettings.mHeight;
        fontLoad.mWidth = mSettings.mWidth;
        fontLoad.mLoadType = pReloadDesc->mType;
        loadFontSystem(&fontLoad);
		
        return true;
    }

    void Unload(ReloadDesc *pReloadDesc)
    {
        waitQueueIdle(pGraphicsQueue);
        
        unloadFontSystem(pReloadDesc->mType);
        unloadUserInterface(pReloadDesc->mType);
        
        removeResource(pHeightMap);
        for (uint32_t i = 0; i < 6; i += 1) removeResource(pSkyboxTextures[i]);
        
        for (uint32_t i = 0; i < TF_ARRAY_COUNT(pGrassGeoms); i += 1) {
        	removeResource(pGrassGeoms[i]);
        	removeResource(pGrassGeomDatas[i]);
	    }
        
        removeSampler(pRenderer, pSampler);
        
        removeRootSignature(pRenderer, pRootSignature);
        removeRootSignature(pRenderer, pGrassDrawRootSignature);
        
        removePipeline(pRenderer, pTerrainPipeline);
        removePipeline(pRenderer, pGrassPipeline);
        removePipeline(pRenderer, pSkyboxPipeline);
        removePipeline(pRenderer, pGrassDrawComputePipeline);
        
        removeDescriptorSet(pRenderer, pDescriptorSetGrass);
        removeDescriptorSet(pRenderer, pDescriptorSetGrassDrawCompute);
        removeDescriptorSet(pRenderer, pDescriptorSetSkyboxUbos);
        removeDescriptorSet(pRenderer, pDescriptorSetSkyboxTextures);
        removeDescriptorSet(pRenderer, pDescriptorSetTerrainUbo);
        removeDescriptorSet(pRenderer, pDescriptorSetHeightMap);
        removeDescriptorSet(pRenderer, pDescriptorSetHeightMapDrawCompute);
        
        for (uint32_t i = 0; i < gNumberOfFrames; i += 1) removeResource(pSceneUbos[i]);
        removeResource(pGrassTileBuffer);
        removeResource(pGrassInstanceVbo);
        removeResource(pGrassVbo);
        removeResource(pGrassIbo);
        for (uint32_t i = 0; i < gNumberOfFrames; i += 1) removeResource(pGrassDrawUbos[i]);
        removeResource(pGrassDrawBuffer);
        for (uint32_t i = 0; i < gNumberOfFrames; i += 1) removeResource(pSkyboxUbos[i]);
        
    	removeShader(pRenderer, pTerrainShader);
    	removeShader(pRenderer, pGrassShader);
    	removeShader(pRenderer, pSkyboxShader);
    	removeShader(pRenderer, pGrassDrawShader);
    	
        removeSwapChain(pRenderer, pSwapChain);
        
        removeRenderTarget(pRenderer, pDepthBuffer);
        
        uiRemoveComponent(pGuiWindow);
        unloadProfilerUI();
    }

    void Update(float deltaTime)
    { 
    
    	resetTemporaryStorage();
    
    	///
	    // Update camera
    
    	if (!uiIsFocused()) {
	    	pCameraController->onMove({ inputGetValue(0, CUSTOM_MOVE_X), inputGetValue(0, CUSTOM_MOVE_Y) });
	        pCameraController->onRotate({ inputGetValue(0, CUSTOM_LOOK_X), inputGetValue(0, CUSTOM_LOOK_Y) });
	        pCameraController->onMoveY(inputGetValue(0, CUSTOM_MOVE_UP));
	        
	        if (inputGetValue(0, CUSTOM_TOGGLE_UI))
            {
                uiToggleActive();
            }
    	}
    
    	pCameraController->update(deltaTime);
    	
    	mat4 viewMat = pCameraController->getViewMatrix();

        const float aspectInverse = (float)mSettings.mHeight / (float)mSettings.mWidth;
        const float horizontal_fov = PI / 2.0f;
        CameraMatrix projMat = CameraMatrix::perspectiveReverseZ(horizontal_fov, aspectInverse, 0.1f, 10000.0f);
        CameraMatrix pv = projMat * viewMat;
        
        gSceneUniformData.mCameraToClip = pv;
        
        static float currentTime = 0.0;
        currentTime += deltaTime;
        
	    gSceneUniformData.mViewDir = (pCameraController->getViewMatrix() * Vector4(0, 0, 1, 0.0)).getXYZ();
	    gSceneUniformData.mCameraPos = pCameraController->getViewPosition();
	    gSceneUniformData.mTime = currentTime;
	    gSceneUniformData.mMaxInstancesPerTile = MAX_GRASS_CAP/GRASS_TILE_COUNT;
	    
	    gSkyboxUniformData.mView = pCameraController->getViewMatrix();
	    gSkyboxUniformData.mProjection = projMat;
	    
	    gGrassDrawUniformData.mViewPosition =
            float3(pCameraController->getViewPosition().getX(), pCameraController->getViewPosition().getY(),
                                                     pCameraController->getViewPosition().getZ());
        
        CameraMatrix::extractFrustumClipPlanes(
            pv, 
        	gGrassDrawUniformData.rcp,
        	gGrassDrawUniformData.lcp,
        	gGrassDrawUniformData.tcp,
        	gGrassDrawUniformData.bcp,
        	gGrassDrawUniformData.fcp,
        	gGrassDrawUniformData.ncp,
        	true
    	);
    	
    	// Forward horizontal plane
        gGrassDrawUniformData.fhp = Vector4(normalize(cross(gGrassDrawUniformData.rcp.getXYZ(), Vector3(0, 1, 0))), 1);
		
		// Forward vertical plane
        gGrassDrawUniformData.fvp = Vector4(normalize(cross(Vector3(0, 1, 0), gGrassDrawUniformData.rcp.getXYZ())), 1);
    }

    void Draw()
    {
        if ((bool)pSwapChain->mEnableVsync != mSettings.mVSyncEnabled)
        {
            waitQueueIdle(pGraphicsQueue);
            ::toggleVSync(pRenderer, &pSwapChain);
        }
        
        // Grab next frame
        uint32_t swapchainImageIndex;
        acquireNextImage(pRenderer, pSwapChain, pImageAcquiredSemaphore, NULL, &swapchainImageIndex);
        RenderTarget *pRenderTarget = pSwapChain->ppRenderTargets[swapchainImageIndex];
        GpuCmdRingElement elem = getNextGpuCmdRingElement(&gGraphicsCmdRing, true, 1);
        
        // Wait for last command buffer to be done on this frame (as it is potentially still being used)
        waitForFences(pRenderer, 1, &elem.pFence);
        
        
        // Update scene ubo
        BufferUpdateDesc bufferUpdateDesc = { pSceneUbos[gFrameIndex] };
        beginUpdateResource(&bufferUpdateDesc);
        memcpy(bufferUpdateDesc.pMappedData, &gSceneUniformData, sizeof(SceneUniformData));
        endUpdateResource(&bufferUpdateDesc);
        
        // Update grass draw ubo
        bufferUpdateDesc = { pGrassDrawUbos[gFrameIndex] };
        beginUpdateResource(&bufferUpdateDesc);
        memcpy(bufferUpdateDesc.pMappedData, &gGrassDrawUniformData, sizeof(GrassDrawUniformData));
        endUpdateResource(&bufferUpdateDesc);
        
        // Update skybox ubo
        bufferUpdateDesc = { pSkyboxUbos[gFrameIndex] };
        beginUpdateResource(&bufferUpdateDesc);
        memcpy(bufferUpdateDesc.pMappedData, &gSkyboxUniformData, sizeof(gSkyboxUniformData));
        endUpdateResource(&bufferUpdateDesc);
        
        resetCmdPool(pRenderer, elem.pCmdPool);
        
        Cmd* cmd = elem.pCmds[0];
        beginCmd(cmd);
        
        cmdBeginGpuFrameProfile(cmd, gGpuProfileToken);
        
        // Bind render targets
		RenderTargetBarrier barriers[] = {
			{ pRenderTarget, RESOURCE_STATE_PRESENT, RESOURCE_STATE_RENDER_TARGET },
		};
        cmdResourceBarrier(cmd, 0, NULL, 0, NULL, 1, barriers);
        BindRenderTargetsDesc bindRenderTargets = {};
        bindRenderTargets.mRenderTargetCount = 1;
        bindRenderTargets.mRenderTargets[0] = { pRenderTarget, LOAD_ACTION_CLEAR };
        bindRenderTargets.mDepthStencil = { pDepthBuffer, LOAD_ACTION_CLEAR };
        cmdBindRenderTargets(cmd, &bindRenderTargets); // Load action is CLEAR, so render target will be cleared here
         
        cmdSetViewport(cmd, 0.0f, 0.0f, (float)pRenderTarget->mWidth, (float)pRenderTarget->mHeight, 0.0f, 1.0f);
        cmdSetScissor(cmd, 0, 0, pRenderTarget->mWidth, pRenderTarget->mHeight);
        
        
        ///
        // Draw skybox
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw Skybox");
        cmdBindPipeline(cmd, pSkyboxPipeline);
        cmdBindDescriptorSet(cmd, gFrameIndex, pDescriptorSetSkyboxUbos);
        cmdBindDescriptorSet(cmd, 0, pDescriptorSetSkyboxTextures);
        // 6 verts * 6 faces
        cmdDraw(cmd, 6*6, 0);
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);
        
        ///
        // Draw terrain
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw terrain");
        cmdBindPipeline(cmd, pTerrainPipeline);
        cmdBindDescriptorSet(cmd, gFrameIndex, pDescriptorSetTerrainUbo);
        cmdBindDescriptorSet(cmd, 0, pDescriptorSetHeightMap);
        
        uint32_t numberOfQuads = (uint32_t)
        	((gSceneUniformData.mTerrainSize.getX()/gSceneUniformData.mSampleGranularity)
        	* (gSceneUniformData.mTerrainSize.getY()/gSceneUniformData.mSampleGranularity));
        	
        cmdDraw(cmd, numberOfQuads*6, 0);
        
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);
        
        ///
        // Compute grass draw calls
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Compute grass draw calls");
        cmdBindPipeline(cmd, pGrassDrawComputePipeline);
        
        cmdBindDescriptorSet(cmd, gFrameIndex, pDescriptorSetGrassDrawCompute);
        cmdBindDescriptorSet(cmd, 0, pDescriptorSetHeightMapDrawCompute);
        
        cmdDispatch(cmd, (uint32_t)ceil(GRASS_TILE_COUNT_X/32.0f), (uint32_t)ceil(GRASS_TILE_COUNT_Y/32.0f), 1);
        
        BufferBarrier drawBufferBarrier = { pGrassDrawBuffer, RESOURCE_STATE_SHADER_RESOURCE, RESOURCE_STATE_INDIRECT_ARGUMENT };
        cmdResourceBarrier(cmd, 1, &drawBufferBarrier, 0, NULL, 0, NULL);
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);
        
        ///
        // Draw grass
        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw grass");
    	cmdBindPipeline(cmd, pGrassPipeline);
        cmdBindDescriptorSet(cmd, 0, pDescriptorSetHeightMap);
        cmdBindDescriptorSet(cmd, gFrameIndex, pDescriptorSetGrass);
        
        cmdBindIndexBuffer(cmd, pGrassIbo, INDEX_TYPE_UINT32, 0);
        
        // We bind Instance vbo + LOD models
        uint32_t strides[2] = { sizeof(GrassVertex), sizeof(uint32_t) };
        uint64_t offsets[2] = { 0, 0 };
        Buffer   *vbos[2]   = { pGrassVbo, pGrassInstanceVbo };
        
        cmdBindVertexBuffer(cmd, 2, vbos, strides, offsets);
        
        cmdExecuteIndirect(cmd, INDIRECT_DRAW_INDEX, GRASS_TILE_COUNT, pGrassDrawBuffer, 0, NULL, 0);
        
        drawBufferBarrier = { pGrassDrawBuffer, RESOURCE_STATE_INDIRECT_ARGUMENT, RESOURCE_STATE_SHADER_RESOURCE };
        cmdResourceBarrier(cmd, 1, &drawBufferBarrier, 0, NULL, 0, NULL);
        
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);
        
        ///
        // Draw UI
        
        FontDrawDesc infoDraw;
        infoDraw.mFontColor = 0xff00ffff;
        infoDraw.mFontSize = 18.0f;
        infoDraw.mFontID = gFontID;
        float2 txtSizePx = cmdDrawCpuProfile(cmd, float2(8.f, 15.f), &infoDraw);
        float2 textPos = float2(8.f, txtSizePx.y + 75.f);
        cmdDrawGpuProfile(cmd, textPos, gGpuProfileToken, &infoDraw);

        cmdBeginGpuTimestampQuery(cmd, gGpuProfileToken, "Draw UI");
        cmdDrawUserInterface(cmd);
        cmdEndGpuTimestampQuery(cmd, gGpuProfileToken);
        
        cmdBindRenderTargets(cmd, NULL);
        
        barriers[0] = { pRenderTarget, RESOURCE_STATE_RENDER_TARGET, RESOURCE_STATE_PRESENT };
        cmdResourceBarrier(cmd, 0, NULL, 0, NULL, 1, barriers);
        
        cmdEndGpuFrameProfile(cmd, gGpuProfileToken);
        
        endCmd(cmd);
        
        FlushResourceUpdateDesc flushUpdateDesc = {};
        flushUpdateDesc.mNodeIndex = 0;
        flushResourceUpdates(&flushUpdateDesc);
        
        Semaphore* waitSemaphores[] = { flushUpdateDesc.pOutSubmittedSemaphore, pImageAcquiredSemaphore };
        
        // Submit commands
        QueueSubmitDesc submitDesc = {};
        submitDesc.mCmdCount = 1;
        submitDesc.mSignalSemaphoreCount = 1;
        submitDesc.mWaitSemaphoreCount = TF_ARRAY_COUNT(waitSemaphores);
        submitDesc.ppCmds = &cmd;
        submitDesc.ppSignalSemaphores = &elem.pSemaphore;
        submitDesc.ppWaitSemaphores = waitSemaphores;
        submitDesc.pSignalFence = elem.pFence;
        uint64_t submitToken = cpuProfileEnter(gQueueSubmitToken);
        queueSubmit(pGraphicsQueue, &submitDesc);
        cpuProfileLeave(gQueueSubmitToken, submitToken);

		// Present
        QueuePresentDesc presentDesc = {};
        presentDesc.mIndex = (uint8_t)swapchainImageIndex;
        presentDesc.mWaitSemaphoreCount = 1;
        presentDesc.pSwapChain = pSwapChain;
        presentDesc.ppWaitSemaphores = &elem.pSemaphore;
        presentDesc.mSubmitDone = true;

        queuePresent(pGraphicsQueue, &presentDesc);
        flipProfiler();
        
        gFrameIndex = (gFrameIndex + 1) % gNumberOfFrames;
    }

    const char *GetName() { return "Charlie_Submission"; }
};

void addUiWidgets() {
	SliderUintWidget numberOfGrassWidget;
    numberOfGrassWidget.mMin = 0;
    numberOfGrassWidget.mMax = MAX_GRASS_CAP;
    numberOfGrassWidget.mStep = 1;
    numberOfGrassWidget.pData = &gGrassDrawUniformData.mPerceivedNumberOfGrass;
    uiAddComponentWidget(pGuiWindow, "Perceived number of grass", &numberOfGrassWidget, WIDGET_TYPE_SLIDER_UINT);
    
    SliderFloat3Widget sunDirWidget;
    sunDirWidget.pData = (float3*)&gSceneUniformData.mSunDirection;
    sunDirWidget.mMin = float3(-1);
    sunDirWidget.mMax = float3(1);
    uiAddComponentWidget(pGuiWindow, "Sun direction", &sunDirWidget, WIDGET_TYPE_SLIDER_FLOAT3);
    
    SliderFloatWidget floatWidget;
    floatWidget.pData = &gSceneUniformData.mWindStrength;
    floatWidget.mMin = 0;
    floatWidget.mMax = 1;
    uiAddComponentWidget(pGuiWindow, "Wind Strength", &floatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    floatWidget.pData = &gSceneUniformData.mMaxWindLeanAngle;
    floatWidget.mMin = 0;
    floatWidget.mMax = TAU*0.5f;
    uiAddComponentWidget(pGuiWindow, "Max Wind Bend Angle", &floatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    floatWidget.pData = &gSceneUniformData.mWindSpeed;
    floatWidget.mMin = 0;
    floatWidget.mMax = 300;
    uiAddComponentWidget(pGuiWindow, "Wind Speed", &floatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    floatWidget.pData = &gSceneUniformData.mMaxNaturalAngle;
    floatWidget.mMin = 0;
    floatWidget.mMax = TAU*0.5;
    uiAddComponentWidget(pGuiWindow, "Max Natural Angle", &floatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    floatWidget.pData = &gSceneUniformData.mMinGrassWidth;
    floatWidget.mMin = 0;
    floatWidget.mMax = 3;
    uiAddComponentWidget(pGuiWindow, "Min grass width", &floatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    floatWidget.pData = &gSceneUniformData.mMaxGrassWidth;
    floatWidget.mMin = 0;
    floatWidget.mMax = 3;
    uiAddComponentWidget(pGuiWindow, "Max grass width", &floatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    floatWidget.pData = &gSceneUniformData.mMinGrassHeight;
    floatWidget.mMin = 0;
    floatWidget.mMax = 35;
    uiAddComponentWidget(pGuiWindow, "Min grass height", &floatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    floatWidget.pData = &gSceneUniformData.mMaxGrassHeight;
    floatWidget.mMin = 0;
    floatWidget.mMax = 35;
    uiAddComponentWidget(pGuiWindow, "Max grass height", &floatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    floatWidget.pData = &gSceneUniformData.mDaylightFactor;
    floatWidget.mMin = 0;
    floatWidget.mMax = 2;
    uiAddComponentWidget(pGuiWindow, "Daylight", &floatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    floatWidget.pData = &gSceneUniformData.mMaxFloorY;
    floatWidget.mMin = 1;
    floatWidget.mMax = 300;
    uiAddComponentWidget(pGuiWindow, "Slope Levels", &floatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    
    SliderFloat3Widget windDirWidget;
    windDirWidget.pData = (float3*)&gSceneUniformData.mWindDir;
    windDirWidget.mMin = float3(-1);
    windDirWidget.mMax = float3(1);
    uiAddComponentWidget(pGuiWindow, "Wind direction", &windDirWidget, WIDGET_TYPE_SLIDER_FLOAT3);
    
    
    SliderFloatWidget lodFloatWidget;
    lodFloatWidget.mMin = 0.01f;
    lodFloatWidget.mMax = 0.99f;
    lodFloatWidget.pData = &gGrassDrawUniformData.mLod.mDensityFadeStartPercent;
    uiAddComponentWidget(pGuiWindow, "Density Fade Start %%", &lodFloatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    lodFloatWidget.mMin = 0.01f;
    lodFloatWidget.mMax = 0.99f;
    lodFloatWidget.pData = &gGrassDrawUniformData.mLod.mMinDensityPercent;
    uiAddComponentWidget(pGuiWindow, "Min density %%", &lodFloatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    lodFloatWidget.mMin = 100.0f;
    lodFloatWidget.mMax = 4000.0f;
    lodFloatWidget.pData = &gGrassDrawUniformData.mLod.mLowestDetailDistance;
    uiAddComponentWidget(pGuiWindow, "Lowest detail distance", &lodFloatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    lodFloatWidget.mMin = 1.0f;
    lodFloatWidget.mMax = 4000.0f;
    const char *labels[NUMBER_OF_GRASS_LOD] = {
    	"LOD Threshold 0",
    	"LOD Threshold 1",
    	"LOD Threshold 2",
    	"LOD Threshold 3",
    };
    for (uint32_t i = 0; i < NUMBER_OF_GRASS_LOD; i += 1) {
    	lodFloatWidget.pData = &gGrassDrawUniformData.mLod.mLevels[i].mThreshold;
    	uiAddComponentWidget(pGuiWindow, labels[i], &lodFloatWidget, WIDGET_TYPE_SLIDER_FLOAT);
    }
    
    Color3PickerWidget baseColorWidget;
    baseColorWidget.pData = (float3*)&gSceneUniformData.mGrassBaseColor;
    uiAddComponentWidget(pGuiWindow, "Grass base color", &baseColorWidget, WIDGET_TYPE_COLOR3_PICKER);
    baseColorWidget.pData = (float3*)&gSceneUniformData.mGrassTipColor;
    uiAddComponentWidget(pGuiWindow, "Grass tip color", &baseColorWidget, WIDGET_TYPE_COLOR3_PICKER);
}



DEFINE_APPLICATION_MAIN(Charlie_Submission)
