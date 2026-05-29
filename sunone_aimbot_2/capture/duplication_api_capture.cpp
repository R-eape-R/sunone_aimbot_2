#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <thread>
#include <d3dcompiler.h>

#ifdef USE_CUDA
#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#endif

#include "duplication_api_capture.h"
#include "sunone_aimbot_2.h"
#include "dml_detector.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

extern DirectMLDetector* dml_detector;

template <typename T>
inline void SafeRelease(T** ppInterface)
{
    if (*ppInterface)
    {
        (*ppInterface)->Release();
        *ppInterface = nullptr;
    }
}

const char* HLSL_COMPUTE_SHADER = R"(
Texture2D<unorm float4> InputTex : register(t0);
SamplerState LinearSampler : register(s0);
RWBuffer<float> OutputBuf : register(u0);

cbuffer Config : register(b0) {
    uint capWidth;
    uint capHeight;
    uint modelWidth;
    uint modelHeight;
    uint circleFovEnabled;
    uint circleFovRadiusPercent;
    uint pad1;
    uint pad2;
};

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= modelWidth || id.y >= modelHeight) return;
    
    uint planeSize = modelWidth * modelHeight;
    uint idx = id.y * modelWidth + id.x;
    
    if (circleFovEnabled != 0) {
        float cx = (float)modelWidth * 0.5f;
        float cy = (float)modelHeight * 0.5f;
        float dx = (float)id.x + 0.5f - cx;
        float dy = (float)id.y + 0.5f - cy;
        
        float maxR = (float)min(modelWidth, modelHeight) * 0.5f;
        float r = maxR * ((float)circleFovRadiusPercent / 100.0f);
        
        if ((dx * dx + dy * dy) > (r * r)) {
            OutputBuf[idx]                 = 0.0f; // Zero out R
            OutputBuf[planeSize + idx]     = 0.0f; // Zero out G
            OutputBuf[2 * planeSize + idx] = 0.0f; // Zero out B
            return;
        }
    }
    
    float2 uv = float2((id.x + 0.5f) / (float)modelWidth, (id.y + 0.5f) / (float)modelHeight);
    float4 pixel = InputTex.SampleLevel(LinearSampler, uv, 0);
    
    OutputBuf[idx]                 = pixel.z; // R
    OutputBuf[planeSize + idx]     = pixel.y; // G
    OutputBuf[2 * planeSize + idx] = pixel.x; // B
}
)";

struct FrameContext
{
    ID3D11Texture2D* texture = nullptr;
    bool hasAcquiredFrame = false;
};

class DDAManager
{
public:
    DDAManager()
        : m_device(nullptr)
        , m_context(nullptr)
        , m_duplication(nullptr)
        , m_output1(nullptr)
        , m_sharedTexture(nullptr)
        , m_computeShader(nullptr)
        , m_textureSRV(nullptr)
        , m_sampler(nullptr)
        , m_tensorBuffer(nullptr)
        , m_tensorStagingBuffer(nullptr)
        , m_tensorUAV(nullptr)
        , m_constantBuffer(nullptr)
        , m_frameAcquired(false)
        , m_initialized(false)
    {
        ZeroMemory(&m_duplDesc, sizeof(m_duplDesc));
        ZeroMemory(&m_stagingBox, sizeof(m_stagingBox));
    }

    ~DDAManager()
    {
        Release();
    }

    HRESULT Initialize(
        int monitorIndex,
        int captureWidth,
        int captureHeight,
        int modelWidth,
        int modelHeight,
        int& outScreenWidth,
        int& outScreenHeight,
        ID3D11Device** outDevice,
        ID3D11DeviceContext** outContext)
    {
        HRESULT hr = S_OK;
        Release();

        m_capWidth = captureWidth;
        m_capHeight = captureHeight;
        m_modelWidth = modelWidth;
        m_modelHeight = modelHeight;

        IDXGIFactory1* factory = nullptr;
        hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory);
        if (FAILED(hr)) return hr;

        IDXGIAdapter1* adapter = nullptr;
        IDXGIOutput* output = nullptr;
        const int targetMonitorIndex = std::max(0, monitorIndex);

        int currentMonitorIndex = 0;
        bool foundOutput = false;
        for (UINT adapterIdx = 0; ; ++adapterIdx)
        {
            IDXGIAdapter1* candidateAdapter = nullptr;
            hr = factory->EnumAdapters1(adapterIdx, &candidateAdapter);
            if (hr == DXGI_ERROR_NOT_FOUND) break;
            if (FAILED(hr)) { SafeRelease(&factory); return hr; }

            for (UINT outputIdx = 0; ; ++outputIdx)
            {
                IDXGIOutput* candidateOutput = nullptr;
                hr = candidateAdapter->EnumOutputs(outputIdx, &candidateOutput);
                if (hr == DXGI_ERROR_NOT_FOUND) break;
                if (FAILED(hr)) { SafeRelease(&candidateAdapter); SafeRelease(&factory); return hr; }

                if (currentMonitorIndex == targetMonitorIndex)
                {
                    adapter = candidateAdapter;
                    output = candidateOutput;
                    foundOutput = true;
                    break;
                }
                ++currentMonitorIndex;
                candidateOutput->Release();
            }
            if (foundOutput) break;
            candidateAdapter->Release();
        }

        if (!foundOutput || !adapter || !output)
        {
            SafeRelease(&adapter); SafeRelease(&output); SafeRelease(&factory);
            return DXGI_ERROR_NOT_FOUND;
        }

        D3D_FEATURE_LEVEL fl[] = { D3D_FEATURE_LEVEL_11_0 };
        hr = D3D11CreateDevice(adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            fl, 1, D3D11_SDK_VERSION, &m_device, nullptr, &m_context);
        if (FAILED(hr)) { SafeRelease(&output); SafeRelease(&adapter); SafeRelease(&factory); return hr; }

        IDXGIDevice1* dxgiDevice = nullptr;
        if (SUCCEEDED(m_device->QueryInterface(__uuidof(IDXGIDevice1), (void**)&dxgiDevice)))
        {
            dxgiDevice->SetMaximumFrameLatency(1);
            dxgiDevice->Release();
        }

        hr = output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&m_output1);
        if (FAILED(hr)) { SafeRelease(&m_context); SafeRelease(&m_device); SafeRelease(&output); SafeRelease(&adapter); SafeRelease(&factory); return hr; }

        hr = m_output1->DuplicateOutput(m_device, &m_duplication);
        if (FAILED(hr)) { SafeRelease(&m_output1); SafeRelease(&m_context); SafeRelease(&m_device); SafeRelease(&output); SafeRelease(&adapter); SafeRelease(&factory); return hr; }

        m_duplication->GetDesc(&m_duplDesc);
        DXGI_OUTPUT_DESC oDesc{};
        output->GetDesc(&oDesc);
        outScreenWidth = oDesc.DesktopCoordinates.right - oDesc.DesktopCoordinates.left;
        outScreenHeight = oDesc.DesktopCoordinates.bottom - oDesc.DesktopCoordinates.top;

        SafeRelease(&output); SafeRelease(&adapter); SafeRelease(&factory);

        D3D11_TEXTURE2D_DESC sharedTexDesc = {};
        sharedTexDesc.Width = (UINT)captureWidth;
        sharedTexDesc.Height = (UINT)captureHeight;
        sharedTexDesc.MipLevels = 1;
        sharedTexDesc.ArraySize = 1;
        sharedTexDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        sharedTexDesc.SampleDesc.Count = 1;
        sharedTexDesc.Usage = D3D11_USAGE_DEFAULT;
        sharedTexDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        sharedTexDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

        hr = m_device->CreateTexture2D(&sharedTexDesc, nullptr, &m_sharedTexture);
        if (FAILED(hr)) return hr;

        m_stagingBox.left = (UINT)((outScreenWidth - captureWidth) / 2);
        m_stagingBox.top = (UINT)((outScreenHeight - captureHeight) / 2);
        m_stagingBox.front = 0;
        m_stagingBox.right = m_stagingBox.left + (UINT)captureWidth;
        m_stagingBox.bottom = m_stagingBox.top + (UINT)captureHeight;
        m_stagingBox.back = 1;

        ID3DBlob* csBlob = nullptr;
        hr = D3DCompile(HLSL_COMPUTE_SHADER, strlen(HLSL_COMPUTE_SHADER), nullptr, nullptr, nullptr, "CSMain", "cs_5_0", 0, 0, &csBlob, nullptr);
        if (FAILED(hr)) return hr;

        hr = m_device->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &m_computeShader);
        csBlob->Release();
        if (FAILED(hr)) return hr;

        D3D11_SAMPLER_DESC sampDesc = {};
        sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        hr = m_device->CreateSamplerState(&sampDesc, &m_sampler);
        if (FAILED(hr)) return hr;

        hr = m_device->CreateShaderResourceView(m_sharedTexture, nullptr, &m_textureSRV);
        if (FAILED(hr)) return hr;

        D3D11_BUFFER_DESC bufDesc = {};
        bufDesc.ByteWidth = (UINT)(modelWidth * modelHeight * 3 * sizeof(float));
        bufDesc.Usage = D3D11_USAGE_DEFAULT;
        bufDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        bufDesc.StructureByteStride = sizeof(float);
        hr = m_device->CreateBuffer(&bufDesc, nullptr, &m_tensorBuffer);
        if (FAILED(hr)) return hr;

        D3D11_BUFFER_DESC stagingTensorDesc = {};
        stagingTensorDesc.ByteWidth = bufDesc.ByteWidth;
        stagingTensorDesc.Usage = D3D11_USAGE_STAGING;
        stagingTensorDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        hr = m_device->CreateBuffer(&stagingTensorDesc, nullptr, &m_tensorStagingBuffer);
        if (FAILED(hr)) return hr;

        D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R32_FLOAT;
        uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.NumElements = (UINT)(modelWidth * modelHeight * 3);
        hr = m_device->CreateUnorderedAccessView(m_tensorBuffer, &uavDesc, &m_tensorUAV);
        if (FAILED(hr)) return hr;

        D3D11_BUFFER_DESC cbd = {};
        cbd.ByteWidth = 32; 
        cbd.Usage = D3D11_USAGE_DYNAMIC;
        cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        cbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        hr = m_device->CreateBuffer(&cbd, nullptr, &m_constantBuffer);
        if (FAILED(hr)) return hr;

        if (outDevice)  *outDevice = m_device;
        if (outContext) *outContext = m_context;

        m_initialized = true;
        return S_OK;
    }

    HRESULT AcquireFrame(FrameContext& frameCtx, UINT timeout = 100)
    {
        frameCtx.texture = nullptr;
        frameCtx.hasAcquiredFrame = false;
        if (!m_duplication) return E_FAIL;

        DXGI_OUTDUPL_FRAME_INFO frameInfo{};
        IDXGIResource* resource = nullptr;

        HRESULT hr = m_duplication->AcquireNextFrame(timeout, &frameInfo, &resource);
        if (FAILED(hr)) return hr;

        frameCtx.hasAcquiredFrame = true;
        m_frameAcquired = true;

        if (resource)
        {
            hr = resource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&frameCtx.texture);
            resource->Release();
        }
        return hr;
    }

    void ExecuteComputeShader()
    {
        if (!m_initialized || !m_computeShader) return;

        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(m_context->Map(m_constantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
            uint32_t* cbData = (uint32_t*)mapped.pData;
            cbData[0] = m_capWidth;
            cbData[1] = m_capHeight;
            cbData[2] = m_modelWidth;
            cbData[3] = m_modelHeight;
            cbData[4] = config.circle_fov_enabled ? 1 : 0; 
            cbData[5] = static_cast<uint32_t>(config.circle_fov_radius_percent); 
            cbData[6] = 0; cbData[7] = 0; 
            m_context->Unmap(m_constantBuffer, 0);
        }

        m_context->CSSetShader(m_computeShader, nullptr, 0);
        m_context->CSSetSamplers(0, 1, &m_sampler);
        m_context->CSSetConstantBuffers(0, 1, &m_constantBuffer);
        m_context->CSSetShaderResources(0, 1, &m_textureSRV);
        m_context->CSSetUnorderedAccessViews(0, 1, &m_tensorUAV, nullptr);
        m_context->Dispatch((m_modelWidth + 7) / 8, (m_modelHeight + 7) / 8, 1);

        m_context->CopyResource(m_tensorStagingBuffer, m_tensorBuffer);
    }

    void CopyFromDesktopTexture(ID3D11Texture2D* srcTexture)
    {
        if (m_initialized && srcTexture) {
            m_context->CopySubresourceRegion(m_sharedTexture, 0, 0, 0, 0, srcTexture, 0, &m_stagingBox);
            ExecuteComputeShader();
        }
    }

    const float* GetTensorCPU()
    {
        if (!m_initialized || !m_tensorStagingBuffer) return nullptr;
        D3D11_MAPPED_SUBRESOURCE mapped;
        if (SUCCEEDED(m_context->Map(m_tensorStagingBuffer, 0, D3D11_MAP_READ, 0, &mapped))) {
            return static_cast<const float*>(mapped.pData);
        }
        return nullptr;
    }

    void UnlockTensorCPU()
    {
        if (m_initialized && m_tensorStagingBuffer) {
            m_context->Unmap(m_tensorStagingBuffer, 0);
        }
    }

    void ReleaseFrame()
    {
        if (!m_duplication || !m_frameAcquired)
            return;

        m_duplication->ReleaseFrame();
        m_frameAcquired = false;
    }

    void Release()
    {
        m_initialized = false;
        if (m_duplication)
        {
            ReleaseFrame();
            m_duplication->Release();
            m_duplication = nullptr;
        }
        SafeRelease(&m_sharedTexture);
        SafeRelease(&m_computeShader);
        SafeRelease(&m_textureSRV);
        SafeRelease(&m_sampler);
        SafeRelease(&m_tensorBuffer);
        SafeRelease(&m_tensorStagingBuffer);
        SafeRelease(&m_tensorUAV);
        SafeRelease(&m_constantBuffer);
        SafeRelease(&m_output1);
        SafeRelease(&m_context);
        SafeRelease(&m_device);
    }

public:
    ID3D11Device* m_device;
    ID3D11DeviceContext* m_context;
    IDXGIOutputDuplication* m_duplication;
    IDXGIOutput1* m_output1;
    DXGI_OUTDUPL_DESC m_duplDesc;
    D3D11_BOX m_stagingBox;
    int m_capWidth, m_capHeight, m_modelWidth, m_modelHeight;
    ID3D11Texture2D* m_sharedTexture;
    ID3D11ComputeShader* m_computeShader;
    ID3D11ShaderResourceView* m_textureSRV;
    ID3D11SamplerState* m_sampler;
    ID3D11Buffer* m_tensorBuffer;
    ID3D11Buffer* m_tensorStagingBuffer;
    ID3D11UnorderedAccessView* m_tensorUAV;
    ID3D11Buffer* m_constantBuffer;
    bool m_frameAcquired;
    bool m_initialized;
};

DuplicationAPIScreenCapture::DuplicationAPIScreenCapture(int desiredWidth, int desiredHeight, int monitorIndex)
    : d3dDevice(nullptr)
    , d3dContext(nullptr)
    , deskDupl(nullptr)
    , output1(nullptr)
    , sharedTexture(nullptr)
    , stagingTextureCPU(nullptr)
    , screenWidth(0)
    , screenHeight(0)
    , regionWidth(desiredWidth)
    , regionHeight(desiredHeight)
{
    m_ddaManager = std::make_unique<DDAManager>();

    int mw = desiredWidth;
    int mh = desiredHeight;

    if (dml_detector)
    {
        mw = dml_detector->model_width;
        mh = dml_detector->model_height;
    }
    else
    {
        mw = static_cast<int>(std::round(mw / 32.0f) * 32);
        mh = static_cast<int>(std::round(mh / 32.0f) * 32);
        if (mw < 32) mw = 32;
        if (mh < 32) mh = 32;
    }

    HRESULT hr = m_ddaManager->Initialize(
        monitorIndex,
        regionWidth,
        regionHeight,
        mw,
        mh,
        screenWidth,
        screenHeight,
        &d3dDevice,
        &d3dContext
    );
    if (FAILED(hr))
    {
        std::cerr << "[DDA] DDAManager Initialize failed hr=0x" << std::hex << hr << std::endl;
        return;
    }

    regionWidth = std::clamp(regionWidth, 1, std::max(1, screenWidth));
    regionHeight = std::clamp(regionHeight, 1, std::max(1, screenHeight));

    createStagingTextureCPU();
#ifdef USE_CUDA
    createCudaInteropTexture();
#endif
}

DuplicationAPIScreenCapture::~DuplicationAPIScreenCapture()
{
#ifdef USE_CUDA
    releaseCudaInteropTexture();
#endif
    if (m_ddaManager)
    {
        m_ddaManager->Release();
        m_ddaManager.reset();
    }
    SafeRelease(&stagingTextureCPU);
    SafeRelease(&sharedTexture);

    d3dDevice = nullptr;
    d3dContext = nullptr;
    deskDupl = nullptr;
    output1 = nullptr;
}

const float* DuplicationAPIScreenCapture::GetPrecomputedTensor()
{
    if (m_ddaManager) return m_ddaManager->GetTensorCPU();
    return nullptr;
}

void DuplicationAPIScreenCapture::UnlockTensor()
{
    if (m_ddaManager) m_ddaManager->UnlockTensorCPU();
}

cv::Mat DuplicationAPIScreenCapture::GetNextFrameCpu()
{
    if (!m_ddaManager || !m_ddaManager->m_duplication || !stagingTextureCPU)
        return cv::Mat();

    FrameContext frameCtx;
    HRESULT hr = m_ddaManager->AcquireFrame(frameCtx, 0);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return cv::Mat();
    }
    else if (hr == DXGI_ERROR_ACCESS_LOST ||
        hr == DXGI_ERROR_DEVICE_RESET ||
        hr == DXGI_ERROR_DEVICE_REMOVED ||
        hr == DXGI_ERROR_INVALID_CALL)
    {
        capture_method_changed.store(true);
        return cv::Mat();
    }
    else if (FAILED(hr))
    {
        std::cerr << "[DuplicationAPIScreenCapture] AcquireNextFrame (CPU) failed hr=0x"
            << std::hex << hr << std::endl;
        if (frameCtx.hasAcquiredFrame)
            m_ddaManager->ReleaseFrame();
        return cv::Mat();
    }

    if (!frameCtx.texture)
    {
        if (frameCtx.hasAcquiredFrame)
            m_ddaManager->ReleaseFrame();
        return cv::Mat();
    }

    m_ddaManager->CopyFromDesktopTexture(frameCtx.texture);

    const int copyWidth = std::min(regionWidth, std::max(1, screenWidth));
    const int copyHeight = std::min(regionHeight, std::max(1, screenHeight));
    const int left = std::max(0, (screenWidth - copyWidth) / 2);
    const int top = std::max(0, (screenHeight - copyHeight) / 2);

    D3D11_BOX sourceRegion;
    sourceRegion.left = left;
    sourceRegion.top = top;
    sourceRegion.front = 0;
    sourceRegion.right = sourceRegion.left + copyWidth;
    sourceRegion.bottom = sourceRegion.top + copyHeight;
    sourceRegion.back = 1;

    d3dContext->CopySubresourceRegion(
        stagingTextureCPU,
        0,
        0, 0, 0,
        frameCtx.texture,
        0,
        &sourceRegion
    );

    m_ddaManager->ReleaseFrame();
    frameCtx.texture->Release();

    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hrMap = d3dContext->Map(stagingTextureCPU, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hrMap))
    {
        std::cerr << "[DDA] Map stagingTextureCPU failed hr=" << std::hex << hrMap << std::endl;
        if (hrMap == DXGI_ERROR_DEVICE_REMOVED || hrMap == DXGI_ERROR_DEVICE_RESET)
            capture_method_changed.store(true);
        return cv::Mat();
    }

    cv::Mat cpuFrame(regionHeight, regionWidth, CV_8UC4);
    for (int y = 0; y < regionHeight; y++)
    {
        unsigned char* dstRow = cpuFrame.ptr<unsigned char>(y);
        unsigned char* srcRow = (unsigned char*)mapped.pData + y * mapped.RowPitch;
        memcpy(dstRow, srcRow, regionWidth * 4);
    }

    d3dContext->Unmap(stagingTextureCPU, 0);
    return cpuFrame;
}

#ifdef USE_CUDA
bool DuplicationAPIScreenCapture::GetNextFrameGpu(cv::cuda::GpuMat& gpuFrameBgra)
{
    if (!m_ddaManager || !m_ddaManager->m_duplication || !interopTextureGPU || !cudaInteropResource || !cudaInteropReady)
        return false;

    FrameContext frameCtx;
    HRESULT hr = m_ddaManager->AcquireFrame(frameCtx, 0);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return false;
    }
    else if (hr == DXGI_ERROR_ACCESS_LOST ||
        hr == DXGI_ERROR_DEVICE_RESET ||
        hr == DXGI_ERROR_DEVICE_REMOVED ||
        hr == DXGI_ERROR_INVALID_CALL)
    {
        capture_method_changed.store(true);
        return false;
    }
    else if (FAILED(hr))
    {
        std::cerr << "[DuplicationAPIScreenCapture] AcquireNextFrame (GPU) failed hr=0x"
            << std::hex << hr << std::endl;
        if (frameCtx.hasAcquiredFrame)
            m_ddaManager->ReleaseFrame();
        return false;
    }

    if (!frameCtx.texture)
    {
        if (frameCtx.hasAcquiredFrame)
            m_ddaManager->ReleaseFrame();
        return false;
    }

    const int copyWidth = std::min(regionWidth, std::max(1, screenWidth));
    const int copyHeight = std::min(regionHeight, std::max(1, screenHeight));
    const int left = std::max(0, (screenWidth - copyWidth) / 2);
    const int top = std::max(0, (screenHeight - copyHeight) / 2);

    D3D11_BOX sourceRegion;
    sourceRegion.left = left;
    sourceRegion.top = top;
    sourceRegion.front = 0;
    sourceRegion.right = sourceRegion.left + copyWidth;
    sourceRegion.bottom = sourceRegion.top + copyHeight;
    sourceRegion.back = 1;

    d3dContext->CopySubresourceRegion(
        interopTextureGPU,
        0,
        0, 0, 0,
        frameCtx.texture,
        0,
        &sourceRegion
    );

    m_ddaManager->ReleaseFrame();
    frameCtx.texture->Release();

    cudaError_t cuErr = cudaGraphicsMapResources(1, &cudaInteropResource, 0);
    if (cuErr != cudaSuccess)
    {
        std::cerr << "[DDA] cudaGraphicsMapResources failed: " << cudaGetErrorString(cuErr) << std::endl;
        cudaInteropReady = false;
        return false;
    }

    cudaArray_t mappedArray = nullptr;
    cuErr = cudaGraphicsSubResourceGetMappedArray(&mappedArray, cudaInteropResource, 0, 0);
    if (cuErr != cudaSuccess)
    {
        std::cerr << "[DDA] cudaGraphicsSubResourceGetMappedArray failed: " << cudaGetErrorString(cuErr) << std::endl;
        cudaGraphicsUnmapResources(1, &cudaInteropResource, 0);
        cudaInteropReady = false;
        return false;
    }

    gpuFrameBgra.create(regionHeight, regionWidth, CV_8UC4);

    cuErr = cudaMemcpy2DFromArray(
        gpuFrameBgra.ptr<unsigned char>(),
        gpuFrameBgra.step,
        mappedArray,
        0, 0,
        static_cast<size_t>(regionWidth) * 4,
        static_cast<size_t>(regionHeight),
        cudaMemcpyDeviceToDevice
    );

    cudaError_t unmapErr = cudaGraphicsUnmapResources(1, &cudaInteropResource, 0);
    if (unmapErr != cudaSuccess)
    {
        std::cerr << "[DDA] cudaGraphicsUnmapResources failed: " << cudaGetErrorString(unmapErr) << std::endl;
        cudaInteropReady = false;
    }

    if (cuErr != cudaSuccess)
    {
        std::cerr << "[DDA] cudaMemcpy2DFromArray failed: " << cudaGetErrorString(cuErr) << std::endl;
        cudaInteropReady = false;
        return false;
    }

    return true;
}

bool DuplicationAPIScreenCapture::createCudaInteropTexture()
{
    if (!d3dDevice)
        return false;

    releaseCudaInteropTexture();

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = regionWidth;
    desc.Height = regionHeight;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    HRESULT hr = d3dDevice->CreateTexture2D(&desc, nullptr, &interopTextureGPU);
    if (FAILED(hr))
    {
        std::cerr << "[DDA] CreateTexture2D(interop) failed hr=" << std::hex << hr << std::endl;
        return false;
    }

    cudaError_t cuErr = cudaGraphicsD3D11RegisterResource(
        &cudaInteropResource,
        interopTextureGPU,
        cudaGraphicsRegisterFlagsNone
    );

    if (cuErr != cudaSuccess)
    {
        std::cerr << "[DDA] cudaGraphicsD3D11RegisterResource failed: "
            << cudaGetErrorString(cuErr) << std::endl;
        SafeRelease(&interopTextureGPU);
        cudaInteropResource = nullptr;
        cudaInteropReady = false;
        return false;
    }

    cudaInteropReady = true;
    return true;
}

void DuplicationAPIScreenCapture::releaseCudaInteropTexture()
{
    if (cudaInteropResource)
    {
        cudaError_t cuErr = cudaGraphicsUnregisterResource(cudaInteropResource);
        if (cuErr != cudaSuccess)
        {
            std::cerr << "[DDA] cudaGraphicsUnregisterResource failed: "
                << cudaGetErrorString(cuErr) << std::endl;
        }
        cudaInteropResource = nullptr;
    }

    SafeRelease(&interopTextureGPU);
    cudaInteropReady = false;
}
#endif

bool DuplicationAPIScreenCapture::createStagingTextureCPU()
{
    if (!d3dDevice) return false;

    SafeRelease(&stagingTextureCPU);

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = regionWidth;
    desc.Height = regionHeight;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;

    HRESULT hr = d3dDevice->CreateTexture2D(&desc, nullptr, &stagingTextureCPU);
    if (FAILED(hr))
    {
        std::cerr << "[DDA] CreateTexture2D(staging) failed hr=" << std::hex << hr << std::endl;
        return false;
    }
    return true;
}
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <algorithm>
#include <iostream>

#ifdef USE_CUDA
#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#endif

#include "duplication_api_capture.h"
#include "sunone_aimbot_2.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

template <typename T>
inline void SafeRelease(T** ppInterface)
{
    if (*ppInterface)
    {
        (*ppInterface)->Release();
        *ppInterface = nullptr;
    }
}

struct FrameContext
{
    ID3D11Texture2D* texture = nullptr;
    bool hasAcquiredFrame = false;
};

class DDAManager
{
public:
    DDAManager()
        : m_device(nullptr)
        , m_context(nullptr)
        , m_duplication(nullptr)
        , m_output1(nullptr)
        , m_frameAcquired(false)
    {
        ZeroMemory(&m_duplDesc, sizeof(m_duplDesc));
    }

    ~DDAManager()
    {
        Release();
    }

    HRESULT Initialize(
        int monitorIndex,
        int /*captureWidth*/,
        int /*captureHeight*/,
        int& outScreenWidth,
        int& outScreenHeight,
        ID3D11Device** outDevice,
        ID3D11DeviceContext** outContext)
    {
        HRESULT hr = S_OK;

        IDXGIFactory1* factory = nullptr;
        hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory);
        if (FAILED(hr))
        {
            std::cerr << "[DDA] CreateDXGIFactory1 failed hr=" << std::hex << hr << std::endl;
            return hr;
        }

        IDXGIAdapter1* adapter = nullptr;
        IDXGIOutput* output = nullptr;
        const int targetMonitorIndex = std::max(0, monitorIndex);

        int currentMonitorIndex = 0;
        bool foundOutput = false;
        for (UINT adapterIdx = 0; ; ++adapterIdx)
        {
            IDXGIAdapter1* candidateAdapter = nullptr;
            hr = factory->EnumAdapters1(adapterIdx, &candidateAdapter);
            if (hr == DXGI_ERROR_NOT_FOUND)
                break;
            if (FAILED(hr))
            {
                std::cerr << "[DDA] EnumAdapters1 failed hr=" << std::hex << hr << std::endl;
                SafeRelease(&factory);
                return hr;
            }

            for (UINT outputIdx = 0; ; ++outputIdx)
            {
                IDXGIOutput* candidateOutput = nullptr;
                hr = candidateAdapter->EnumOutputs(outputIdx, &candidateOutput);
                if (hr == DXGI_ERROR_NOT_FOUND)
                    break;
                if (FAILED(hr))
                {
                    std::cerr << "[DDA] EnumOutputs failed hr=" << std::hex << hr << std::endl;
                    SafeRelease(&candidateAdapter);
                    SafeRelease(&factory);
                    return hr;
                }

                if (currentMonitorIndex == targetMonitorIndex)
                {
                    adapter = candidateAdapter;
                    output = candidateOutput;
                    foundOutput = true;
                    break;
                }

                ++currentMonitorIndex;
                candidateOutput->Release();
            }

            if (foundOutput)
                break;

            candidateAdapter->Release();
        }

        if (!foundOutput || !adapter || !output)
        {
            std::cerr << "[DDA] No monitor with index " << targetMonitorIndex << std::endl;
            SafeRelease(&adapter);
            SafeRelease(&output);
            SafeRelease(&factory);
            return DXGI_ERROR_NOT_FOUND;
        }

        {
            D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
            UINT createDeviceFlags = 0;

            hr = D3D11CreateDevice(
                adapter,
                D3D_DRIVER_TYPE_UNKNOWN,
                nullptr,
                createDeviceFlags,
                featureLevels,
                1,
                D3D11_SDK_VERSION,
                &m_device,
                nullptr,
                &m_context
            );
            if (FAILED(hr))
            {
                std::cerr << "[DDA] D3D11CreateDevice failed hr=" << std::hex << hr << std::endl;
                SafeRelease(&output);
                SafeRelease(&adapter);
                SafeRelease(&factory);
                return hr;
            }
        }

        hr = output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&m_output1);
        if (FAILED(hr))
        {
            std::cerr << "[DDA] QueryInterface(IDXGIOutput1) failed hr=" << std::hex << hr << std::endl;
            SafeRelease(&m_context);
            SafeRelease(&m_device);
            SafeRelease(&output);
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }

        hr = m_output1->DuplicateOutput(m_device, &m_duplication);
        if (FAILED(hr))
        {
            std::cerr << "[DDA] DuplicateOutput failed hr=" << std::hex << hr << std::endl;
            SafeRelease(&m_output1);
            SafeRelease(&m_context);
            SafeRelease(&m_device);
            SafeRelease(&output);
            SafeRelease(&adapter);
            SafeRelease(&factory);
            return hr;
        }

        m_duplication->GetDesc(&m_duplDesc);

        DXGI_OUTPUT_DESC oDesc{};
        output->GetDesc(&oDesc);
        outScreenWidth = oDesc.DesktopCoordinates.right - oDesc.DesktopCoordinates.left;
        outScreenHeight = oDesc.DesktopCoordinates.bottom - oDesc.DesktopCoordinates.top;

        SafeRelease(&output);
        SafeRelease(&adapter);
        SafeRelease(&factory);

        if (outDevice)  *outDevice = m_device;
        if (outContext) *outContext = m_context;

        return hr;
    }

    HRESULT AcquireFrame(FrameContext& frameCtx, UINT timeout = 100)
    {
        frameCtx.texture = nullptr;
        frameCtx.hasAcquiredFrame = false;
        if (!m_duplication) return E_FAIL;

        DXGI_OUTDUPL_FRAME_INFO frameInfo{};
        IDXGIResource* resource = nullptr;

        HRESULT hr = m_duplication->AcquireNextFrame(timeout, &frameInfo, &resource);
        if (FAILED(hr)) return hr;

        frameCtx.hasAcquiredFrame = true;
        m_frameAcquired = true;

        if (resource)
        {
            hr = resource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&frameCtx.texture);
            resource->Release();
        }
        return hr;
    }

    void ReleaseFrame()
    {
        if (!m_duplication || !m_frameAcquired)
            return;

        m_duplication->ReleaseFrame();
        m_frameAcquired = false;
    }

    void Release()
    {
        if (m_duplication)
        {
            ReleaseFrame();
            m_duplication->Release();
            m_duplication = nullptr;
        }
        SafeRelease(&m_output1);
        SafeRelease(&m_context);
        SafeRelease(&m_device);
    }

public:
    ID3D11Device* m_device;
    ID3D11DeviceContext* m_context;
    IDXGIOutputDuplication* m_duplication;
    IDXGIOutput1* m_output1;
    DXGI_OUTDUPL_DESC m_duplDesc;
    bool m_frameAcquired;
};

DuplicationAPIScreenCapture::DuplicationAPIScreenCapture(int desiredWidth, int desiredHeight, int monitorIndex)
    : d3dDevice(nullptr)
    , d3dContext(nullptr)
    , deskDupl(nullptr)
    , output1(nullptr)
    , sharedTexture(nullptr)
    , stagingTextureCPU(nullptr)
    , screenWidth(0)
    , screenHeight(0)
    , regionWidth(desiredWidth)
    , regionHeight(desiredHeight)
{
    m_ddaManager = std::make_unique<DDAManager>();

    HRESULT hr = m_ddaManager->Initialize(
        monitorIndex,
        regionWidth,
        regionHeight,
        screenWidth,
        screenHeight,
        &d3dDevice,
        &d3dContext
    );
    if (FAILED(hr))
    {
        std::cerr << "[DDA] DDAManager Initialize failed hr=0x" << std::hex << hr << std::endl;
        return;
    }

    regionWidth = std::clamp(regionWidth, 1, std::max(1, screenWidth));
    regionHeight = std::clamp(regionHeight, 1, std::max(1, screenHeight));

    initialized_ = createStagingTextureCPU();
    if (!initialized_)
        return;
#ifdef USE_CUDA
    createCudaInteropTexture();
#endif
}

DuplicationAPIScreenCapture::~DuplicationAPIScreenCapture()
{
#ifdef USE_CUDA
    releaseCudaInteropTexture();
#endif
    if (m_ddaManager)
    {
        m_ddaManager->Release();
        m_ddaManager.reset();
    }
    SafeRelease(&stagingTextureCPU);
    SafeRelease(&sharedTexture);

    d3dDevice = nullptr;
    d3dContext = nullptr;
    deskDupl = nullptr;
    output1 = nullptr;
}

cv::Mat DuplicationAPIScreenCapture::GetNextFrameCpu()
{
    if (!m_ddaManager || !m_ddaManager->m_duplication || !stagingTextureCPU)
        return cv::Mat();

    FrameContext frameCtx;
    HRESULT hr = m_ddaManager->AcquireFrame(frameCtx, 5);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT)
    {
        return cv::Mat();
    }
    else if (hr == DXGI_ERROR_ACCESS_LOST ||
        hr == DXGI_ERROR_DEVICE_RESET ||
        hr == DXGI_ERROR_DEVICE_REMOVED ||
        hr == DXGI_ERROR_INVALID_CALL)
    {
        capture_method_changed.store(true);
        return cv::Mat();
    }
    else if (FAILED(hr))
    {
        std::cerr << "[DuplicationAPIScreenCapture] AcquireNextFrame (CPU) failed hr=0x"
            << std::hex << hr << std::endl;
        if (frameCtx.hasAcquiredFrame)
            m_ddaManager->ReleaseFrame();
        return cv::Mat();
    }

    if (!frameCtx.texture)
    {
        if (frameCtx.hasAcquiredFrame)
            m_ddaManager->ReleaseFrame();
        return cv::Mat();
    }

    const int copyWidth = std::min(regionWidth, std::max(1, screenWidth));
    const int copyHeight = std::min(regionHeight, std::max(1, screenHeight));
    const int left = std::max(0, (screenWidth - copyWidth) / 2);
    const int top = std::max(0, (screenHeight - copyHeight) / 2);

    D3D11_BOX sourceRegion;
    sourceRegion.left = left;
    sourceRegion.top = top;
    sourceRegion.front = 0;
    sourceRegion.right = sourceRegion.left + copyWidth;
    sourceRegion.bottom = sourceRegion.top + copyHeight;
    sourceRegion.back = 1;

    d3dContext->CopySubresourceRegion(
        stagingTextureCPU,
        0,
        0, 0, 0,
        frameCtx.texture,
        0,
        &sourceRegion
    );

    m_ddaManager->ReleaseFrame();
    frameCtx.texture->Release();

    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hrMap = d3dContext->Map(stagingTextureCPU, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hrMap))
    {
        std::cerr << "[DDA] Map stagingTextureCPU failed hr=" << std::hex << hrMap << std::endl;
        if (hrMap == DXGI_ERROR_DEVICE_REMOVED || hrMap == DXGI_ERROR_DEVICE_RESET)
            capture_method_changed.store(true);
        return cv::Mat();
    }

    cv::Mat cpuFrame(regionHeight, regionWidth, CV_8UC4);
    for (int y = 0; y < regionHeight; y++)
    {
        unsigned char* dstRow = cpuFrame.ptr<unsigned char>(y);
        unsigned char* srcRow = (unsigned char*)mapped.pData + y * mapped.RowPitch;
        memcpy(dstRow, srcRow, regionWidth * 4);
    }

    d3dContext->Unmap(stagingTextureCPU, 0);
    return cpuFrame;
}

#ifdef USE_CUDA
static void SetGpuCaptureStatus(GpuCaptureStatus* status, GpuCaptureStatus value)
{
    if (status)
        *status = value;
}

bool DuplicationAPIScreenCapture::GetNextFrameGpu(cv::cuda::GpuMat& gpuFrameBgra, GpuCaptureStatus* status)
{
    if (!m_ddaManager || !m_ddaManager->m_duplication || !interopTextureGPU || !cudaInteropResource || !cudaInteropReady)
    {
        SetGpuCaptureStatus(status, GpuCaptureStatus::NotReady);
        return false;
    }

    FrameContext frameCtx;
    HRESULT hr = m_ddaManager->AcquireFrame(frameCtx, 5);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT)
    {
        SetGpuCaptureStatus(status, GpuCaptureStatus::Timeout);
        return false;
    }
    else if (hr == DXGI_ERROR_ACCESS_LOST ||
        hr == DXGI_ERROR_DEVICE_RESET ||
        hr == DXGI_ERROR_DEVICE_REMOVED ||
        hr == DXGI_ERROR_INVALID_CALL)
    {
        capture_method_changed.store(true);
        SetGpuCaptureStatus(status, GpuCaptureStatus::DeviceLost);
        return false;
    }
    else if (FAILED(hr))
    {
        std::cerr << "[DuplicationAPIScreenCapture] AcquireNextFrame (GPU) failed hr=0x"
            << std::hex << hr << std::endl;
        if (frameCtx.hasAcquiredFrame)
            m_ddaManager->ReleaseFrame();
        SetGpuCaptureStatus(status, GpuCaptureStatus::AcquireFailed);
        return false;
    }

    if (!frameCtx.texture)
    {
        if (frameCtx.hasAcquiredFrame)
            m_ddaManager->ReleaseFrame();
        SetGpuCaptureStatus(status, GpuCaptureStatus::MissingTexture);
        return false;
    }

    const int copyWidth = std::min(regionWidth, std::max(1, screenWidth));
    const int copyHeight = std::min(regionHeight, std::max(1, screenHeight));
    const int left = std::max(0, (screenWidth - copyWidth) / 2);
    const int top = std::max(0, (screenHeight - copyHeight) / 2);

    D3D11_BOX sourceRegion;
    sourceRegion.left = left;
    sourceRegion.top = top;
    sourceRegion.front = 0;
    sourceRegion.right = sourceRegion.left + copyWidth;
    sourceRegion.bottom = sourceRegion.top + copyHeight;
    sourceRegion.back = 1;

    d3dContext->CopySubresourceRegion(
        interopTextureGPU,
        0,
        0, 0, 0,
        frameCtx.texture,
        0,
        &sourceRegion
    );

    m_ddaManager->ReleaseFrame();
    frameCtx.texture->Release();

    cudaError_t cuErr = cudaGraphicsMapResources(1, &cudaInteropResource, 0);
    if (cuErr != cudaSuccess)
    {
        std::cerr << "[DDA] cudaGraphicsMapResources failed: " << cudaGetErrorString(cuErr) << std::endl;
        cudaInteropReady = false;
        SetGpuCaptureStatus(status, GpuCaptureStatus::CudaMapFailed);
        return false;
    }

    cudaArray_t mappedArray = nullptr;
    cuErr = cudaGraphicsSubResourceGetMappedArray(&mappedArray, cudaInteropResource, 0, 0);
    if (cuErr != cudaSuccess)
    {
        std::cerr << "[DDA] cudaGraphicsSubResourceGetMappedArray failed: " << cudaGetErrorString(cuErr) << std::endl;
        cudaGraphicsUnmapResources(1, &cudaInteropResource, 0);
        cudaInteropReady = false;
        SetGpuCaptureStatus(status, GpuCaptureStatus::CudaArrayFailed);
        return false;
    }

    gpuFrameBgra.create(regionHeight, regionWidth, CV_8UC4);

    cuErr = cudaMemcpy2DFromArray(
        gpuFrameBgra.ptr<unsigned char>(),
        gpuFrameBgra.step,
        mappedArray,
        0, 0,
        static_cast<size_t>(regionWidth) * 4,
        static_cast<size_t>(regionHeight),
        cudaMemcpyDeviceToDevice
    );

    cudaError_t unmapErr = cudaGraphicsUnmapResources(1, &cudaInteropResource, 0);
    if (unmapErr != cudaSuccess)
    {
        std::cerr << "[DDA] cudaGraphicsUnmapResources failed: " << cudaGetErrorString(unmapErr) << std::endl;
        cudaInteropReady = false;
    }

    if (cuErr != cudaSuccess)
    {
        std::cerr << "[DDA] cudaMemcpy2DFromArray failed: " << cudaGetErrorString(cuErr) << std::endl;
        cudaInteropReady = false;
        SetGpuCaptureStatus(status, GpuCaptureStatus::CudaCopyFailed);
        return false;
    }

    SetGpuCaptureStatus(status, GpuCaptureStatus::Captured);
    return true;
}

bool DuplicationAPIScreenCapture::createCudaInteropTexture()
{
    if (!d3dDevice)
        return false;

    releaseCudaInteropTexture();

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = regionWidth;
    desc.Height = regionHeight;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    HRESULT hr = d3dDevice->CreateTexture2D(&desc, nullptr, &interopTextureGPU);
    if (FAILED(hr))
    {
        std::cerr << "[DDA] CreateTexture2D(interop) failed hr=" << std::hex << hr << std::endl;
        return false;
    }

    cudaError_t cuErr = cudaGraphicsD3D11RegisterResource(
        &cudaInteropResource,
        interopTextureGPU,
        cudaGraphicsRegisterFlagsNone
    );

    if (cuErr != cudaSuccess)
    {
        std::cerr << "[DDA] cudaGraphicsD3D11RegisterResource failed: "
            << cudaGetErrorString(cuErr) << std::endl;
        SafeRelease(&interopTextureGPU);
        cudaInteropResource = nullptr;
        cudaInteropReady = false;
        return false;
    }

    cudaInteropReady = true;
    return true;
}

void DuplicationAPIScreenCapture::releaseCudaInteropTexture()
{
    if (cudaInteropResource)
    {
        cudaError_t cuErr = cudaGraphicsUnregisterResource(cudaInteropResource);
        if (cuErr != cudaSuccess)
        {
            std::cerr << "[DDA] cudaGraphicsUnregisterResource failed: "
                << cudaGetErrorString(cuErr) << std::endl;
        }
        cudaInteropResource = nullptr;
    }

    SafeRelease(&interopTextureGPU);
    cudaInteropReady = false;
}
#endif

bool DuplicationAPIScreenCapture::createStagingTextureCPU()
{
    if (!d3dDevice) return false;

    SafeRelease(&stagingTextureCPU);

    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = regionWidth;
    desc.Height = regionHeight;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;

    HRESULT hr = d3dDevice->CreateTexture2D(&desc, nullptr, &stagingTextureCPU);
    if (FAILED(hr))
    {
        std::cerr << "[DDA] CreateTexture2D(staging) failed hr=" << std::hex << hr << std::endl;
        return false;
    }
    return true;
}
