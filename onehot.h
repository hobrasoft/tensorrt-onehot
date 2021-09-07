#ifndef __ONE_HOT_LAYER__H_
#define __ONE_HOT_LAYER__H_

#include "NvInferPlugin.h"
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>
#include <sstream>
#include <string>

#include <cuda.h>
#include <cuda_fp16.h>

namespace nvinfer1
{
namespace plugin
{

using namespace nvinfer1;

class OneHotLayer : public IPluginV2DynamicExt {
  public:
   ~OneHotLayer() override = default;
    OneHotLayer();
    OneHotLayer(int32_t axis);
    OneHotLayer(const PluginFieldCollection *fc);
    OneHotLayer(const void* buffer, size_t size);
    OneHotLayer(const Weights* weights, int nbWeights);

    void setName(const char *name);
  
    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) override;
  
    int getNbOutputs() const override;
    DimsExprs getOutputDimensions(int index, const DimsExprs* inputs, int nbInputDims, IExprBuilder& exprBuilder) override;
  
    int initialize() override;
    void terminate() override;
    void destroy() override;
  
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc *output, int32_t nbOutputs) const override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;
  
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
  
    void configurePlugin(DynamicPluginTensorDesc const *in,  int32_t nbInputs,
                         DynamicPluginTensorDesc const *out, int32_t nbOutputs) override;

    const char *getPluginType() const override;
    const char *getPluginVersion() const override;
    void setPluginNamespace(const char *pluginNamespace) override;
    const char *getPluginNamespace() const override;
    DataType getOutputDataType(int index, const DataType *inputTypes, int nbInputs) const override;
    // bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;
    // bool canBroadcastInputAcrossBatch(int inputIndex) const override;
    IPluginV2DynamicExt *clone() const override;

    typedef enum {
        STATUS_SUCCESS = 0,
        STATUS_FAILURE = 1,
        STATUS_BAD_PARAM = 2,
        STATUS_NOT_SUPPORTED = 3,
        STATUS_NOT_INITIALIZED = 4
        } PluginExitStatus;

  protected:
    Dims2 mInputDim;
    Dims3 mOutputDim;
    int32_t mDepth;
    int32_t mAxis;
    float   mOnValue;
    float   mOffValue;
    std::string mNamespace;
    std::string mName;
};


class OneHotLayerCreator : public IPluginCreator {
  public:
    OneHotLayerCreator();
   ~OneHotLayerCreator() override = default;

    const char *getPluginName() const override;
    const char *getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char *libNamespace) override;
    const char *getPluginNamespace() const override;

  private:
    static PluginFieldCollection mFC;

    // Parameters for DetectionOutput
    DetectionOutputParameters params;
    static std::vector<PluginField> mPluginAttributes;

    std::string mNamespace;
};


}

}

#endif
