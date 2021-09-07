#include "onehot.h"
#include "onehotkernel.h"
#include <cstring>


// CHANGE THIS TO YOUR DEPTH
#define DEPTH 36


using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace {
const char *ONEHOT_PLUGIN_NAME{"OneHot"};
const char *ONEHOT_PLUGIN_VERSION{"1"};
}

REGISTER_TENSORRT_PLUGIN(OneHotLayerCreator);

PluginFieldCollection OneHotLayerCreator::mFC{};
std::vector<PluginField> OneHotLayerCreator::mPluginAttributes;

#define ASSERT(assertion)                                                                                              \
    {                                                                                                                  \
        if (!(assertion))                                                                                              \
        {                                                                                                              \
            std::cerr << "#assertion" << __FILE__ << "," << __LINE__ << std::endl;                                     \
            abort();                                                                                                   \
        }                                                                                                              \
    }

const char *toString(const nvinfer1::DataType type) {
    switch (type) {
        case DataType::kFLOAT: return "kFLOAT";
        case DataType::kHALF:  return "kHALF";
        case DataType::kINT8:  return "kINT8";
        case DataType::kINT32: return "kINT32";
        case DataType::kBOOL:  return "kBOOL";
        }
    return "unknown";
}


const char *toString(const nvinfer1::TensorFormat format) {
    switch (format) {
        case TensorFormat::kLINEAR:     return "kLINEAR";
        case TensorFormat::kCHW2:       return "kCHW2";
        case TensorFormat::kHWC8:       return "kHWC8";
        case TensorFormat::kCHW4:       return "kCHW4";
        case TensorFormat::kCHW16:      return "kCHW16";
        case TensorFormat::kCHW32:      return "kCHW32";
        }
    return "unknown";
}


std::string toString(const Dims& dims) {
    std::string output = "[";
    std::string delimiter = "";
    for (int i=0; i<dims.nbDims; i++) {
        output += delimiter;
        output += std::to_string(dims.d[i]);
        delimiter=",";
        }
    output += "]";
    return output;
}

std::string toString(const Dims* dims, const int nbDims) {
    std::string output = "(";
    std::string delimiter = "";
    if (nbDims == 0) { return "[]"; }
    for (int i=0; i<nbDims; i++) {
        output += delimiter;
        output += toString(dims[i]);
        delimiter = "";
        }
    output += ")";
    return output;
}


OneHotLayer::OneHotLayer() {
    mDepth = DEPTH;
    mAxis = -1;
    // std::cout << "OneHotLayer::OneHotLayer() mDepth=" << mDepth << std::endl;
}


OneHotLayer::OneHotLayer(const PluginFieldCollection *fc) {
    mDepth = DEPTH;
    mAxis = -1;
    // std::cout << "OneHotLayer::OneHotLayer(fc) mDepth=" << mDepth << std::endl;
    for (int i=0; i<fc->nbFields; i++) {
        if (!strcmp(fc->fields[i].name, "axis")) {
            ASSERT(fc->fields[i].type == PluginFieldType::kINT32);
            mAxis = static_cast<int>(*static_cast<const int*>(fc->fields[i].data));
            }
        }
}


OneHotLayer::OneHotLayer(int32_t axis) {
    mDepth = DEPTH;
    mAxis = axis;
    // std::cout << "OneHotLayer::OneHotLayer(axis) mDepth=" << mDepth << " axis=" << axis << std::endl;
}



// Used for deserialized data
OneHotLayer::OneHotLayer(const void* buffer, size_t size) {
    mDepth = DEPTH;
    mAxis = -1;
    // std::cout << "OneHotLayer::OneHotLayer(const void* buffer, size_t size)" << std::endl;
}


OneHotLayer::OneHotLayer(const Weights* weights, int nbWeights) {
    mDepth = DEPTH;
    ASSERT(weights[0].type == DataType::kINT32);
    ASSERT(weights[0].count == 1);
    mDepth = *(reinterpret_cast<const int32_t*>(weights[0].values));
    // std::cout << "OneHotLayer::OneHotLayer(weights, nbWeights) mDepth=" << mDepth << nbWeights << std::endl;

    ASSERT(weights[1].type == DataType::kFLOAT);
    ASSERT(weights[1].count == 1);
    mOnValue = *(reinterpret_cast<const float*>(weights[1].values));

    ASSERT(weights[2].type == DataType::kFLOAT);
    ASSERT(weights[2].count == 1);
    mOffValue = *(reinterpret_cast<const float*>(weights[2].values));

    // std::cout << "mDepth=" << mDepth << std::endl;
    // std::cout << "mOnValue=" << mOnValue << std::endl;
    // std::cout << "mOffValue=" << mOffValue << std::endl;
}

int OneHotLayer::getNbOutputs() const {
    // std::cout << "OneHotLayer::getNbOutputs" << std::endl;
    return 1;
}


bool OneHotLayer::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs)  {
    return true;
}


// index = index výstupního tensoru
// inputs[0] = indices, input tensor            kINT32
// inputs[1] = depth = scalar                   kInt32
// inputs[2] = values [off_value, on_value]     kFLOAT, kHALF
DimsExprs OneHotLayer::getOutputDimensions(int index, const DimsExprs* inputs, int nbInputDims, IExprBuilder& exprBuilder) {
/*
    std::cout << "OneHotLayer::getOutputDimensions()" << mName << std::endl;

    for (int i=0; i<nbInputDims; i++) {    
        // std::cout << "    input (" << toString(inputs[i].desc.dims) << ") " <<  toString(inputs[i].desc.type) << " " << toString(inputs[i].desc.format) << std::endl;
        std::cout << "    input nbDims=" << inputs[i].nbDims << " size=" << inputs[i].d[0]->getConstantValue()  << " constant: " << (inputs[i].d[0]->isConstant() ? "true" : "false")  << std::endl;
        }
*/
    ASSERT(nbInputDims == 3);
    ASSERT(inputs[0].nbDims == 1); 
    ASSERT(inputs[1].nbDims == 1); 
    ASSERT(inputs[2].nbDims == 1); 
    ASSERT(inputs[0].d[0]->getConstantValue() == 1);
    ASSERT(inputs[1].d[0]->getConstantValue() == 1);
    ASSERT(inputs[2].d[0]->getConstantValue() == 2);

    DimsExprs x;
    x.nbDims = 2;
    x.d[0] = exprBuilder.constant(1);
    x.d[1] = exprBuilder.constant(DEPTH);
    x.d[2] = nullptr;
    return x;
}


const char* OneHotLayer::getPluginType() const {
    // std::cout << "OneHotLayer::getPluginType() " << ONEHOT_PLUGIN_NAME << std::endl;
    return ONEHOT_PLUGIN_NAME;
}


const char* OneHotLayer::getPluginVersion() const {
    // std::cout << "OneHotLayer::getPluginType() " << ONEHOT_PLUGIN_VERSION << std::endl;
    return ONEHOT_PLUGIN_VERSION;
}


int OneHotLayer::initialize() {
    // std::cout << "OneHotLayer::initialize" << std::endl;
    return STATUS_SUCCESS;
}


void OneHotLayer::terminate() {
    // std::cout << "OneHotLayer::terminate" << std::endl;
}


void OneHotLayer::destroy() {
    // std::cout << "OneHotLayer::destroy" << std::endl;
    delete this;
}


IPluginV2DynamicExt *OneHotLayer::clone() const {
    // std::cout << "OneHotLayer::clone" << std::endl;
    OneHotLayer *obj= new OneHotLayer(mAxis);
    obj->setPluginNamespace(mNamespace.c_str());
    obj->setName(mName.c_str());
    return obj;
}


// inputs[0] = indices, input tensor            kINT32
// inputs[1] = depth = scalar                   kInt32
// inputs[2] = values [off_value, on_value]     kFLOAT
int OneHotLayer::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                         const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) {
//  std::cout << "OneHotLayer::enqueue()" << std::endl;

/*
    int nbInputs = 3;
    for (int i=0; i<nbInputs; i++) {    
        std::cout << "    input (" << toString(inputDesc[i].dims) << ") " <<  toString(inputDesc[i].type) << " " << toString(inputDesc[i].format) << std::endl;
        }

    int nbOutputs = 1;
    for (int i=0; i<nbOutputs; i++) {
        std::cout << "   output (" << toString(outputDesc[i].dims) << ") " <<  toString(outputDesc[i].type) << " " << toString(outputDesc[i].format) << std::endl;
        }
*/

    switch (inputDesc[2].type) {
        case DataType::kFLOAT:  {
            int32_t lIndex;
            float lOutput[DEPTH];
            computeOneHot((const int32_t *)inputs[0], (const int32_t *)inputs[1], (const float *)inputs[2], (float *)outputs[0], stream); 
            cudaMemcpy(&lIndex,  inputs[1],  sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lOutput, outputs[0], sizeof(float)*DEPTH, cudaMemcpyDeviceToHost);
            /*
            std::cout << "    ";
            for (int i=0; i<DEPTH; i++) { std::cout << ((lOutput[i] < 0.5) ? "." : "0"); }
            std::cout << std::endl;
            */
            break;
            }
        case DataType::kHALF: {
            int32_t lIndex;
            half lOutput[DEPTH];
            computeOneHot((const int32_t *)inputs[0], (const int32_t *)inputs[1], (const half *)inputs[2], (half *)outputs[0], stream); 
            cudaMemcpy(&lIndex,  inputs[1],  sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lOutput, outputs[0], sizeof(half)*DEPTH, cudaMemcpyDeviceToHost);
            /*
            std::cout << "    ";
            for (int i=0; i<DEPTH; i++) { std::cout << ((lOutput[i] < 0.5) ? "." : "0"); }
            std::cout << std::endl;
            */
            break;
            }
        default:
            std::cout << "Unsupported data type in enqueue() " << toString(inputDesc[2].format) << std::endl;
            abort();
            break;
        }


    return 0;
}


size_t OneHotLayer::getSerializationSize() const {
    std::cout << "OneHotLayer::getSerializationSize" << std::endl;
    return 0;
}


void OneHotLayer::serialize(void* buffer) const {
    std::cout << "OneHotLayer::serialize" << std::endl;
    // char *d = reinterpret_cast<char*>(buffer);
    // write(d, mAxis);
}


size_t OneHotLayer::getWorkspaceSize(const PluginTensorDesc* inputDesc, int32_t nbInputs, const PluginTensorDesc *output, int32_t nbOutputs) const {
    size_t size;
    ASSERT(nbInputs == 3);
    switch (inputDesc[2].type) {
        case DataType::kFLOAT:  
            size = DEPTH*sizeof(float);
            break;
        case DataType::kHALF:   
            size = DEPTH*sizeof(half);
            break;
        default: 
            std::cout << "Unsupported data type in getWorkspaceSize() " << toString(inputDesc[2].type) << std::endl;
            abort();
            break;
        }
    std::cout << "OneHotLayer::getWorkspaceSize() " << size << std::endl;
    return size;
}


void OneHotLayer::setPluginNamespace(const char *pluginNamespace) {
    // std::cout << "OneHotLayer::setPluginNamespace(" << pluginNamespace << ")" << std::endl;
    mNamespace = pluginNamespace;
}


const char *OneHotLayer::getPluginNamespace() const {
    // std::cout << "OneHotLayer::getPluginNamespace" << std::endl;
    return mNamespace.c_str();
}


DataType OneHotLayer::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
    ASSERT(index == 0);
    /*
    std::cout << "OneHotLayer::getOutputDataType(" << index << ", inputTypes, " <<  nbInputs << "): " << toString(inputTypes[2]) << std::endl;
    for (int i=0; i<nbInputs; i++) {
        std::cout << "    i: " << i << " datatype: " << toString(inputTypes[i]) << std::endl;
        }
    */
    return inputTypes[2];
}


void OneHotLayer::configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out, int32_t nbOutputs) {
    /*
    std::cout << "OneHotLayer::configurePlugin()" << std::endl;

    for (int i=0; i<nbInputs; i++) {    
        std::cout << "    input (" << toString(in[i].desc.dims) << ") " <<  toString(in[i].desc.type) << " " << toString(in[i].desc.format) << std::endl;
        }

    for (int i=0; i<nbOutputs; i++) {
        std::cout << "   output (" << toString(out[i].desc.dims) << ") " <<  toString(out[i].desc.type) << " " << toString(out[i].desc.format) << std::endl;
        }
    */
//  mInputDim = Dims2(inputs[0].d[0], inputs[0].d[1]);
//  mOutputDim = Dims3(1, inputs[0].d[1], mDepth);
}


void OneHotLayer::setName(const char *name) {
    mName = name;
}

/************************************************************************************************************************************************/
OneHotLayerCreator::OneHotLayerCreator() {
//    std::cout << "OneHotLayerCreator::OneHotLayerCreator()" << std::endl;

    mPluginAttributes.emplace_back(PluginField("axis", nullptr,      PluginFieldType::kINT32,   1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();

}


nvinfer1::IPluginV2Ext *OneHotLayerCreator::createPlugin(const char *name, const PluginFieldCollection *fc) {
    std::cout << "OneHotLayerCreator::createPlugin() " << name << " nb:" << fc->nbFields << std::endl;
    OneHotLayer *obj = new OneHotLayer(fc);
    obj->setName(name);
    return obj;
}


nvinfer1::IPluginV2Ext* OneHotLayerCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) {
    std::cout << "OneHotLayerCreator::deserializePlugin()" << std::endl;
    // abort();
    OneHotLayer* obj = new OneHotLayer(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    obj->setName(name);
    return obj;
}


const char *OneHotLayerCreator::getPluginName() const {
    // std::cout << "OneHotLayerCreator::getPluginName() " << ONEHOT_PLUGIN_NAME << std::endl;
    return ONEHOT_PLUGIN_NAME;
}


const char *OneHotLayerCreator::getPluginVersion() const {
    // std::cout << "OneHotLayerCreator::getPluginVersion() " << ONEHOT_PLUGIN_VERSION << std::endl;
    return ONEHOT_PLUGIN_VERSION;
}


void OneHotLayerCreator::setPluginNamespace(const char *libNamespace) {
    // std::cout << "OneHotLayerCreator::setPluginNamespace()" << std::endl;
    mNamespace = libNamespace;
}


const char *OneHotLayerCreator::getPluginNamespace() const {
    // std::cout << "OneHotLayerCreator::getPluginNamespace() -" << mNamespace.c_str() << "-" << std::endl;
    return mNamespace.c_str();
}

const PluginFieldCollection* OneHotLayerCreator::getFieldNames() {
    // std::cout << "OneHotLayerCreator::getFieldNames()" << std::endl;
    return &mFC;
}


