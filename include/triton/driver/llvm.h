#include <string>
#include "triton/driver/dispatch.h"

namespace llvm{
class Module;
}

namespace triton{
namespace driver{

void init_llvm();
std::string path_to_ptxas(int& version);
std::string llir_to_ptx(llvm::Module* module, int cc, int version);
std::string ptx_to_cubin(const std::string& ptx, const std::string& ptxas_path, int cc);
CUmodule ptx_to_cumodule(const std::string& ptx, int cc);
std::tuple<std::string, std::string> llir_to_amdgcn(llvm::Module* module, const std::string& proc);
hipModule_t amdgpu_to_hipmodule(const std::string& path);

}
}

#define STRINGIFY_HELPER(X) #X
#define STRINGIFY(X) STRINGIFY_HELPER(X)

