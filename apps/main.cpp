#include <iostream>
#include <vector>
#include <cmath>

#include "astra/Globals.h"
#include "astra/VolumeGeometry2D.h"
#include "astra/ProjectionGeometry2D.h"
#include "astra/Projector2D.h"
#include "astra/FilteredBackProjectionAlgorithm.h"
#include "astra/Data2D.h"
#include "astra/SheppLogan.h"
#include "astra/ParallelBeamLineKernelProjector2D.h"
#include "astra/DataProjectorPolicies.h"
#include "astra/Filters.h"
int main() {
    std::cout << "=== FBP with Factory Functions ===" << std::endl;

    try {
        // 1. 创建几何结构
        std::cout << "1. Creating geometries..." << std::endl;

        // 体积几何
        int volumeSize = 256;
        astra::CVolumeGeometry2D vol_geom(volumeSize, volumeSize);

        // 投影几何（平行束）
        int detectorCount = 384;
        int angleCount = 180;
        std::vector<float> angles(angleCount);
        for (int i = 0; i < angleCount; ++i) {
            angles[i] = static_cast<float>(i * 3.14 / angleCount);
        }
        astra::CParallelProjectionGeometry2D proj_geom(
			angleCount, detectorCount, 1.0f, angles.data()
        );

        std::cout << "   Volume: " << volumeSize << "x" << volumeSize << std::endl;
        std::cout << "   Projections: " << angleCount << " angles, " << detectorCount << " detectors" << std::endl;
       
        std::cout << "2. Creating data objects using factory functions..." << std::endl;

        astra::CFloat32VolumeData2D* volume_data = astra::createCFloat32VolumeData2DMemory(vol_geom);
        astra::generateSheppLogan(volume_data, true);  // false = 标准 Shepp-Logan

        std::cout << "   Created Shepp-Logan phantom" << std::endl;

        
        astra::CFloat32ProjectionData2D* projection_data = astra::createCFloat32ProjectionData2DMemory(proj_geom);

        std::cout << "   Created projection data container" << std::endl;

    
        std::cout << "3. Performing forward projection..." << std::endl;
        astra::CParallelBeamLineKernelProjector2D projector(proj_geom, vol_geom);
        astra::DefaultFPPolicy forwardPolicy(volume_data, projection_data);
        projector.project(forwardPolicy);
        std::cout << "   Forward projection completed" << std::endl;
        // 4. 创建重建结果容器
        std::cout << "4. Creating reconstruction container..." << std::endl;
        astra::CFloat32VolumeData2D* reconstruction_data = astra::createCFloat32VolumeData2DMemory(vol_geom);

        // 5. 创建并运行 FBP 算法
        std::cout << "5. Running FBP reconstruction..." << std::endl;
        astra::CFilteredBackProjectionAlgorithm fbp_algorithm;
        
        bool init_success = fbp_algorithm.initialize(&projector, reconstruction_data, projection_data);
        if (!init_success) {
            std::cerr << "Failed to initialize FBP algorithm!" << std::endl;
            return 1;
        }

        bool run_success = fbp_algorithm.run(1);
        if (!run_success) {
            std::cerr << "FBP reconstruction failed!" << std::endl;
            return 1;
        }
        std::cout << "   FBP reconstruction completed" << std::endl;
        // 6. 保存结果
        std::cout << "6. Saving results..." << std::endl;
        const float* phantom_ptr = volume_data->getFloat32Memory();
        const float* sino_ptr = projection_data->getFloat32Memory();
        const float* recon_ptr = reconstruction_data->getFloat32Memory();

        if (phantom_ptr) {
            FILE* file = fopen("phantom.raw", "wb");
            fwrite(phantom_ptr, sizeof(float), volume_data->getSize(), file);
            fclose(file);
            std::cout << "Saved phantom.raw, size: " << volume_data->getSize() << std::endl;
        }

        if (sino_ptr) {
            FILE* file = fopen("sinogram.raw", "wb");
            fwrite(sino_ptr, sizeof(float), projection_data->getSize(), file);
            fclose(file);
            std::cout << "Saved sinogram.raw, size: " << projection_data->getSize() << std::endl;
        }

        if( recon_ptr) {
            FILE* file = fopen("reconstruction.raw", "wb");
            fwrite(recon_ptr, sizeof(float), reconstruction_data->getSize(), file);
            fclose(file);
            std::cout << "Saved reconstruction.raw, size: " << reconstruction_data->getSize() << std::endl;
		}
        // 8. 清理内存
        std::cout << "8. Cleaning up..." << std::endl;
        delete volume_data;
        delete projection_data;
        delete reconstruction_data;

        std::cout << "=== Test Completed Successfully ===" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}