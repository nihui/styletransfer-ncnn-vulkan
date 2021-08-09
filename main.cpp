
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// ncnn
#include "gpu.h"
#include "net.h"

#include "styletransfer.id.h"
#include "styletransfer.param.bin.h"

static int styletransfer(const ncnn::Net& net, const cv::Mat& bgr, cv::Mat& outbgr)
{
    const int w = bgr.cols;
    const int h = bgr.rows;

    const int target_size = 1000;
    int target_w = w;
    int target_h = h;
    if (w < h)
    {
        target_h = target_size;
        target_w = target_size * w / h;
    }
    else
    {
        target_w = target_size;
        target_h = target_size * h / w;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, target_w, target_h);

    ncnn::Mat out;
    {
        ncnn::Extractor ex = net.create_extractor();

        ex.input(styletransfer_param_id::BLOB_input1, in);
        ex.extract(styletransfer_param_id::BLOB_output1, out);
    }

    outbgr.create(out.h, out.w, CV_8UC3);
    out.to_pixels(outbgr.data, ncnn::Mat::PIXEL_RGB2BGR);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat bgr = cv::imread(imagepath, 1);
    if (bgr.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    ncnn::create_gpu_instance();

    {
        ncnn::Option opt;
        opt.use_vulkan_compute = true;

        ncnn::Net styletransfernet[4];

        // load
        const char* model_paths[4] = {"candy.bin", "rain_princess.bin", "udnie.bin", "starrynight.bin"};
        for (int i = 0; i < 4; i++)
        {
            styletransfernet[i].opt = opt;

            int ret0 = styletransfernet[i].load_param(styletransfer_param_bin);
            int ret1 = styletransfernet[i].load_model(model_paths[i]);

            fprintf(stderr, "load %d %d\n", ret0, ret1);
        }

        // process and save
        #pragma omp parallel for num_threads(2)
        for (int i = 0; i < 4; i++)
        {
            cv::Mat outbgr;
            styletransfer(styletransfernet[i], bgr, outbgr);

            char outpath[256];
            sprintf(outpath, "%s.%d.jpg", imagepath, i);
            cv::imwrite(outpath, outbgr);
        }
    }

    ncnn::destroy_gpu_instance();

    return 0;
}
