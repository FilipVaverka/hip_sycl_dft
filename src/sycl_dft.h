#pragma once

#ifndef __SYCL_DFT_H__
#define __SYCL_DFT_H__

#include <cstdint>
#include <vector>
#include <SYCL/sycl.hpp>

namespace sycl::dft {

typedef struct SYCL_DFT_DESC * SYCL_DFT_DESC_HANLDE;

enum class precision {
    SINGLE,
    DOUBLE
};

enum class domain {
    REAL,
    COMPLEX
};

enum class config_param {
    FORWARD_DOMAIN,
    DIMENSION,
    LENGHTS,
    PRECISION,

    /*FORWARD_SCALE,
    BACKWARD_SCALE,*/

    NUMBER_OF_TRANSFORMS,

    /*COMPLEX_STORAGE,
    REAL_STORAGE,
    CONJUGATE_EVEN_STORAGE,*/

    PLACEMENT,

    INPUT_STRIDES,
    OUTPUT_STRIDES,

    FWD_DISTANCE,
    BWD_DISTANCE,

    WORKSPACE,
    /*ORDERING,
    TRANSPOSE,
    PACKET_FORMAT,*/
    COMMIT_STATUS
};

enum class config_value {
    COMMITED,
    UNCOMMITED,

    COMPLEX_COMPLEX,
    REAL_COMPLEX,
    //REAL_REAL,

    INPLACE,
    NOT_INPLACE,

    /*ORDERED,
    BACKWARD_SCRAMBLED,*/

    /*ALLOW,
    AVOID,
    NONE,

    CCE_FORMAT*/
};

namespace detail {

typedef sycl::buffer<uint8_t, 1> WorkBufferType_t;

#define SYCL_DFT_PARAM_NAME_INFO_(F) \
    F(dft_param, FORWARD_DOMAIN, config_value) \
    F(dft_param, DIMENSION, int) \
    F(dft_param, LENGHTS, std::vector<size_t>) \
    F(dft_param, PRECISION, precision) \
    F(dft_param, NUMBER_OF_TRANSFORMS, int) \
    F(dft_param, PLACEMENT, config_value) \
    F(dft_param, INPUT_STRIDES, std::vector<size_t>) \
    F(dft_param, OUTPUT_STRIDES, std::vector<size_t>) \
    F(dft_param, FWD_DISTANCE, size_t) \
    F(dft_param, BWD_DISTANCE, size_t) \
    F(dft_param, WORKSPACE, WorkBufferType_t) \
    F(dft_param, COMMIT_STATUS, config_value)

template<typename EnumType, sycl::dft::config_param name>
struct ParamTraits_t {};

#define SYCL_DFT_DECLARE_PARAM_TRAITS_(token, paramName, T) \
    struct token; \
    template<>    \
    struct ParamTraits_t<detail::token, sycl::dft::config_param::paramName> \
    { \
        enum { value = int(sycl::dft::config_param::paramName) }; \
        typedef T paramType; \
    };

SYCL_DFT_PARAM_NAME_INFO_(SYCL_DFT_DECLARE_PARAM_TRAITS_)

}

template<typename descriptor_type, typename data_type, int D>
sycl::event compute_forward(descriptor_type            &desc,
                            sycl::buffer<data_type, D> &inout);

template<typename descriptor_type, typename input_type, typename output_type, int D>
sycl::event compute_forward(descriptor_type              &desc,
                            sycl::buffer<input_type, D>  &in,
                            sycl::buffer<output_type, D> &out);

template<typename descriptor_type, typename data_type, int D>
sycl::event compute_backward(descriptor_type            &desc,
                             sycl::buffer<data_type, D> &inout);

template<typename descriptor_type, typename input_type, typename output_type, int D>
sycl::event compute_backward(descriptor_type              &desc,
                             sycl::buffer<input_type, D>  &in,
                             sycl::buffer<output_type, D> &out);

template<sycl::dft::precision prec, sycl::dft::domain dom>
class descriptor 
{
public:
    descriptor(std::int64_t length);
    descriptor(std::vector<std::int64_t> dimensions);
    ~descriptor();

    template<config_param name>
    typename detail::ParamTraits_t<detail::dft_param, name>::paramType
    get_value() const
    {
        typename detail::ParamTraits_t<detail::dft_param, name>::paramType param;
        get_value(name, param);

        return param;
    }

    template<config_param name>
    void set_value(const typename detail::ParamTraits_t<detail::dft_param, name>::paramType &value)
    {
        set_value(name, value);
    }

    template<typename T>
    void set_value(config_param param, const T &value);

    template<typename T>
    void get_value(config_param param, T &value) const;

    void commit(sycl::queue &queue);

    template<typename descriptor_type, typename data_type, int D>
    friend sycl::event compute_forward(descriptor_type            &desc,
                                    sycl::buffer<data_type, D> &inout);

    template<typename descriptor_type, typename input_type, typename output_type, int D>
    friend sycl::event compute_forward(descriptor_type              &desc,
                                    sycl::buffer<input_type, D>  &in,
                                    sycl::buffer<output_type, D> &out);

    template<typename descriptor_type, typename data_type, int D>
    friend sycl::event compute_backward(descriptor_type            &desc,
                                        sycl::buffer<data_type, D> &inout);

    template<typename descriptor_type, typename input_type, typename output_type, int D>
    friend sycl::event compute_backward(descriptor_type              &desc,
                                        sycl::buffer<input_type, D>  &in,
                                        sycl::buffer<output_type, D> &out);

private:
    SYCL_DFT_DESC_HANLDE handle;
};

}

#endif // __SYCL_DFT_H__