#include <gtest/gtest.h>

#include "iterative_TF16_test.hpp"
#include "express_FP32_test.hpp"
#include "express_FP32_Lt_test.hpp"
#include "lopbcg_test.hpp"
#include "lanczos_test.hpp"
#include "benchmarks.hpp"

int main(int argc, char **argv)
{
    // Execute all the included tests
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}