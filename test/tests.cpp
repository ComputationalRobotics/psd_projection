#include <gtest/gtest.h>

#include "composite_FP32_test.hpp"
#include "lopbcg_test.hpp"
#include "lanczos_test.hpp"

int main(int argc, char **argv)
{
    // Execute all the included tests
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}