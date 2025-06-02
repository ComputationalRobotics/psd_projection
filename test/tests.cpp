#include <gtest/gtest.h>

int main(int argc, char **argv)
{
    // Execute all the included tests
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}