# Runs a command and checks its exact exit code and (optionally) a stderr regex.
# A crash (signal) never matches a numeric EXPECT_CODE, unlike CTest's WILL_FAIL.
#   cmake -DEXPECT_CODE=2 [-DEXPECT_STDERR=<regex>] -DCMD="prog;arg1;arg2" -P expect_exit.cmake
if(NOT DEFINED EXPECT_CODE OR NOT DEFINED CMD)
  message(FATAL_ERROR "expect_exit.cmake needs -DEXPECT_CODE and -DCMD")
endif()
execute_process(COMMAND ${CMD} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
if(NOT "${rc}" STREQUAL "${EXPECT_CODE}")
  message(FATAL_ERROR "expected exit code ${EXPECT_CODE}, got '${rc}'\nstdout: ${out}\nstderr: ${err}")
endif()
if(DEFINED EXPECT_STDERR AND NOT "${err}" MATCHES "${EXPECT_STDERR}")
  message(FATAL_ERROR "stderr does not match '${EXPECT_STDERR}'\nstderr: ${err}")
endif()
