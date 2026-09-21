set(DLSIM_PYARROW_DIR "/home/yanggl/anaconda3/lib/python3.11/site-packages/pyarrow" CACHE PATH "pyarrow package directory providing Arrow/Parquet C++")
set(DLSIM_ARROW_SOVERSION "2100" CACHE STRING "Arrow shared library version suffix")

add_executable(dlsim-import main.cpp)
target_include_directories(dlsim-import SYSTEM PRIVATE ${DLSIM_PYARROW_DIR}/include)
target_link_libraries(dlsim-import PRIVATE dlsim_core
  ${DLSIM_PYARROW_DIR}/libparquet.so.${DLSIM_ARROW_SOVERSION}
  ${DLSIM_PYARROW_DIR}/libarrow.so.${DLSIM_ARROW_SOVERSION})
set_target_properties(dlsim-import PROPERTIES BUILD_RPATH ${DLSIM_PYARROW_DIR} INSTALL_RPATH ${DLSIM_PYARROW_DIR})

add_executable(dlsim-perfdata-report report_main.cpp)
target_link_libraries(dlsim-perfdata-report PRIVATE dlsim_core)
