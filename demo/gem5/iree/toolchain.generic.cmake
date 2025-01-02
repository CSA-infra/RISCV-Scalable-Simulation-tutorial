# Script derived and adapted from this source:
# https://kubasejdak.com/how-to-cross-compile-for-embedded-with-cmake-like-a-champ

set(CMAKE_SYSTEM_NAME               Generic)
set(CMAKE_SYSTEM_PROCESSOR          ${TOOLCHAIN_TARGET})

# Without that flag CMake is not able to pass test compilation check
set(CMAKE_TRY_COMPILE_TARGET_TYPE   STATIC_LIBRARY)

set(CMAKE_AR                        ${TOOLCHAIN_PATH}/bin/${TOOLCHAIN_PREFIX}ar${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_ASM_COMPILER              ${TOOLCHAIN_PATH}/bin/${TOOLCHAIN_PREFIX}gcc${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_C_COMPILER                ${TOOLCHAIN_PATH}/bin/${TOOLCHAIN_PREFIX}gcc${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_CXX_COMPILER              ${TOOLCHAIN_PATH}/bin/${TOOLCHAIN_PREFIX}g++${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_LINKER                    ${TOOLCHAIN_PATH}/bin/${TOOLCHAIN_PREFIX}ld${CMAKE_EXECUTABLE_SUFFIX})
set(CMAKE_OBJCOPY                   ${TOOLCHAIN_PATH}/bin/${TOOLCHAIN_PREFIX}objcopy${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_RANLIB                    ${TOOLCHAIN_PATH}/bin/${TOOLCHAIN_PREFIX}ranlib${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_SIZE                      ${TOOLCHAIN_PATH}/bin/${TOOLCHAIN_PREFIX}size${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")
set(CMAKE_STRIP                     ${TOOLCHAIN_PATH}/bin/${TOOLCHAIN_PREFIX}strip${CMAKE_EXECUTABLE_SUFFIX} CACHE INTERNAL "")

set(CMAKE_C_FLAGS                   "-static -Wno-psabi -fdata-sections -ffunction-sections -Wl,--gc-sections" CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS                 "${CMAKE_C_FLAGS} -fno-exceptions" CACHE INTERNAL "")

set(CMAKE_C_FLAGS_DEBUG             "-Os -g" CACHE INTERNAL "")
set(CMAKE_C_FLAGS_RELEASE           "-Os -DNDEBUG" CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS_DEBUG           "${CMAKE_C_FLAGS_DEBUG}" CACHE INTERNAL "")
set(CMAKE_CXX_FLAGS_RELEASE         "${CMAKE_C_FLAGS_RELEASE}" CACHE INTERNAL "")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
